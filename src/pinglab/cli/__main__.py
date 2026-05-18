"""Oscilloscope -- PING network toolkit.

CLI entrypoint with subcommands: sim, image, video, train.

Usage:
    uv run python src/pinglab/oscilloscope.py                             # sim only (metrics)
    uv run python src/pinglab/oscilloscope.py sim                         # sim only (metrics)
    uv run python src/pinglab/oscilloscope.py image                       # snapshot
    uv run python src/pinglab/oscilloscope.py video --scan-var dt         # dt sweep video
    uv run python src/pinglab/oscilloscope.py train --epochs 10           # train on scikit digits
"""

from __future__ import annotations

import json
import math
import sys
import time as _time
from pathlib import Path

# Ensure src/pinglab/ is FIRST on sys.path so the sibling top-level modules
# (models, inputs, metrics, config, plot) are importable as bare names.
_pkg_dir = str(Path(__file__).resolve().parent.parent)  # src/pinglab/
if _pkg_dir in sys.path:
    sys.path.remove(_pkg_dir)
sys.path.insert(0, _pkg_dir)

import logging
import numpy as np
import torch
from torch import nn

log = logging.getLogger("oscilloscope")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

import models as M
from inputs import (
    make_spike_drive,
    make_step_drive,
    make_reference_noise,
    make_step_drive_from_ref,
    DT_CAL,
)
from config import (
    Config,
    cfg,
    _MODEL_CLASSES,
    N_E,
    N_I,
    FPS,
    SEED,
    SIM_MS,
    BURN_IN_MS,
    T_E_ASYNC_DEFAULT,
    SIGMA_E,
    STEP_ON_MS,
    STEP_OFF_MS,
    W_EI,
    W_IE,
    NOISE_SIGMA,
    NOISE_TAU,
    SPIKE_RATE_BASE,
    ARTIFACT_ROOT,
    W_IN_SPIKES,
    W_IN_SPARSITY,
    BIAS,
    EI_RATIO,
    DEVICE,
    patch_dt,
    run_sim,
    run_sim_batch,
    run_sim_image,
    _extract_records,
    extract_weights,
    make_net,
    make_ping_net,
    _run_sim_with_net,
    build_net,
    build_config,
    _sync_globals_from_cfg,
)
from metrics import (
    report_metrics,
    metrics_str,
    compute_metrics,
    format_metrics,
)
from plot import (
    prof,
    make_transient_fig,
    draw_transient_frame,
    reset_weight_xlims,
    LAYOUT_PRESETS,
    PANEL_CATALOG,
    ACTIVE_PANELS,
    CLR,
)


# =============================================================================
# =============================================================================
# Image → spike encoding with stimulus window
# =============================================================================


from cli.encoders import (  # noqa: E402,F401
    EVAL_SEED,
    FROZEN_MODES,
    FrozenEncoder,
    downsample_spikes_count,
    encode_batch,
    encode_image_spikes,
    encode_images_poisson,
    encode_smnist,
    transport_spikes_bin,
    upsample_spikes_zeropad,
)



from cli.scan import (  # noqa: E402
    SCAN_DEFAULTS,
    _SWEEP_XLABELS,
    _WEIGHT_SCAN_VARS,
    _apply_scan_var,
    _auto_device,
    _scan_dt,
    _scan_od_batched,
    _scan_streaming,
    generate_scan,
    primary_hid_key,
    primary_inh_key,
)




# =============================================================================
# Snapshot generators
# =============================================================================

from cli.snapshot import (  # noqa: E402
    generate_image_snapshot,
    generate_sim_only,
    generate_snapshot,
    generate_spike_snapshot,
)

from cli.datasets import (  # noqa: E402,F401
    DATASET_N_HIDDEN_DEFAULTS,
    SHD_N_CHANNELS,
    _load_dataset_image,
    _load_shd,
    _shd_cache_dir,
    load_dataset,
)



# =============================================================================
# Training (moved to oscilloscope/train.py)
# =============================================================================

from cli.train import (  # noqa: E402
    BATCH_SIZE,
    GRAD_CLIP,
    observe_epoch,
    seed_everything,
    train,
)



# =============================================================================
# CLI
# =============================================================================


from cli.video import (  # noqa: E402
    _plot_dt_sweep,
    _render_dt_sweep_video,
)
from cli.infer import infer  # noqa: E402


def _apply_from_dir(args, argv):
    """Fill infer args from a training run's config.json + weights.pth.

    Only sets values the user didn't explicitly pass on the CLI.
    """
    from_dir = Path(args.from_dir)
    cfg_path = from_dir / "config.json"
    if not cfg_path.exists():
        print(f"Error: {cfg_path} not found")
        sys.exit(1)
    cfg = json.loads(cfg_path.read_text())

    # Resolve legacy model names from earlier refactors.
    from config import LEGACY_MODEL_ALIASES

    if cfg.get("model") in LEGACY_MODEL_ALIASES:
        old = cfg["model"]
        cfg["model"] = LEGACY_MODEL_ALIASES[old]
        print(f"  legacy model name {old!r} → {cfg['model']!r}")

    # Auto-detect weights
    if args.load_weights is None:
        weights_path = from_dir / "weights.pth"
        if not weights_path.exists():
            print(f"Error: {weights_path} not found (pass --load-weights explicitly)")
            sys.exit(1)
        args.load_weights = str(weights_path)

    # Auto-set out-dir to a sibling infer/ dir if not specified
    if args.out_dir is None:
        args.out_dir = str(from_dir / "infer")

    # Map config.json keys → argparse dest names.
    # Only apply if the user didn't explicitly pass the flag.
    _CONFIG_TO_ARGS = {
        "model": "model",
        "dt": "dt",
        "t_ms": "t_ms",
        "dataset": "dataset",
        "hidden_sizes": "n_hidden",
        "ei_strength": "ei_strength",
        "ei_ratio": "ei_ratio",
        "sparsity": "sparsity",
        "w_in_sparsity": "w_in_sparsity",
        "w_in": "w_in",
        "input_rate": "spike_rate",
        "max_samples": "max_samples",
        "kaiming_init": "kaiming_init",
        "dales_law": "dales_law",
        "w_rec": "w_rec",
        "rec_layers": "rec_layers",
        "ei_layers": "ei_layers",
        "seed": "seed",
    }
    # Backwards compat: old config.json has "n_hidden" as int
    if "n_hidden" in cfg and "hidden_sizes" not in cfg:
        val = cfg["n_hidden"]
        if isinstance(val, int):
            cfg["hidden_sizes"] = [val]
        elif isinstance(val, list):
            cfg["hidden_sizes"] = val

    # Build set of flags explicitly passed on CLI
    explicit = set()
    for a in argv:
        if a.startswith("--"):
            explicit.add(a.split("=")[0])

    # Reverse lookup: argparse dest → CLI flag name
    _DEST_TO_FLAG = {
        "model": "--model",
        "dt": "--dt",
        "t_ms": "--t-ms",
        "dataset": "--dataset",
        "n_hidden": "--n-hidden",
        "ei_strength": "--ei-strength",
        "ei_ratio": "--ei-ratio",
        "sparsity": "--ei-sparsity",
        "w_in_sparsity": "--w-in-sparsity",
        "w_in": "--w-in",
        "spike_rate": "--input-rate",
        "max_samples": "--max-samples",
        "kaiming_init": "--kaiming-init",
        "dales_law": "--dales-law",
        "w_rec": "--w-rec",
        "hidden_sizes": "--n-hidden",
        "rec_layers": "--rec-layers",
        "ei_layers": "--ei-layers",
        "seed": "--seed",
    }

    inherited = []
    for cfg_key, dest in _CONFIG_TO_ARGS.items():
        if cfg_key not in cfg or cfg[cfg_key] is None:
            continue
        flag = _DEST_TO_FLAG.get(dest, f"--{dest}")
        if flag in explicit or flag.replace("--", "--no-") in explicit:
            continue
        val = cfg[cfg_key]
        setattr(args, dest, val)
        inherited.append(f"{cfg_key}={val}")

    if inherited:
        print(f"  from-dir: inherited {', '.join(inherited)}")

    # Warn about critical flags missing from config.json (old training runs)
    critical = ["kaiming_init", "dales_law"]
    missing = [k for k in critical if k not in cfg]
    if missing:
        print(
            f"  WARNING: config.json missing {missing} — this training run "
            f"predates these flags. Pass them explicitly on the CLI or retrain."
        )


def parse_args():
    """Parse command-line arguments with subparsers for sim/image/video/train."""
    import argparse

    _examples = """\
Each subcommand has its own complete argument listing. The top-level help
above only shows the dispatcher; for the actual flags accepted by a mode,
run:

  python -m cli sim    --help
  python -m cli image  --help
  python -m cli video  --help
  python -m cli train  --help
  python -m cli infer  --help

The flags fall into the following groups (every group is documented in
each subcommand's --help):

  Network        --model, --n-hidden, --n-input, --ei-strength, --ei-ratio,
                 --ei-sparsity, --w-in-sparsity, --bias, --dt, --t-ms,
                 --burn-in, --tau-mem, --tau-syn, --device, --seed
  Readout        --readout {rate,li,spike-count,mem-mean}, --readout-tau-out,
                 --readout-w-out-scale, --kaiming-init, --dales-law,
                 --no-dales-law, --rec-layers, --ei-layers
  Input          --input, --input-rate, --stim-overdrive, --drive, --dataset,
                 --digit, --sample
  Weights        --w-in, --w-ee, --w-ei, --w-ie, --w-rec, --trainable-w-ee
  Slow + ALIF    --slow-syn, --tau-nmda, --slow-syn-gain,
                 --alif, --tau-adapt, --alif-beta
  Gradient       --v-grad-dampen, --sgcc, --sgcc-alpha, --grad-clip,
                 --surrogate-slope, --coba-integrator
  Train (train)  --lr, --epochs, --batch-size, --max-samples, --optimizer,
                 --loss, --adaptive-lr, --early-stopping, --observe,
                 --observe-every, --frame-rate, --profile,
                 --fr-reg-lower-theta, --fr-reg-lower-strength,
                 --fr-reg-upper-theta, --fr-reg-upper-strength,
                 --skip-bad-grad-threshold
  Image (image)  --fake-progress
  Scan (video)   --scan-var, --scan-min, --scan-max, --frames, --frame-rate,
                 --resample-input
  Infer (infer)  --load-weights, --from-dir, --dt-sweep, --eval-encoder,
                 --frozen-inputs-mode
  Output / exec  --out-dir, --wipe-dir, --raster, --layout, --panels,
                 --modal, --modal-gpu

Examples:
  python -m cli                                    # sim (metrics only)
  python -m cli image                              # snapshot (ping default)
  python -m cli video --scan-var ei_strength       # sweep E-I coupling
  python -m cli video --scan-var spike_rate --scan-min 5 --scan-max 100
  python -m cli train --epochs 100 --observe video
  python -m cli train --epochs 100 --sgcc --sgcc-alpha 0.5
  python -m cli train --epochs 100 --slow-syn --trainable-w-ee
  python -m cli image --input dataset --dataset mnist --digit 3
  python -m cli infer --load-weights weights.pth --dt 0.5
  python -m cli infer --from-dir runs/foo --dt-sweep 0.05 0.1 0.25 0.5

Models:
  ping        COBANet with E↔I coupling. With --ei-strength > 0 the
              recurrent inhibitory loop is wired up and frozen at init;
              feedforward weights train against this fixed substrate.
  cuba        COBANet with --ei-strength 0 (E cells only, no I-loop).
              The articles/models page calls this "coba" — naming is for
              CLI-vs-pedagogy reasons.
  standard-snn   Dimensionless mem = β·mem + I from snnTorch tutorial 5.
                 Not dt-invariant; β is a fitted hyperparameter.
  snntorch-library   External snnTorch reference path; uses the library's
                     Leaky/Synaptic primitives directly.

For the underlying theory of the gradient-stabilisation flags
(--v-grad-dampen, --sgcc) see /articles/art006/.
"""
    parser = argparse.ArgumentParser(
        description="Oscilloscope — PING network toolkit",
        epilog=_examples,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Shared parent for network/input args
    parent = argparse.ArgumentParser(add_help=False)
    net_group = parent.add_argument_group("Network")
    net_group.add_argument(
        "--model",
        type=str,
        default="ping",
        choices=list(_MODEL_CLASSES.keys()),
        help="Model to simulate (default: ping)",
    )
    net_group.add_argument(
        "--n-hidden",
        type=int,
        nargs="+",
        default=None,
        help="Hidden layer sizes. Single value = 1 layer, "
        "multiple = stacked layers (e.g. --n-hidden 128 256). "
        "(default: dataset-aware smart default; "
        "scikit=64, mnist=1024, smnist=32)",
    )
    net_group.add_argument(
        "--kaiming-init",
        action="store_true",
        help="Use plain nn.Linear Kaiming uniform init "
        "(signed weights, no fan-in normalization). "
        "Matches canonical snnTorch tutorial setup. "
        "Only applies to standard-snn / cuba; "
        "--w-in is ignored when this is set.",
    )
    net_group.add_argument(
        "--readout",
        choices=["rate", "li", "spike-count", "mem-mean"],
        default="rate",
        dest="readout_mode",
        help="Output layer: 'rate' sums last-hidden spikes "
        "and projects linearly at the final timestep "
        "(default); 'li' uses a non-spiking leaky "
        "integrator per class with max-over-time, the "
        "standard SHD-style readout.",
    )
    net_group.add_argument(
        "--dales-law",
        action="store_true",
        default=True,
        help="Enforce Dale's law: clamp weights to non-negative (default: True)",
    )
    net_group.add_argument(
        "--no-dales-law",
        dest="dales_law",
        action="store_false",
        help="Allow signed (positive + negative) weights (standard-snn / cuba only)",
    )
    net_group.add_argument(
        "--rec-layers",
        type=int,
        nargs="+",
        default=None,
        help="Which hidden layers get recurrence (1-indexed). "
        "Default: all layers when --w-rec is set.",
    )
    net_group.add_argument(
        "--ei-layers",
        type=int,
        nargs="+",
        default=None,
        help="Which hidden layers get E-I structure (1-indexed). "
        "Default: all layers (PING only).",
    )
    net_group.add_argument(
        "--n-input", type=int, default=None, help="N_IN input neurons (default: N_E)"
    )
    net_group.add_argument(
        "--ei-strength",
        type=float,
        default=0.5,
        help="E-I coupling: sets W_EI=s, W_IE=s*ratio (default: 0.5)",
    )
    net_group.add_argument(
        "--ei-ratio", type=float, default=2.0, help="W_IE/W_EI ratio (default: 2.0)"
    )
    net_group.add_argument(
        "--ei-sparsity",
        type=float,
        default=None,
        dest="sparsity",
        help="E-I connection sparsity (default: 0.2)",
    )
    net_group.add_argument(
        "--w-in-sparsity",
        type=float,
        default=None,
        help="W_in sparsity (default: 0.95)",
    )
    net_group.add_argument(
        "--bias",
        type=float,
        default=None,
        help="Background conductance to E neurons in uS",
    )
    net_group.add_argument(
        "--dt",
        type=float,
        default=0.25,
        help="Integration timestep in ms (default: 0.25)",
    )
    net_group.add_argument(
        "--t-ms",
        type=float,
        default=200.0,
        help="Total simulation duration in ms (default: 200). "
        "For synthetic-step modes, must exceed STEP_ON_MS "
        "(default 200) so the stimulus window is reached; "
        "values <= STEP_ON_MS leave the trial in flat baseline.",
    )
    net_group.add_argument(
        "--burn-in",
        type=float,
        default=None,
        help="Burn-in period in ms hidden from cli plots/videos. "
        "Default: 100 ms in shared cfg (legacy); pass 0 to make the entire "
        "trial visible.",
    )
    net_group.add_argument(
        "--tau-mem",
        type=float,
        default=None,
        help="Membrane time constant τ_mem in ms "
        "(default: 10 ms, module-level `tau_snn`). "
        "Cramer et al. SHD: 20 ms.",
    )
    net_group.add_argument(
        "--tau-syn",
        type=float,
        default=None,
        help="Synaptic time constant τ_syn in ms for the "
        "exponential synapse (default: 2 ms, "
        "module-level `tau_ampa`). Cramer et al. SHD: "
        "10 ms. Only affects models with "
        "exponential_synapse=True (coba / ping).",
    )
    net_group.add_argument(
        "--readout-tau-out",
        type=float,
        default=None,
        help="Output-LIF time constant τ_out in ms for "
        "the spike-count readout (default: 5 ms, "
        "module-level `tau_out_ms`). Smaller values "
        "speed up output-membrane leak so it doesn't "
        "saturate under high-rate hidden drive — "
        "needed for snnTorch-family models at coarse "
        "dt. No-op for --readout rate or li.",
    )
    net_group.add_argument(
        "--readout-w-out-scale",
        type=float,
        default=1.0,
        help="Multiply the readout matrix W_ff[-1] (and "
        "bias b_ff[-1] if present) by this scalar "
        "after build_net. Use to compensate for the "
        "10× lower hidden firing rate of COBANet "
        "models vs CUBANet under mem-mean / "
        "spike-count readouts: bumping W_out by ~10 "
        "equalises the trial-level drive into the "
        "output LIF and recovers gradient signal. "
        "Train-mode only. Default 1.0.",
    )
    net_group.add_argument(
        "--grad-clip",
        type=float,
        default=None,
        help="Global-norm gradient-clip threshold passed to "
        "torch.nn.utils.clip_grad_norm_. Default 1.0 "
        "(M.GRAD_CLIP). Set high (e.g. 100+) to let "
        "Adam handle wildly-scaled BPTT gradients via "
        "its second-moment preconditioner.",
    )
    net_group.add_argument(
        "--surrogate-slope",
        type=float,
        default=None,
        help="Fast-sigmoid surrogate-gradient slope β. "
        "Larger = narrower active window around "
        "threshold. pinglab default 1.0; Cramer et al. "
        "use 40 for SHD RSNNs. Applies to "
        "SurrogateSpike (cuba / standard-snn / ping) "
        "and snntorch-library's fast_sigmoid.",
    )
    net_group.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "mps", "cuda"],
        help="Compute device. If unset, auto-detects: cuda > mps > cpu.",
    )

    inp_group = parent.add_argument_group("Input")
    inp_group.add_argument(
        "--input",
        type=str,
        default="synthetic-spikes",
        choices=["synthetic-conductance", "synthetic-spikes", "dataset"],
        help="Input mode (default: synthetic-spikes)",
    )
    inp_group.add_argument(
        "--input-rate",
        type=float,
        default=25.0,
        dest="spike_rate",
        help="Baseline input rate in Hz (default: 25)",
    )
    inp_group.add_argument(
        "--stim-overdrive",
        type=float,
        default=1.0,
        dest="overdrive",
        help="Stimulus multiplier (default: 1.0)",
    )
    inp_group.add_argument(
        "--drive",
        type=float,
        default=None,
        help="Baseline tonic conductance for synthetic-conductance",
    )
    inp_group.add_argument(
        "--digit", type=int, default=0, help="Digit class for dataset input (0-9)"
    )
    inp_group.add_argument(
        "--sample", type=int, default=0, help="Sample index for dataset input"
    )
    inp_group.add_argument(
        "--dataset",
        type=str,
        default="scikit",
        choices=["scikit", "mnist", "smnist", "shd"],
        help="Dataset (default: scikit)",
    )

    wt_group = parent.add_argument_group("Weights (advanced)")
    wt_group.add_argument(
        "--w-in",
        type=float,
        nargs="+",
        default=None,
        metavar=("MEAN", "STD"),
        help="W_in init mean std (default: 0.3 0.06; standard-snn needs ~10 2 dense)",
    )
    wt_group.add_argument(
        "--w-ee",
        type=float,
        nargs=2,
        default=None,
        metavar=("MEAN", "STD"),
        help="W_EE init (mean std). E→E recurrent excitation; default (0, 0) per Börgers PING.",
    )
    wt_group.add_argument(
        "--w-ei",
        type=float,
        nargs=2,
        default=None,
        metavar=("MEAN", "STD"),
        help="W_EI init (mean std)",
    )
    wt_group.add_argument(
        "--w-ie",
        type=float,
        nargs=2,
        default=None,
        metavar=("MEAN", "STD"),
        help="W_IE init (mean std)",
    )
    wt_group.add_argument(
        "--w-rec",
        type=float,
        nargs=2,
        default=None,
        metavar=("MEAN", "STD"),
        help="W_rec recurrent init (mean std, default: 0 0.1)",
    )
    wt_group.add_argument(
        "--trainable-w-ee",
        action="store_true",
        help="Make COBANet's E→E recurrent matrix gradient-carrying "
        "(default: frozen). W_ei / W_ie stay frozen. Use for working-"
        "memory tasks where the E attractor needs to learn.",
    )
    net_group.add_argument(
        "--slow-syn",
        action="store_true",
        help="COBANet only: add an NMDA-like slow excitatory channel "
        "running in parallel with AMPA on every E-driving projection. "
        "Sample-period spikes leave a long-decay residue (tau_nmda, "
        "default 100 ms) on E neurons via the same W matrices — useful "
        "for working-memory tasks. No new trainable parameters.",
    )
    net_group.add_argument(
        "--tau-nmda",
        type=float,
        default=None,
        help="Decay time constant for the slow synapse in ms "
        "(default: 100 ms, module-level `tau_nmda`). Only relevant "
        "when --slow-syn is set.",
    )
    net_group.add_argument(
        "--slow-syn-gain",
        type=float,
        default=0.5,
        help="Gain on the slow synapse drive relative to the fast "
        "(AMPA) drive (default: 0.5). 0.0 makes the network behaviour "
        "identical to plain COBA. Only relevant when --slow-syn is set.",
    )
    net_group.add_argument(
        "--alif",
        action="store_true",
        help="COBANet only: add a per-neuron slow adaptation variable "
        "that raises the firing threshold proportional to recent "
        "spiking (LSNN-style). tau_adapt default 700 ms — provides a "
        "long output-side timescale complementing slow-syn's "
        "input-side timescale.",
    )
    net_group.add_argument(
        "--tau-adapt",
        type=float,
        default=None,
        help="Adaptation decay time constant in ms "
        "(default: 700 ms, module-level `tau_adapt`). Only relevant "
        "when --alif is set.",
    )
    net_group.add_argument(
        "--alif-beta",
        type=float,
        default=1.7,
        help="ALIF threshold-bump per accumulated spike, in mV "
        "(default: 1.7). 0.0 makes the network behaviour identical "
        "to plain COBA. Only relevant when --alif is set.",
    )
    net_group.add_argument(
        "--sgcc",
        action="store_true",
        help="COBANet only: enable Surrogate Gradients by Costate "
        "Control (Burghi et al. 2024). Scales the gradient on the "
        "voltage↔conductance cross-coupling by --sgcc-alpha, taming "
        "the conductance Jacobian explosion without uniformly damping "
        "all gradients the way --v-grad-dampen does.",
    )
    net_group.add_argument(
        "--sgcc-alpha",
        type=float,
        default=0.5,
        help="SGCC retained fraction of cross-coupling gradient "
        "(default: 0.5). alpha=1.0 is no-op (parity with no SGCC); "
        "alpha=0.0 kills the cross-coupling entirely. Paper uses "
        "alpha ≈ 0.5–0.7.",
    )
    out_group = parent.add_argument_group("Output")
    out_group.add_argument("--out-dir", type=str, default=None, help="Output directory")
    out_group.add_argument(
        "--wipe-dir", action="store_true", help="Clear output directory before run"
    )
    out_group.add_argument(
        "--raster",
        type=str,
        default="scatter",
        choices=["scatter", "imshow"],
        help="Raster style (default: scatter)",
    )
    out_group.add_argument(
        "--layout",
        type=str,
        default="full",
        choices=list(LAYOUT_PRESETS.keys()),
        help="Panel layout (default: full)",
    )
    out_group.add_argument(
        "--panels", type=str, default=None, help="Comma-separated panel names"
    )

    exec_group = parent.add_argument_group("Execution")
    exec_group.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed. Seeds Python, NumPy, and torch "
        "(CPU + CUDA + MPS) before dataset load and "
        "model init. Persisted to config.json.",
    )
    exec_group.add_argument(
        "--modal",
        action="store_true",
        help="Run on Modal.com instead of locally. "
        "Artifacts sync back to --out-dir after completion.",
    )
    exec_group.add_argument(
        "--modal-gpu",
        type=str,
        default="T4",
        choices=["none", "T4", "L4", "A10G", "A100", "H100"],
        help="GPU type for Modal runs (default: T4). Use 'none' for CPU-only.",
    )
    exec_group.add_argument(
        "--coba-integrator",
        type=str,
        default="expeuler",
        choices=["expeuler", "fwd"],
        help="Membrane ODE integrator for COBA/PING "
        "(default: expeuler). 'fwd' falls back to "
        "forward Euler for parity comparisons.",
    )

    subparsers = parser.add_subparsers(
        dest="mode", help="Mode: sim (metrics only) | image | video | train | infer"
    )

    # -- sim subcommand --
    sim_parser = subparsers.add_parser(
        "sim",
        parents=[parent],
        help="Run simulation, report metrics, no plot",
        description="Run a single simulation and report firing-rate metrics "
        "without generating plots or video.",
        epilog="Examples:\n"
        "  oscilloscope.py sim --model ping --ei-strength 0.5\n"
        "  oscilloscope.py sim --model standard-snn --dt 0.1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # -- image subcommand --
    image_parser = subparsers.add_parser(
        "image",
        parents=[parent],
        help="Generate an oscilloscope snapshot image",
        description="Run one forward pass and save a still-image "
        "oscilloscope figure (E/I rasters, weight histograms, PSD).",
        epilog="Examples:\n"
        "  # untrained PING on MNIST digit 3\n"
        "  oscilloscope.py image --model ping --dataset mnist --digit 3\n"
        "  # with trained weights\n"
        "  oscilloscope.py image --from-dir path/to/trained --digit 5",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    image_parser.add_argument(
        "--from-dir",
        type=str,
        default=None,
        help="Load trained weights + inherit config from "
        "a training run directory "
        "(e.g. src/artifacts/calibration/mnist/ping-mnist). "
        "CLI flags override inherited values.",
    )
    image_parser.add_argument(
        "--load-weights",
        type=str,
        default=None,
        help="Path to a weights.pth file (alternative to --from-dir).",
    )
    image_parser.add_argument(
        "--fake-progress",
        type=float,
        default=None,
        help="Overlay a progress-bar indicator at level 0-1 "
        "(for demo / teaching; default: off).",
    )

    # -- video subcommand --
    video_parser = subparsers.add_parser(
        "video",
        parents=[parent],
        help="Sweep a parameter, save oscilloscope video",
        description="Sweep one parameter linearly between --scan-min and "
        "--scan-max over --frames, rendering one frame per value. "
        "Supports trained networks via --from-dir. "
        "Special scan vars: 'digit' iterates dataset classes, "
        "'noise' adds Poisson noise to input.",
        epilog="Examples:\n"
        "  # untrained PING ei_strength sweep (archive reproduction)\n"
        "  oscilloscope.py video --model ping --n-hidden 1024 \\\n"
        "    --scan-var ei_strength --scan-min 0 --scan-max 0.4 \\\n"
        "    --input synthetic-conductance --frames 600 --frame-rate 120\n\n"
        "  # trained network digit tour\n"
        "  oscilloscope.py video --from-dir path/to/trained \\\n"
        "    --input dataset --scan-var digit --scan-min 0 --scan-max 9",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    video_parser.add_argument(
        "--scan-var",
        type=str,
        default="stim-overdrive",
        choices=list(SCAN_DEFAULTS.keys()),
        help="Parameter to sweep. 'digit' iterates "
        "dataset classes; 'noise' adds input "
        "Poisson noise (Hz). See SCAN_DEFAULTS. "
        "(default: stim-overdrive)",
    )
    video_parser.add_argument(
        "--scan-min",
        type=float,
        default=1.0,
        help="Scan start value, in the variable's units "
        "(default: 1.0). For digit: integer class.",
    )
    video_parser.add_argument(
        "--scan-max",
        type=float,
        default=50.0,
        help="Scan end value (default: 50.0). For digit: integer class.",
    )
    video_parser.add_argument(
        "--resample-input",
        action="store_true",
        help="Use a different Poisson seed on each frame "
        "(default: same seed for all frames).",
    )
    video_parser.add_argument(
        "--frames",
        type=int,
        default=10,
        help="Number of video frames to render "
        "(default: 10). For 'digit' scan, overridden "
        "by scan range.",
    )
    video_parser.add_argument(
        "--frame-rate",
        type=int,
        default=10,
        help="Output video frame rate in fps (default: 10).",
    )
    video_parser.add_argument(
        "--from-dir",
        type=str,
        default=None,
        help="Load trained weights + inherit config from "
        "a training run directory. CLI flags override.",
    )
    video_parser.add_argument(
        "--load-weights",
        type=str,
        default=None,
        help="Path to weights.pth (alternative to --from-dir).",
    )

    # -- train subcommand --
    train_parser = subparsers.add_parser(
        "train",
        parents=[parent],
        help="Train an SNN to classify digits",
        description="Train a model on MNIST / smnist / scikit-digits using "
        "surrogate-gradient BPTT. Writes weights.pth, metrics.json, "
        "metrics.jsonl, test_predictions.json plus optional video.",
        epilog="Examples:\n"
        "  # standard-snn canonical tutorial mode (full MNIST)\n"
        "  oscilloscope.py train --model standard-snn --kaiming-init \\\n"
        "    --dataset mnist --epochs 40 --lr 0.01 --adaptive-lr\n\n"
        "  # proper continuous-time CUBA LIF\n"
        "  oscilloscope.py train --model cuba --kaiming-init \\\n"
        "    --dataset mnist --epochs 40 --lr 0.01 --adaptive-lr\n\n"
        "  # PING with gamma oscillation on MNIST\n"
        "  oscilloscope.py train --model ping --dataset mnist \\\n"
        "    --ei-strength 0.5 --v-grad-dampen 1000 --lr 0.0001",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    train_parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Adam learning rate. Biophysical models "
        "(coba/ping) typically need 0.0001, "
        "current-based models 0.01 (default: 0.01).",
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=0,
        help="Number of training epochs. 0 = probe only "
        "(init snapshot, no training). Default: 0.",
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Mini-batch size for DataLoader. Default: 64 (from models.BATCH_SIZE).",
    )
    train_parser.add_argument(
        "--observe",
        type=str,
        default=None,
        choices=["video", "images"],
        help="Save oscilloscope per epoch",
    )
    train_parser.add_argument(
        "--observe-every",
        type=int,
        default=1,
        help="Observe every Nth epoch (default: 1)",
    )
    train_parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit dataset to N samples (smoke test)",
    )
    train_parser.add_argument(
        "--v-grad-dampen",
        type=float,
        default=80.0,
        help="Gradient dampening for COBA membrane",
    )
    train_parser.add_argument(
        "--early-stopping",
        type=int,
        default=None,
        help="Stop after N epochs without improvement",
    )
    train_parser.add_argument(
        "--adaptive-lr",
        action="store_true",
        help="Enable ReduceLROnPlateau scheduler (factor=0.5, patience=5)",
    )
    train_parser.add_argument(
        "--frame-rate", type=int, default=10, help="Video fps for observe (default: 10)"
    )
    train_parser.add_argument(
        "--profile",
        type=str,
        default=None,
        metavar="PATH",
        help="Wrap the 3rd batch of epoch 0 in "
        "torch.profiler and write a Chrome-format "
        "trace JSON to PATH. Skips the first two "
        "batches to avoid allocator/JIT warmup "
        "noise. Training continues normally after.",
    )
    train_parser.add_argument(
        "--fr-reg-lower-theta",
        type=float,
        default=0.0,
        help="Firing-rate reg: lower-bound target spike "
        "count per neuron per trial (θ_l). "
        "Penalty s_l · Σ relu(θ_l − <z_i>)² is "
        "added to the loss. Default 0 = off. "
        "Cramer et al. SHD RSNN: 0.01.",
    )
    train_parser.add_argument(
        "--fr-reg-lower-strength",
        type=float,
        default=0.0,
        help="Strength s_l on the lower-bound firing-"
        "rate regulariser (default 0 = off). "
        "Cramer et al.: 1.0.",
    )
    train_parser.add_argument(
        "--fr-reg-upper-theta",
        type=float,
        default=0.0,
        help="Firing-rate reg: upper-bound target spike "
        "count per neuron per trial (θ_u). "
        "Penalty s_u · Σ relu(<z_i> − θ_u)² is "
        "added to the loss. Default 0 = off. "
        "Cramer et al. SHD RSNN: 100.",
    )
    train_parser.add_argument(
        "--fr-reg-upper-strength",
        type=float,
        default=0.0,
        help="Strength s_u on the upper-bound firing-"
        "rate regulariser (default 0 = off). "
        "Cramer et al.: 0.06.",
    )
    train_parser.add_argument(
        "--skip-bad-grad-threshold",
        type=float,
        default=None,
        help="Skip opt.step() (and zero grads) whenever "
        "the batch's clipped gradient norm is NaN, "
        "inf, or exceeds this threshold. Band-aid "
        "against single exploded batches poisoning "
        "Adam's second-moment estimate mid-run. "
        "Default off.",
    )
    train_parser.add_argument(
        "--optimizer",
        choices=["adam", "adamax"],
        default="adam",
        help="Optimizer. Adamax uses the L∞ norm for "
        "the second moment instead of an EMA of "
        "squared grads, so a single pathological "
        "batch cannot poison the preconditioner. "
        "Canonical SNN choice (Cramer, Zenke).",
    )
    train_parser.add_argument(
        "--loss",
        choices=["ce", "mse"],
        default="ce",
        dest="loss_mode",
        help="Training loss. 'ce' = cross-entropy on "
        "logits (default; expects unnormalised "
        "logits). 'mse' = L2 between logits and "
        "one-hot targets — the SNN-paper standard "
        "(Bohte 2002, Lee 2016) when the readout's "
        "logit scale makes CE softmax saturate.",
    )

    # -- infer subcommand --
    infer_parser = subparsers.add_parser(
        "infer",
        parents=[parent],
        help="Run inference with trained weights (optional dt sweep)",
        description="Evaluate a trained model on the test set. With "
        "--dt-sweep, run inference at each dt value to measure "
        "temporal-resolution stability.",
        epilog="Examples:\n"
        "  # single-dt inference\n"
        "  oscilloscope.py infer --from-dir path/to/trained --dt 0.1\n\n"
        "  # frozen-input dt-stability sweep\n"
        "  oscilloscope.py infer --from-dir path/to/trained \\\n"
        "    --dt-sweep 0.05 0.1 0.25 0.5 1.0 2.0 \\\n"
        "    --frozen-inputs-mode upsample --observe video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    infer_parser.add_argument(
        "--from-dir",
        type=str,
        default=None,
        help="Inherit params from a training run directory "
        "(reads config.json + weights.pth). "
        "CLI flags override inherited values.",
    )
    infer_parser.add_argument(
        "--load-weights",
        type=str,
        default=None,
        help="Path to saved weights.pth (auto-detected when --from-dir is set)",
    )
    infer_parser.add_argument(
        "--max-samples", type=int, default=None, help="Limit dataset to N samples"
    )
    infer_parser.add_argument(
        "--dt-sweep",
        type=float,
        nargs="+",
        default=None,
        metavar="DT",
        help="Run inference at each dt value and produce "
        "a sweep summary (e.g. --dt-sweep 0.05 0.1 0.25 0.5 1.0). "
        "Overrides --dt.",
    )
    infer_parser.add_argument(
        "--observe",
        type=str,
        default=None,
        choices=["video", "image"],
        help="Save oscilloscope visualization. "
        "With --dt-sweep: video = one frame per dt. "
        "Without: image = single snapshot.",
    )
    infer_parser.add_argument(
        "--frozen-inputs-mode",
        type=str,
        default=None,
        choices=list(FROZEN_MODES),
        help="How input spikes are transported across dt, "
        "anchored at train-dt (Parthasarathy et al. "
        "§2.1, §2.3): upsample (count-preserving "
        "zero-pad to finer eval-dt per Fig 1B, "
        "requires eval-dt <= train-dt); downsample "
        "(count-preserving sum-pool to coarser "
        "eval-dt per §2.3, requires eval-dt >= "
        "train-dt); resample (fresh Poisson at each "
        "eval-dt, works in both directions but "
        "re-introduces sampling noise).",
    )

    args = parser.parse_args()
    if args.mode is None:
        parser.print_help()
        sys.exit(0)

    # Apply global model knobs as early as possible so every downstream
    # entrypoint (train, sim, image, video, infer) sees the right integrator.
    M.COBA_INTEGRATOR = args.coba_integrator
    if getattr(args, "surrogate_slope", None) is not None:
        M.SURROGATE_SLOPE = float(args.surrogate_slope)
    if getattr(args, "grad_clip", None) is not None:
        global GRAD_CLIP
        GRAD_CLIP = float(args.grad_clip)
        M.GRAD_CLIP = GRAD_CLIP
    # Override τ_mem / τ_syn before patch_dt() re-derives β_snn and decay_ampa.
    if getattr(args, "tau_mem", None) is not None:
        M.tau_snn = float(args.tau_mem)
    if getattr(args, "tau_syn", None) is not None:
        M.tau_ampa = float(args.tau_syn)
    if getattr(args, "tau_nmda", None) is not None:
        M.tau_nmda = float(args.tau_nmda)
    if getattr(args, "tau_adapt", None) is not None:
        M.tau_adapt = float(args.tau_adapt)
    if getattr(args, "readout_tau_out", None) is not None:
        M.tau_out_ms = float(args.readout_tau_out)

    # --from-dir: inherit training params from config.json, fill unset values
    if args.mode in ("infer", "video", "image") and getattr(args, "from_dir", None):
        _apply_from_dir(args, sys.argv[1:])
    if args.mode == "infer" and not getattr(args, "load_weights", None):
        print("Error: infer requires --load-weights or --from-dir")
        sys.exit(1)

    # Auto-detect: if user explicitly passed --dataset/--digit/--sample but
    # left --input at the default "synthetic-spikes", flip to "dataset". The
    # explicit dataset flags only make sense in dataset input mode, so this
    # avoids the silent footgun where "image --dataset mnist --digit 0" went
    # through the synthetic-spikes branch and ignored the digit.
    def _flag_in_argv(*names):
        for arg in sys.argv[1:]:
            for n in names:
                if arg == n or arg.startswith(n + "="):
                    return True
        return False

    args._input_auto = False
    from_dir_set_dataset = getattr(args, "from_dir", None) and getattr(
        args, "dataset", "scikit"
    ) in ("mnist", "smnist")
    if args.input == "synthetic-spikes" and (
        _flag_in_argv("--dataset", "--digit", "--sample") or from_dir_set_dataset
    ):
        args.input = "dataset"
        args._input_auto = True

    return args


def save_run_artifacts(out_dir, args, mode):
    """Save config.json (with provenance), run.sh, set up logging, print intro."""
    import json
    import logging
    import run_log

    out_dir = Path(out_dir)
    if args.wipe_dir and out_dir.exists():
        import shutil

        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # config.json — with provenance metadata at top
    config = {"mode": mode}
    config.update(run_log.provenance())
    for k, v in vars(args).items():
        if v is not None:
            config[k] = v
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)

    # run.sh
    with open(out_dir / "run.sh", "w") as f:
        f.write("#!/bin/bash\n")
        f.write(" ".join(sys.argv) + "\n")

    # output.log — file handler strips ANSI, stdout keeps it
    log = logging.getLogger("oscilloscope")
    log.setLevel(logging.DEBUG)
    log.handlers.clear()

    class _StripAnsiFormatter(logging.Formatter):
        def format(self, record):
            msg = super().format(record)
            return run_log._strip_ansi(msg)

    fh = logging.FileHandler(out_dir / "output.log", mode="w")
    fh.setFormatter(_StripAnsiFormatter("%(message)s"))
    log.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(sh)

    # Print structured intro
    _print_intro(log, config, args, mode)

    return log


def _print_intro(log, config, args, mode):
    """Group CLI args into sections and print via run_log.print_intro."""
    import run_log

    model = config.get("model", "ping")
    dataset = config.get("dataset", "scikit")

    g = lambda *keys: {k: config[k] for k in keys if k in config}

    sections = {
        "Data": g("dataset", "digit", "sample", "max_samples"),
        "Simulation": {
            "dt": f"{config.get('dt', '?')} ms",
            "t_ms": f"{config.get('t_ms', '?')}",
            "input_rate": f"{config.get('spike_rate', '?')} Hz",
            "input": config.get("input", "?"),
            "burn_in": f"{config.get('burn_in', '?')} ms",
        },
        "Network": {
            "model": model,
            "hidden_sizes": config.get("n_hidden", "?"),
            "kaiming_init": config.get("kaiming_init", False),
            "dales_law": config.get("dales_law", True),
            "ei_strength": config.get("ei_strength"),
            "ei_ratio": config.get("ei_ratio"),
        },
        "Weights": g("w_in", "w_in_sparsity", "w_ei", "w_ie", "w_rec"),
        "Training": g("epochs", "lr", "adaptive_lr", "v_grad_dampen")
        if mode == "train"
        else {},
        "Scan": g("scan_var", "scan_min", "scan_max", "frames", "frame_rate")
        if mode == "video"
        else {},
        "Output": g("out_dir", "observe", "wipe_dir"),
        "Provenance": {
            "git_sha": config.get("git_sha"),
            "device": config.get("device"),
            "run_id": config.get("run_id"),
            "started_at": config.get("started_at"),
        },
    }
    run_log.print_intro(log, mode, model, dataset, sections)


if __name__ == "__main__":
    _t0 = _time.monotonic()

    args = parse_args()
    mode = args.mode

    # --modal: re-dispatch to Modal and exit
    if getattr(args, "modal", False):
        out_dir = args.out_dir
        if out_dir is None:
            out_dir = str(Path(__file__).parent.parent / "artifacts" / "oscilloscope")
        # Rebuild CLI args without --modal / --modal-gpu
        skip = {"--modal"}
        cli_args = []
        argv = sys.argv[1:]
        i = 0
        while i < len(argv):
            if argv[i] == "--modal-gpu":
                i += 2  # skip flag and its value
            elif argv[i] in skip:
                i += 1
            else:
                cli_args.append(argv[i])
                i += 1
        from modal_app import dispatch_to_modal

        dispatch_to_modal(cli_args, out_dir, gpu=args.modal_gpu)
        sys.exit(0)

    # Build and sync config for sim/image/video modes
    if mode != "train":
        c = build_config(args)
        _sync_globals_from_cfg(c)

    import config as C

    # Determine output directory
    out_dir = args.out_dir
    if out_dir is None:
        out_dir = str(Path(__file__).parent.parent / "artifacts" / "oscilloscope")
    out_dir = Path(out_dir)

    # Save run artifacts for all modes
    log = save_run_artifacts(out_dir, args, mode)

    # Create enriched .running marker (PID, start time, run_id, cmd).
    # Deleted on normal exit via atexit hook.
    import run_log as _rl
    import json as _json_marker

    try:
        _cfg = _json_marker.loads((out_dir / "config.json").read_text())
        _rid = _cfg.get("run_id", _rl.run_id())
    except Exception:
        _rid = _rl.run_id()
    _running_marker = _rl.write_running_marker(out_dir, _rid)

    import atexit as _atexit

    def _cleanup_running_marker():
        try:
            _running_marker.unlink(missing_ok=True)
        except Exception:
            pass

    _atexit.register(_cleanup_running_marker)

    # Single source of truth for input Poisson rate and sim duration. Every
    # code path reads M.max_rate_hz and M.T_ms, so setting them here once means
    # all dispatch branches (sim/image/video/train/infer × all input types)
    # respect --input-rate / --t-ms without per-call plumbing. Subfunctions
    # that change dt will recalc M.p_scale and M.T_steps via patch_dt as usual.
    M.max_rate_hz = args.spike_rate
    M.T_ms = args.t_ms
    if args.t_ms <= C.STEP_ON_MS:
        log.warning(
            f"  --t-ms={args.t_ms} <= STEP_ON_MS={C.STEP_ON_MS}: "
            f"the stimulus window never fires within the trial. "
            f"Consider --t-ms >= {C.STEP_OFF_MS:.0f} (STEP_OFF_MS)."
        )

    if args._input_auto:
        log.info(f"  --input auto → dataset (inferred from --dataset/--digit/--sample)")

    if mode == "sim":
        t_e_async = C.T_E_ASYNC_DEFAULT
        log.info(f"Device: {C.DEVICE}")
        generate_sim_only(
            spike_rate=args.spike_rate,
            overdrive=args.overdrive,
            dt=args.dt,
            model_name=args.model,
            input_mode=args.input,
            t_e_async=t_e_async,
        )

    elif mode == "image":
        t_e_async = C.T_E_ASYNC_DEFAULT
        log.info(f"Device: {C.DEVICE}")
        if args.input == "dataset":
            generate_image_snapshot(
                digit_class=args.digit,
                sample_idx=args.sample,
                dt=args.dt,
                dataset=args.dataset,
                overdrive=args.overdrive,
                model_name=args.model,
                load_weights=getattr(args, "load_weights", None),
            )
        elif args.input == "synthetic-spikes":
            generate_spike_snapshot(
                spike_rate=args.spike_rate,
                overdrive=args.overdrive,
                dt=args.dt,
                model_name=args.model,
            )
        else:
            generate_snapshot(
                args.overdrive,
                dt=args.dt,
                fake_progress=args.fake_progress,
                model_name=args.model,
                t_e_async=t_e_async,
            )

    elif mode == "video":
        t_e_async = C.T_E_ASYNC_DEFAULT
        log.info(f"Device: {C.DEVICE}")
        # Emit init_d0s0.png alongside the scan video when input is dataset.
        # Mirrors train mode so video runs have a comparable static reference.
        if args.input == "dataset":
            generate_image_snapshot(
                digit_class=args.digit,
                sample_idx=args.sample,
                dt=args.dt,
                dataset=args.dataset,
                overdrive=args.overdrive,
                model_name=args.model,
                out_filename="init_d0s0.png",
            )
        generate_scan(
            scan_var=args.scan_var,
            scan_min=args.scan_min,
            scan_max=args.scan_max,
            resample_input=args.resample_input,
            n_frames=args.frames,
            t_e_async=t_e_async,
            overdrive=args.overdrive,
            spike_rate=args.spike_rate,
            input_mode=args.input,
            dataset=args.dataset,
            digit_class=args.digit,
            sample_idx=args.sample,
            load_weights=getattr(args, "load_weights", None),
        )

    elif mode == "train":
        w_in = args.w_in or [0.3, 0.06]
        if len(w_in) == 1:
            w_in = [w_in[0], w_in[0] * 0.1]
        train(
            model_name=args.model,
            lr=args.lr,
            epochs=args.epochs,
            dt=args.dt or 0.1,
            observe=args.observe,
            out_dir=str(out_dir),
            device_name=args.device,
            w_in=w_in,
            ei_strength=args.ei_strength,
            ei_ratio=args.ei_ratio,
            sparsity=args.sparsity or 0.0,
            w_in_sparsity=args.w_in_sparsity or 0.0,
            dataset=args.dataset,
            snapshot_init=True,
            snapshot_end=True,
            t_ms=args.t_ms,
            burn_in_ms=args.burn_in if args.burn_in is not None else 20.0,
            hidden_sizes=args.n_hidden,
            max_samples=args.max_samples,
            v_grad_dampen=args.v_grad_dampen,
            early_stopping=args.early_stopping,
            observe_every=args.observe_every,
            adaptive_lr=args.adaptive_lr,
            kaiming_init=args.kaiming_init,
            dales_law=args.dales_law,
            w_rec=args.w_rec,
            rec_layers=args.rec_layers,
            ei_layers=args.ei_layers,
            batch_size=args.batch_size,
            seed=args.seed,
            readout_w_out_scale=args.readout_w_out_scale,
            readout_mode=args.readout_mode,
            fr_reg_lower_theta=args.fr_reg_lower_theta,
            fr_reg_lower_strength=args.fr_reg_lower_strength,
            fr_reg_upper_theta=args.fr_reg_upper_theta,
            fr_reg_upper_strength=args.fr_reg_upper_strength,
            skip_bad_grad_threshold=args.skip_bad_grad_threshold,
            optimizer=args.optimizer,
            loss_mode=args.loss_mode,
            profile_path=args.profile,
            trainable_w_ee=args.trainable_w_ee,
            slow_synapse=args.slow_syn,
            slow_syn_gain=args.slow_syn_gain,
            alif=args.alif,
            alif_beta=args.alif_beta,
            sgcc=args.sgcc,
            sgcc_alpha=args.sgcc_alpha,
        )

    elif mode == "infer":
        w_in = args.w_in or [0.3, 0.06]
        if len(w_in) == 1:
            w_in = [w_in[0], w_in[0] * 0.1]
        infer_kwargs = dict(
            model_name=args.model,
            load_weights=args.load_weights,
            dataset=args.dataset,
            max_samples=args.max_samples,
            t_ms=args.t_ms,
            w_in=w_in,
            ei_strength=args.ei_strength,
            ei_ratio=args.ei_ratio,
            sparsity=args.sparsity or 0.0,
            w_in_sparsity=args.w_in_sparsity or 0.0,
            hidden_sizes=args.n_hidden,
            kaiming_init=args.kaiming_init,
            dales_law=args.dales_law,
            w_rec=args.w_rec,
            rec_layers=args.rec_layers,
            ei_layers=args.ei_layers,
            seed=args.seed,
        )
        if args.dt_sweep:
            dt_values = sorted(args.dt_sweep)
            log.info(f"dt sweep: {dt_values}")

            encoder = None
            frozen_mode = getattr(args, "frozen_inputs_mode", None)
            if frozen_mode is not None:
                # Anchor the reference at the training dt (paper's framing):
                # spikes are generated at dt_ref = args.dt once, then
                # transported to each sweep dt via upsample zero-pad (finer,
                # §2.1 Fig 1B) or downsample sum-pool (coarser, §2.3).
                # resample draws fresh at the target.
                dt_ref = float(args.dt)
                if frozen_mode != "resample":
                    for d in dt_values:
                        if abs(d - dt_ref) < 1e-9:
                            continue
                        if frozen_mode == "upsample" and d > dt_ref + 1e-9:
                            raise ValueError(
                                f"--frozen-inputs-mode upsample requires "
                                f"eval-dt <= train-dt; got dt={d}, "
                                f"dt_ref={dt_ref}"
                            )
                        if frozen_mode == "downsample" and d < dt_ref - 1e-9:
                            raise ValueError(
                                f"--frozen-inputs-mode downsample requires "
                                f"eval-dt >= train-dt; got dt={d}, "
                                f"dt_ref={dt_ref}"
                            )
                        ratio = max(d, dt_ref) / min(d, dt_ref)
                        if abs(ratio - round(ratio)) > 1e-6:
                            raise ValueError(
                                f"--frozen-inputs-mode {frozen_mode} requires "
                                f"integer dt ratios vs train-dt; "
                                f"dt={d}, dt_ref={dt_ref}, ratio={ratio:.4f}"
                            )
                encoder = FrozenEncoder(dt_ref, t_ms=args.t_ms, mode=frozen_mode)
                log.info(
                    f"  frozen inputs: ref dt={dt_ref} (train-dt), mode={frozen_mode}"
                )

            train_dt = float(args.dt)
            base_rate = float(args.spike_rate)

            sweep_results = []
            for sweep_dt in dt_values:
                if encoder is not None:
                    encoder.reset()
                res = infer(
                    dt=sweep_dt, out_dir=None, encode_fn=encoder, **infer_kwargs
                )
                sweep_results.append(
                    {
                        "dt": sweep_dt,
                        "acc": res["acc"],
                        "input_rate": base_rate,
                        "hid_rate_hz": res.get("hid_rate_hz"),
                        "rates_hz": res.get("rates_hz", {}),
                    }
                )
            ref = next((r for r in sweep_results if r["dt"] == train_dt), None)
            ref_acc = ref["acc"] if ref else None

            log.info(f"\n{'=' * 40}")
            log.info(f"dt sweep summary ({args.model}):")
            log.info(f"  {'dt':>8s}  {'acc':>6s}  {'Δacc':>6s}")
            for r in sweep_results:
                delta = f"{r['acc'] - ref_acc:+.1f}%" if ref_acc is not None else ""
                marker = " ←train" if r["dt"] == train_dt else ""
                log.info(f"  {r['dt']:8.4f}  {r['acc']:5.1f}%  {delta:>6s}{marker}")

            sweep_blob = {
                "model": args.model,
                "train_dt": train_dt,
                "input_rate": args.spike_rate,
                "t_ms": args.t_ms,
                "dataset": args.dataset,
                "load_weights": args.load_weights,
                "frozen_inputs_mode": frozen_mode,
                "sweep": sweep_results,
            }
            results_path = out_dir / "results.json"
            with open(results_path, "w") as f:
                json.dump(sweep_blob, f, indent=2)
            log.info(f"  → {results_path}")

            _plot_dt_sweep(sweep_results, train_dt, args.model, out_dir)
            log.info(f"  → {out_dir / 'dt_sweep.png'}")

            if args.observe == "video":
                randomize = not args.kaiming_init
                vid_net = build_net(
                    args.model,
                    w_in=w_in,
                    w_in_sparsity=args.w_in_sparsity or 0.0,
                    ei_strength=args.ei_strength,
                    ei_ratio=args.ei_ratio,
                    sparsity=args.sparsity or 0.0,
                    randomize_init=randomize,
                    kaiming_init=args.kaiming_init,
                    dales_law=args.dales_law,
                    w_rec=args.w_rec,
                    hidden_sizes=args.n_hidden,
                    rec_layers=args.rec_layers,
                    ei_layers=args.ei_layers,
                )
                vid_net.load_state_dict(
                    torch.load(args.load_weights, map_location="cpu"), strict=False
                )
                vid_net.eval()
                _render_dt_sweep_video(
                    vid_net,
                    dt_values,
                    sweep_results,
                    train_dt,
                    args.model,
                    args.dataset,
                    out_dir,
                    frozen_inputs=bool(frozen_mode),
                )
                log.info(f"  → {out_dir / 'dt_sweep.mp4'}")
        else:
            acc = infer(dt=args.dt, out_dir=str(out_dir), **infer_kwargs)["acc"]
            if args.observe:
                import config as C

                C.N_E = M.N_HID
                C.N_I = M.N_INH
                randomize = not args.kaiming_init
                vis_net = build_net(
                    args.model,
                    w_in=w_in,
                    w_in_sparsity=args.w_in_sparsity or 0.0,
                    ei_strength=args.ei_strength,
                    ei_ratio=args.ei_ratio,
                    sparsity=args.sparsity or 0.0,
                    randomize_init=randomize,
                    kaiming_init=args.kaiming_init,
                    dales_law=args.dales_law,
                    w_rec=args.w_rec,
                    hidden_sizes=args.n_hidden,
                    rec_layers=args.rec_layers,
                    ei_layers=args.ei_layers,
                )
                vis_net.load_state_dict(
                    torch.load(args.load_weights, map_location="cpu"), strict=False
                )
                vis_net.eval()
                loader_dataset = (
                    "mnist" if args.dataset in ("mnist", "smnist") else args.dataset
                )
                ref_pixel_vec, ref_image = _load_dataset_image(loader_dataset, 0, 0)
                ref_input = torch.from_numpy(ref_pixel_vec).unsqueeze(0)
                use_smnist = args.dataset == "smnist"
                ref_spikes = encode_batch(ref_input, args.dt, use_smnist)
                fig, axes = make_transient_fig(layout="train")
                observe_epoch(
                    vis_net,
                    ref_spikes,
                    0,
                    acc,
                    0.0,
                    args.dt,
                    args.model,
                    fig,
                    axes,
                    None,
                    digit_image=ref_image,
                    total_epochs=1,
                )
                fname = out_dir / "infer_d0s0.png"
                fig.savefig(fname, dpi=120)
                plt.close(fig)
                log.info(f"  → {fname}")

    _elapsed = _time.monotonic() - _t0
    _m, _s = divmod(int(_elapsed), 60)
    log.info(f"Done in {_m}m {_s}s.")
