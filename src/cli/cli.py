"""PING network toolkit — CLI entrypoint.

Subcommands: sim, train.

Usage:
    uv run python src/cli/cli.py                             # sim only (metrics)
    uv run python src/cli/cli.py sim --image                 # snapshot
    uv run python src/cli/cli.py sim --video --scan-var dt   # dt sweep video
    uv run python src/cli/cli.py sim --infer --from-dir RUN  # evaluate trained net
    uv run python src/cli/cli.py train --epochs 10           # train on scikit digits
"""

from __future__ import annotations

import json
import logging
import sys
import time as _time
from pathlib import Path

import torch

import models as M
from config import (
    DEFAULT_ARTIFACT_ROOT,
    _MODEL_CLASSES,
    build_config,
    build_net,
)

# Re-exported through cli/__init__.py for notebook runners (nb003–006).
from config import _extract_records  # noqa: F401

from figkit import plt
from plot import make_transient_fig

# =============================================================================
# Image → spike encoding with stimulus window
# =============================================================================

from encoders import (  # noqa: E402,F401
    EVAL_SEED,
    downsample_spikes_count,
    encode_batch,
    encode_image_spikes,
    encode_images_poisson,
    encode_smnist,
    transport_spikes_bin,
    upsample_spikes_zeropad,
)

from scan import (  # noqa: E402,F401
    SCAN_DEFAULTS,
    _apply_scan_var,
    _auto_device,
    generate_scan,
    primary_hid_key,
    primary_inh_key,
)

# =============================================================================
# Snapshot generators
# =============================================================================

from snapshot import (  # noqa: E402
    generate_image_snapshot,
    generate_sim_only,
    generate_snapshot,
    generate_spike_snapshot,
)

from datasets import (  # noqa: E402,F401
    DATASET_N_HIDDEN_DEFAULTS,
    SHD_N_CHANNELS,
    _load_dataset_image,
    _load_shd,
    _shd_cache_dir,
    load_dataset,
)

# =============================================================================
# Training (moved to train.py)
# =============================================================================

from train import (  # noqa: E402,F401
    observe_epoch,
    seed_everything,
    train,
)

# =============================================================================
# CLI
# =============================================================================

from infer import infer  # noqa: E402

log = logging.getLogger("cli")


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
        "dales_law": "dales_law",
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
        "dales_law": "--dales-law",
        "hidden_sizes": "--n-hidden",
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
    critical = ["dales_law"]
    missing = [k for k in critical if k not in cfg]
    if missing:
        print(
            f"  WARNING: config.json missing {missing} — this training run "
            f"predates these flags. Pass them explicitly on the CLI or retrain."
        )


def _build_parent_parser():
    """Shared parent parser holding every cross-mode argument group."""
    import argparse

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
        help="Allow signed (positive + negative) weights.",
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
        "--ei-strength",
        type=float,
        default=0.5,
        help="E-I coupling: sets W_EI=s, W_IE=s*ratio (default: 0.5)",
    )
    net_group.add_argument(
        "--ei-ratio", type=float, default=2.0, help="W_IE/W_EI ratio (default: 2.0)"
    )
    net_group.add_argument(
        "--w-in-sparsity",
        type=float,
        default=None,
        help="W_in sparsity (default: 0.95)",
    )
    net_group.add_argument(
        "--ei-sparsity",
        type=float,
        default=0.0,
        help="Sparsity of the recurrent E↔I matrices (W_EE / W_EI / W_IE / "
        "W_II). Fraction of entries zeroed independently; surviving entries "
        "rescaled by 1/(1-s) to preserve total expected drive. Default 0 "
        "(dense). Use ≈ 1 - K/N for Brunel/Vreeswijk-style sparse random "
        "connectivity where each post-cell draws ≈ K presynaptic inputs.",
    )
    net_group.add_argument(
        "--independent-drive",
        type=float,
        nargs=2,
        default=None,
        metavar=("RATE_HZ", "G_PER_SPIKE"),
        help="Per-E-cell independent Poisson drive (bypasses W_in). Generates "
        "N_E uncorrelated Poisson streams at RATE_HZ each and adds "
        "G_PER_SPIKE μS of g_E to each E cell per spike. Works on the "
        "synthetic-spikes input mode. Use for Brunel/Vreeswijk-style "
        "experiments where input correlations across cells should be zero.",
    )
    net_group.add_argument(
        "--independent-drive-i",
        type=float,
        nargs=2,
        default=None,
        metavar=("RATE_HZ", "G_PER_SPIKE"),
        help="Per-I-cell independent Poisson drive on the I population's "
        "excitatory conductance. Same semantics as --independent-drive but "
        "targets the I cells directly, so their membrane fluctuations are "
        "no longer driven entirely by E spikes via W^EI. Required for full "
        "V&S-style AI state where both populations need uncorrelated noise.",
    )
    net_group.add_argument(
        "--noise-std",
        type=float,
        default=0.0,
        help="Additive white-noise amplitude (mV-scale) injected into each E "
        "cell's excitatory conductance every timestep — independent neural "
        "noise on top of the stimulus drive. Use with a fixed drive to vary "
        "the input SNR (contrast / noise) independently of the mean. Works on "
        "synthetic-spikes mode.",
    )
    net_group.add_argument(
        "--quenched-drive",
        type=float,
        nargs=2,
        default=None,
        metavar=("MEAN", "STD"),
        help="Per-E-cell constant-in-time DC excitatory conductance, drawn "
        "once from N(MEAN, STD) μS (clamped ≥ 0) and frozen for the whole "
        "trial — V&S's quenched random input. Unlike --independent-drive it "
        "has no per-timestep fluctuation, so it cannot pin spike times; the "
        "Lyapunov probe then measures the network's autonomous chaos rather "
        "than input entrainment. Works on synthetic-spikes mode.",
    )
    net_group.add_argument(
        "--quenched-drive-i",
        type=float,
        nargs=2,
        default=None,
        metavar=("MEAN", "STD"),
        help="Per-I-cell frozen DC excitatory conductance. Same semantics as "
        "--quenched-drive but targets the I population.",
    )
    net_group.add_argument(
        "--exact-k",
        action="store_true",
        help="Use fixed-fan-in (exact-K) recurrent connectivity instead of "
        "per-entry Bernoulli sparsity: every post cell draws exactly "
        "K = round((1−ei_sparsity)·N_pre) presynaptic inputs. Removes the "
        "binomial cell-to-cell fan-in variance — the Brunel/Vreeswijk "
        "convention. No effect unless --ei-sparsity > 0.",
    )
    net_group.add_argument(
        "--lyapunov-eps",
        type=float,
        default=0.0,
        help="If > 0 (synthetic-spikes image mode), rerun the forward pass "
        "on identical input with all membrane voltages ε-perturbed at t=0 "
        "and save the membrane-divergence curve ‖ΔV(t)‖ to snapshot.npz. "
        "Its exponential growth rate is the max Lyapunov exponent: positive "
        "for the chaotic V&S balanced state, ≈ 0 for cycle-locked PING.",
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
        "after build_net. Use to compensate for low "
        "hidden firing rate under mem-mean / "
        "spike-count readouts: bumping W_out up "
        "equalises the trial-level drive into the "
        "output LIF and recovers gradient signal. "
        "Train-mode only. Default 1.0.",
    )
    net_group.add_argument(
        "--surrogate-slope",
        type=float,
        default=None,
        help="Fast-sigmoid surrogate-gradient slope β. "
        "Larger = narrower active window around "
        "threshold. pinglab default 1.0; Cramer et al. "
        "use 40 for SHD RSNNs. Applies to the "
        "fast-sigmoid surrogate used by every spike.",
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
        "--w-ii",
        type=float,
        nargs=2,
        default=None,
        metavar=("MEAN", "STD"),
        help="W_II (I→I) init (mean std). Default: 0 0 (no I→I, canonical "
        "PING). Enable for Brunel/Vreeswijk balanced-network experiments.",
    )
    wt_group.add_argument(
        "--trainable-w-ei",
        action="store_true",
        help="Make COBANet's E→I matrix gradient-carrying (default: "
        "frozen). Use to ask whether the optimiser would discover the "
        "PING-loop weights from scratch.",
    )
    wt_group.add_argument(
        "--trainable-w-ie",
        action="store_true",
        help="Make COBANet's I→E matrix gradient-carrying (default: "
        "frozen). Use to ask whether the optimiser would discover the "
        "PING-loop weights from scratch.",
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
    return parent


def _build_subparsers(parser, parent):
    """Attach the per-mode subcommands (sim/train)."""
    import argparse

    subparsers = parser.add_subparsers(
        dest="mode", help="Mode: sim (--image / --video / --infer for outputs) | train"
    )

    # -- sim subcommand (forward pass; --image / --video add outputs) --
    sim_parser = subparsers.add_parser(
        "sim",
        parents=[parent],
        help="Forward pass + metrics; --image / --video add outputs",
        description="Run a forward pass and report firing-rate metrics. Add "
        "--image to also save a still oscilloscope figure, or --video to "
        "sweep a parameter (--scan-var) and render an MP4.",
        epilog="Examples:\n"
        "  cli.py sim --model ping --ei-strength 0.5\n"
        "  cli.py sim --image --model ping --dataset mnist --digit 3\n"
        "  cli.py sim --video --scan-var ei_strength --scan-min 0 "
        "--scan-max 0.4 --frames 600",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sim_parser.add_argument(
        "--image",
        action="store_true",
        help="Also save a still oscilloscope snapshot (E/I rasters, weight "
        "histograms, PSD).",
    )
    sim_parser.add_argument(
        "--video",
        action="store_true",
        help="Sweep a parameter (--scan-var) and render an MP4, one frame per "
        "value.",
    )
    sim_parser.add_argument(
        "--infer",
        action="store_true",
        help="Load trained weights (--from-dir / --load-weights) and evaluate "
        "test-set accuracy; writes results.json.",
    )
    sim_parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="[--infer] Limit the evaluation dataset to N samples.",
    )
    sim_parser.add_argument(
        "--from-dir",
        type=str,
        default=None,
        help="Load trained weights + inherit config from a training run "
        "directory. CLI flags override inherited values.",
    )
    sim_parser.add_argument(
        "--load-weights",
        type=str,
        default=None,
        help="Path to a weights.pth file (alternative to --from-dir).",
    )
    sim_parser.add_argument(
        "--scan-var",
        type=str,
        default="stim-overdrive",
        choices=list(SCAN_DEFAULTS.keys()),
        help="[--video] Parameter to sweep. 'digit' iterates dataset classes; "
        "'noise' adds input Poisson noise (Hz). (default: stim-overdrive)",
    )
    sim_parser.add_argument(
        "--scan-min",
        type=float,
        default=1.0,
        help="[--video] Scan start value, in the variable's units "
        "(default: 1.0). For digit: integer class.",
    )
    sim_parser.add_argument(
        "--scan-max",
        type=float,
        default=50.0,
        help="[--video] Scan end value (default: 50.0). For digit: integer class.",
    )
    sim_parser.add_argument(
        "--frames",
        type=int,
        default=10,
        help="[--video] Number of frames to render (default: 10). For 'digit' "
        "scan, overridden by scan range.",
    )
    sim_parser.add_argument(
        "--frame-rate",
        type=int,
        default=10,
        help="[--video] Output video frame rate in fps (default: 10).",
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
        "  # PING with gamma oscillation on MNIST\n"
        "  cli.py train --model ping --dataset mnist \\\n"
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
        "--frame-rate", type=int, default=10, help="Video fps for observe (default: 10)"
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
        "--tau-gaba",
        type=float,
        default=None,
        help="Override GABA synaptic decay τ_GABA (ms). Default = models.py "
        "tau_gaba (9.0 ms; Börgers / Buzsaki-Wang range). Used by nb041 to "
        "sweep τ_GABA across {4.5..27} ms while training PING from scratch — "
        "the realised gamma frequency f_γ tracks 1/τ_GABA, and the post-"
        "training rate ceiling tracks f_γ.",
    )
    train_parser.add_argument(
        "--fr-reg-mode",
        type=str,
        default="per-neuron",
        choices=["per-neuron", "population"],
        help="Firing-rate regulariser pooling. 'per-neuron' "
        "(default) computes relu(<z_i> - θ_u)² for each neuron "
        "i and sums — concentrates pressure on the highest-firing "
        "cells (Cramer recipe; same as nb035/036). 'population' "
        "uses a single scalar relu(<z>_pop - θ_u)² where <z>_pop "
        "is the grand mean across batch and neurons, then scaled "
        "by n_neurons to keep s_u's effective magnitude comparable. "
        "Distributes pressure uniformly across cells.",
    )


def parse_args(argv=None):
    """Parse command-line arguments with subparsers for sim/train."""
    import argparse

    argv = sys.argv[1:] if argv is None else list(argv)

    _examples = """\
Each subcommand has its own complete argument listing. The top-level help
above only shows the dispatcher; for the actual flags accepted by a mode,
run:

  python src/cli/cli.py sim    --help
  python src/cli/cli.py train  --help

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
  Weights        --w-in, --w-ee, --w-ei, --w-ie, --w-rec
  Gradient       --v-grad-dampen, --surrogate-slope, --coba-integrator
  Train (train)  --lr, --epochs, --batch-size, --max-samples, --optimizer,
                 --loss, --adaptive-lr, --early-stopping, --observe,
                 --observe-every, --frame-rate, --profile,
                 --fr-reg-upper-theta, --fr-reg-upper-strength,
                 --fr-reg-mode, --skip-bad-grad-threshold
  Sim (sim)      --image, --video, --infer, --from-dir, --load-weights,
                 --scan-var, --scan-min, --scan-max, --frames, --frame-rate,
                 --max-samples
  Output / exec  --out-dir, --wipe-dir, --raster, --layout, --panels,
                 --modal, --modal-gpu

Examples:
  python -m cli                                    # sim (metrics only)
  python -m cli sim --image                         # snapshot (ping default)
  python -m cli sim --video --scan-var ei_strength  # sweep E-I coupling
  python -m cli sim --video --scan-var spike_rate --scan-min 5 --scan-max 100
  python -m cli train --epochs 100 --observe video
  python -m cli sim --image --input dataset --dataset mnist --digit 3
  python -m cli sim --infer --load-weights weights.pth --dt 0.5
  python -m cli sim --infer --from-dir runs/foo

Models:
  ping        COBANet with E↔I coupling. With --ei-strength > 0 the
              recurrent inhibitory loop is wired up and frozen at init;
              feedforward weights train against this fixed substrate.
              (With --ei-strength 0 the I-loop is silenced — E cells only.)

For the underlying theory of --v-grad-dampen see /articles/ar006/.
"""
    parser = argparse.ArgumentParser(
        description="Oscilloscope — PING network toolkit",
        epilog=_examples,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parent = _build_parent_parser()

    _build_subparsers(parser, parent)

    args = parser.parse_args(argv)
    if args.mode is None:
        parser.print_help()
        sys.exit(0)

    # --from-dir: inherit training params from config.json, fill unset values
    if args.mode == "sim" and getattr(args, "from_dir", None):
        _apply_from_dir(args, argv)
    if getattr(args, "infer", False) and not getattr(args, "load_weights", None):
        print("Error: sim --infer requires --load-weights or --from-dir")
        sys.exit(1)

    # Auto-detect: if user explicitly passed --dataset/--digit/--sample but
    # left --input at the default "synthetic-spikes", flip to "dataset". The
    # explicit dataset flags only make sense in dataset input mode, so this
    # avoids the silent footgun where "image --dataset mnist --digit 0" went
    # through the synthetic-spikes branch and ignored the digit.
    def _flag_in_argv(*names):
        for arg in argv:
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


def configure_models(args):
    """Apply CLI overrides to models.py globals — the one sanctioned boundary.

    Model globals (M.SURROGATE_SLOPE, M.tau_snn, M.max_rate_hz, the dt-derived
    constants, …) are kept as module globals so torch.compile specializes the
    graph on them as constants, so they cannot live on the Config dataclass
    ([[project_models_globals_are_torch_compile_choice]]). This is the single
    place CLI arguments are written into them. The data-dependent globals
    (M.N_IN / M.N_HID / M.T_steps) are set later by train.py / scan.py, where
    the dataset shape is known.

    Declarative table — {arg_name: (M attribute, cast)} — applied only when the
    arg was given. Add a new model global by adding a row, not an if-branch.
    """
    arg_to_global = {
        "surrogate_slope": ("SURROGATE_SLOPE", float),
        "tau_mem": ("tau_snn", float),
        "tau_syn": ("tau_ampa", float),
        "readout_tau_out": ("tau_out_ms", float),
    }
    for arg, (attr, cast) in arg_to_global.items():
        val = getattr(args, arg, None)
        if val is not None:
            setattr(M, attr, cast(val))
    # Special case: a bare flag.
    if getattr(args, "exact_k", False):
        M.EXACT_K_CONNECTIVITY = True
    # Input Poisson rate and trial duration — single source of truth. Every
    # code path reads M.max_rate_hz / M.T_ms, so setting them here once means
    # all dispatch branches (sim/train × all input types) respect
    # --input-rate / --t-ms. Subfunctions that change dt recalc M.p_scale /
    # M.T_steps via patch_dt as usual.
    M.max_rate_hz = args.spike_rate
    M.T_ms = args.t_ms


def save_run_artifacts(out_dir, args, mode):
    """Save config.json (with provenance), run.sh, set up logging, print intro."""
    import json
    import logging
    import runlog

    out_dir = Path(out_dir)
    if args.wipe_dir and out_dir.exists():
        import shutil

        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # config.json — with provenance metadata at top
    config = {"mode": mode}
    config.update(runlog.provenance())
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
    log = logging.getLogger("cli")
    log.setLevel(logging.DEBUG)
    log.handlers.clear()

    class _StripAnsiFormatter(logging.Formatter):
        def format(self, record):
            msg = super().format(record)
            return runlog._strip_ansi(msg)

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
    """Group CLI args into sections and print via runlog.print_intro."""
    import runlog

    model = config.get("model", "ping")
    dataset = config.get("dataset", "scikit")

    def g(*keys):
        return {k: config[k] for k in keys if k in config}

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
            "dales_law": config.get("dales_law", True),
            "ei_strength": config.get("ei_strength"),
            "ei_ratio": config.get("ei_ratio"),
        },
        "Weights": g("w_in", "w_in_sparsity", "w_ei", "w_ie"),
        "Training": g("epochs", "lr", "adaptive_lr", "v_grad_dampen")
        if mode == "train"
        else {},
        "Scan": g("scan_var", "scan_min", "scan_max", "frames", "frame_rate")
        if config.get("video")
        else {},
        "Output": g("out_dir", "observe", "wipe_dir"),
        "Provenance": {
            "git_sha": config.get("git_sha"),
            "device": config.get("device"),
            "run_id": config.get("run_id"),
            "started_at": config.get("started_at"),
        },
    }
    runlog.print_intro(log, mode, model, dataset, sections)


def _run_sim(args, C, out_dir, log):
    """Forward pass; --image / --video select the output.

    Each path runs its own forward pass and reports firing-rate metrics:
    --image renders a still snapshot, --video sweeps a parameter into an
    MP4, and bare `sim` (neither flag) does a metrics-only pass.
    """
    log.info(f"Device: {C.DEVICE}")
    if getattr(args, "infer", False):
        _emit_infer(args, C, out_dir, log)
        return
    want_image = getattr(args, "image", False)
    want_video = getattr(args, "video", False)
    if want_image:
        _emit_image(args, C, log)
    if want_video:
        _emit_video(args, C, log)
    if not (want_image or want_video):
        generate_sim_only(
            spike_rate=args.spike_rate,
            overdrive=args.overdrive,
            dt=args.dt,
            model_name=args.model,
            input_mode=args.input,
            t_e_async=C.T_E_ASYNC_DEFAULT,
        )


def _emit_image(args, C, log):
    """Save a still oscilloscope snapshot (the former `image` mode)."""
    t_e_async = C.T_E_ASYNC_DEFAULT
    # Apply CLI overrides to module-level config so snapshot generators
    # (which read C.cfg) see the requested W^EI / W^IE / W^II / sparsity.
    if args.w_ei is not None:
        C.cfg.w_ei = tuple(args.w_ei)
    if args.w_ie is not None:
        C.cfg.w_ie = tuple(args.w_ie)
    if args.w_ii is not None:
        C.cfg.w_ii = tuple(args.w_ii)
    if getattr(args, "ei_sparsity", 0.0) > 0:
        C.cfg.sparsity = float(args.ei_sparsity)
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
            independent_drive=args.independent_drive,
            independent_drive_i=args.independent_drive_i,
            quenched_drive=args.quenched_drive,
            quenched_drive_i=args.quenched_drive_i,
            noise_std=args.noise_std,
            lyapunov_eps=args.lyapunov_eps,
        )
    else:
        generate_snapshot(
            args.overdrive,
            dt=args.dt,
            model_name=args.model,
            t_e_async=t_e_async,
        )


def _emit_video(args, C, log):
    """Sweep a parameter and render an MP4 (the former `video` mode)."""
    t_e_async = C.T_E_ASYNC_DEFAULT
    # Emit init_d0s0.png alongside the scan video when input is dataset.
    # Mirrors train mode so sweeps have a comparable static reference.
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


def _resolve_w_in(args):
    """Default w_in to (0.3, 0.06); expand a single value to (w, w*0.1)."""
    w_in = args.w_in or [0.3, 0.06]
    if len(w_in) == 1:
        w_in = [w_in[0], w_in[0] * 0.1]
    return w_in


def _run_train(args, C, out_dir, log):
    train(
        model_name=args.model,
        lr=args.lr,
        epochs=args.epochs,
        dt=args.dt or 0.1,
        observe=args.observe,
        out_dir=str(out_dir),
        device_name=args.device,
        w_in=_resolve_w_in(args),
        w_ei=args.w_ei,
        w_ie=args.w_ie,
        w_ii=args.w_ii,
        ei_strength=args.ei_strength,
        ei_ratio=args.ei_ratio,
        w_in_sparsity=args.w_in_sparsity or 0.0,
        dataset=args.dataset,
        snapshot_init=True,
        snapshot_end=True,
        t_ms=args.t_ms,
        hidden_sizes=args.n_hidden,
        max_samples=args.max_samples,
        v_grad_dampen=args.v_grad_dampen,
        dales_law=args.dales_law,
        ei_layers=args.ei_layers,
        batch_size=args.batch_size,
        seed=args.seed,
        readout_w_out_scale=args.readout_w_out_scale,
        readout_mode=args.readout_mode,
        tau_gaba=args.tau_gaba,
        fr_reg_upper_theta=args.fr_reg_upper_theta,
        fr_reg_upper_strength=args.fr_reg_upper_strength,
        fr_reg_mode=args.fr_reg_mode,
        trainable_w_ei=args.trainable_w_ei,
        trainable_w_ie=args.trainable_w_ie,
    )


def _emit_infer(args, C, out_dir, log):
    """Load trained weights and evaluate test-set accuracy (the former `infer` mode)."""
    w_in = _resolve_w_in(args)
    acc = infer(
        dt=args.dt,
        out_dir=str(out_dir),
        model_name=args.model,
        load_weights=args.load_weights,
        dataset=args.dataset,
        max_samples=args.max_samples,
        t_ms=args.t_ms,
        w_in=w_in,
        ei_strength=args.ei_strength,
        ei_ratio=args.ei_ratio,
        w_in_sparsity=args.w_in_sparsity or 0.0,
        hidden_sizes=args.n_hidden,
        dales_law=args.dales_law,
        ei_layers=args.ei_layers,
        seed=args.seed,
    )["acc"]
    if getattr(args, "image", False):
        _render_infer_snapshot(args, C, out_dir, log, w_in, acc)


def _render_infer_snapshot(args, C, out_dir, log, w_in, acc):
    """Render a single oscilloscope frame for a trained net at one dt (--observe)."""
    C.cfg.n_e = M.N_HID
    C.cfg.n_i = M.N_INH
    vis_net = build_net(
        args.model,
        w_in=w_in,
        w_in_sparsity=args.w_in_sparsity or 0.0,
        ei_strength=args.ei_strength,
        ei_ratio=args.ei_ratio,
        randomize_init=True,
        dales_law=args.dales_law,
        hidden_sizes=args.n_hidden,
        ei_layers=args.ei_layers,
    )
    vis_net.load_state_dict(
        torch.load(args.load_weights, map_location="cpu"), strict=False
    )
    vis_net.eval()
    loader_dataset = "mnist" if args.dataset in ("mnist", "smnist") else args.dataset
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


_MODE_HANDLERS = {
    "sim": _run_sim,
    "train": _run_train,
}


def _dispatch_to_modal(args, argv):
    """Re-dispatch the run to Modal, stripping --modal / --modal-gpu from argv."""
    out_dir = args.out_dir or str(DEFAULT_ARTIFACT_ROOT)
    cli_args = []
    i = 0
    while i < len(argv):
        if argv[i] == "--modal-gpu":
            i += 2  # skip flag and its value
        elif argv[i] == "--modal":
            i += 1
        else:
            cli_args.append(argv[i])
            i += 1
    from modal_app import dispatch_to_modal

    dispatch_to_modal(cli_args, out_dir, gpu=args.modal_gpu)


def main(argv=None):
    """Parse args and run the requested mode. Returns a process exit code."""
    _t0 = _time.monotonic()

    argv = sys.argv[1:] if argv is None else list(argv)
    args = parse_args(argv)
    mode = args.mode

    # --modal: re-dispatch to Modal and exit
    if getattr(args, "modal", False):
        _dispatch_to_modal(args, argv)
        return 0

    # Build config for non-train modes (build_config syncs the module aliases).
    if mode != "train":
        build_config(args)

    # Apply CLI overrides to models.py globals (all modes, incl. train).
    configure_models(args)

    import config as C

    # Determine output directory
    out_dir = args.out_dir
    if out_dir is None:
        out_dir = str(DEFAULT_ARTIFACT_ROOT)
    out_dir = Path(out_dir)

    # Save run artifacts for all modes
    log = save_run_artifacts(out_dir, args, mode)

    # M.max_rate_hz / M.T_ms are set in configure_models above.
    if args.t_ms <= C.STEP_ON_MS:
        log.warning(
            f"  --t-ms={args.t_ms} <= STEP_ON_MS={C.STEP_ON_MS}: "
            f"the stimulus window never fires within the trial. "
            f"Consider --t-ms >= {C.STEP_OFF_MS:.0f} (STEP_OFF_MS)."
        )

    if args._input_auto:
        log.info("  --input auto → dataset (inferred from --dataset/--digit/--sample)")

    _MODE_HANDLERS[mode](args, C, out_dir, log)

    _elapsed = _time.monotonic() - _t0
    _m, _s = divmod(int(_elapsed), 60)
    log.info(f"Done in {_m}m {_s}s.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
