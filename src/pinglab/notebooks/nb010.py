"""Notebook runner for entry 003 — cuba vs standard-snn Δt-stability.

Trains *standard-snn*, *snntorch-library*, and *cuba* at two training
*dt* regimes (DT_TRAINS), then for each regime runs inference across the
same eval-*dt* grid (DT_SWEEP) with weights frozen. *cuba* applies
$(1-\beta)/dt$ drive scaling and should hold accuracy flat across
eval-*dt* regardless of train-*dt*; the snnTorch paths sag as eval-*dt*
departs from train-*dt*, and the sag-point moves with train-*dt*. This
is the first rung-to-rung ablation on the [CUBA
ladder](/models/#the-ladder).

Writes (per regime figures have panels for each dt_train):
  * training_curves.png — train loss & test accuracy per epoch
  * dt_sweep.png — accuracy vs eval-dt (the money plot)
  * firing_rates.png — mean hidden firing rate vs eval-dt
  * training_dt{dt}_{model}.mp4 — per-epoch training videos
  * dt_sweep_dt{dt}_{model}.mp4 — per-sweep-dt inference videos
  * numbers.json — config + per-regime/per-model best/final + sweep results

Notebook entry: src/docs/src/pages/notebooks/nb010.mdx
"""
from __future__ import annotations

import json
import math
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import sh

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src" / "pinglab"))

import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from _modal import BatchDispatcher, parse_modal_gpu  # noqa: E402
from _tier import parse_tier  # noqa: E402
from _run_id import next_run_id, persist as persist_run_id  # noqa: E402
from config import build_net  # noqa: E402
from pinglab import theme  # noqa: E402

SLUG = "nb010"
ARTIFACTS = REPO / "src" / "artifacts" / "notebooks" / SLUG
FIGURES = REPO / "src" / "docs" / "public" / "figures" / "notebooks" / SLUG
OSCILLOSCOPE = REPO / "src" / "pinglab" / "oscilloscope.py"

MODELS = ["standard-snn", "snntorch-library", "cuba", "coba", "ping"]
# Three input-transport modes, one per paper-named case (Parthasarathy
# et al. §2.1 / Fig 1B / §2.3). The FrozenEncoder anchors at train-dt:
#   * upsample   — zero-pad the reference stream to finer eval-dt
#                  (§2.1 / Fig 1B); eval-dt <= train-dt only.
#   * downsample — sum-pool the reference stream to coarser eval-dt
#                  (§2.3); eval-dt >= train-dt only.
#   * resample   — fresh Poisson at the target dt (§2.1 alternative);
#                  works in both directions.
ENCODER_MODES = ["upsample", "downsample", "resample"]
T_MS = 200.0
# Two training regimes: fine dt (snnTorch research setting) and coarse dt
# (near τ_mem, where snnTorch models typically saturate).
DT_TRAINS = [0.1, 1.0]
# Per-regime sweep grid. Every point is an integer divisor or multiple of
# its regime's train-dt so the paper-style zero-pad transport is exact in
# both directions.
DT_SWEEPS: dict[float, list[float]] = {
    0.1: [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90,
          1.00, 1.10, 1.20, 1.30, 1.40, 1.50, 1.60, 1.70, 1.80, 1.90, 2.00],
    1.0: [0.05, 0.10, 0.20, 0.25, 0.50, 1.00, 2.00],
}
DT_SWEEP = sorted({d for grid in DT_SWEEPS.values() for d in grid})
SEED = 42
DEFAULT_TIER = "small"  # see src/docs/src/pages/styleguide.md § 10 Run sizing tiers
# Per-tier (max-samples, epochs). t-ms is fixed by T_MS above.
TIER_CONFIG = {
    "extra small": dict(max_samples=100,   epochs=1),
    "small":       dict(max_samples=500,   epochs=5),
    "medium":      dict(max_samples=2000,  epochs=10),
    "large":       dict(max_samples=5000,  epochs=40),
    "extra large": dict(max_samples=10000, epochs=40),
}
TIER = DEFAULT_TIER  # overridable via --tier <name>
TAU_MEM_MS = 10.0  # matches models.py SNN_TAU_MEM_MS

# Per-step drive compensation for cuba at training dt. cuba's update is
#   mem = β·mem + (1-β)/dt · (W·s) + (1-β) · b      (β = exp(-dt/τ_mem))
# so at training-dt the spike drive is scaled by (1-β)/dt ≈ 0.099 and the
# bias drive by (1-β) ≈ 0.025, both ≪ 1. Starting cuba from the same random
# weights as standard-snn leaves it ~10× below threshold for spike drive
# and ~40× below for bias drive — silent at init, no gradient, no learning.
# Scaling cuba's W by dt/(1-β) and b by 1/(1-β) cancels those factors at
# training-dt so both models start with the same per-step effective drive
# (and therefore the same init firing rate). The scaling is a one-shot
# multiplier on init weights, not part of the update rule, so cuba's
# dt-invariance across the eval sweep is preserved.
def cuba_init_scales(dt: float, tau: float = TAU_MEM_MS) -> tuple[float, float]:
    beta = math.exp(-dt / tau)
    return dt / (1.0 - beta), 1.0 / (1.0 - beta)


def init_scales_for(model: str, dt_train: float) -> tuple[float, float]:
    """Per-step drive compensation is (1.0, 1.0) for both snnTorch paths;
    cuba gets dt-dependent (W-scale, b-scale) derived from cuba_init_scales
    so it starts at the same per-step effective drive as standard-snn at
    training-dt."""
    if model == "cuba":
        return cuba_init_scales(dt_train)
    return (1.0, 1.0)

MODEL_LABELS = {
    "standard-snn":     "standard-snn",
    "snntorch-library": "snntorch-library",
    "cuba":             "cuba",
    "coba":             "coba",
    "ping":             "ping",
}
MODEL_COLORS = {
    "standard-snn":     theme.CAT_BLUE,
    "snntorch-library": theme.CAT_ORANGE,
    "cuba":             theme.CAT_GREEN,
    "coba":             theme.CAT_PURPLE,
    "ping":             theme.CAT_RED,
}

# Per-model CLI recipe for training. The CUBANet-family paths
# (standard-snn, snntorch-library, cuba) share a lr=0.01 + kaiming-init
# + Dale's law-off pipeline; cuba additionally gets the (1-β)/dt
# init-scale compensation at train-dt. coba and ping are dispatched
# through PINGNet (coba is ping with ei_strength=0) with lr=1e-4,
# explicit --w-in / --w-in-sparsity, and --v-grad-dampen from
# models.mdx. See src/docs/src/pages/models.mdx "Model training" table.
#
# Keys in each entry become --flag value pairs on the oscilloscope CLI.
# "--flag-only" keys with value True become bare flags. None values skip.
# Extra keys:
#   __build_as: name to pass as --model (coba dispatches via --model ping).
#   __init_scale: if True, also pass --init-scale-weight/-bias from
#                 init_scales_for(); skipped otherwise.
MODEL_CONFIG: dict[str, dict] = {
    "standard-snn": {
        "__build_as": "standard-snn",
        "__init_scale": True,
        "--kaiming-init": True,
        "--lr": "0.01",
    },
    "snntorch-library": {
        "__build_as": "snntorch-library",
        "__init_scale": True,
        "--kaiming-init": True,
        "--lr": "0.01",
    },
    "cuba": {
        "__build_as": "cuba",
        "__init_scale": True,
        "--kaiming-init": True,
        "--lr": "0.01",
    },
    "coba": {
        "__build_as": "ping",
        "__init_scale": False,
        "--ei-strength": "0",
        "--v-grad-dampen": "1000",
        "--w-in": "0.3",
        "--w-in-sparsity": "0.95",
        "--lr": "0.0001",
    },
    "ping": {
        "__build_as": "ping",
        "__init_scale": False,
        "--ei-strength": "0.5",
        "--v-grad-dampen": "1000",
        "--w-in": "1.2",
        "--w-in-sparsity": "0.95",
        "--lr": "0.0001",
    },
}


def training_video_path(out_dir: Path) -> Path:
    return out_dir / "training.mp4"


# Models sharing the CUBANet class — within this family, matched seed +
# --kaiming-init produces bit-identical raw weights because every parameter is
# allocated in the same order. snntorch-library goes through nn.Linear, which
# uses a different kaiming_uniform_ convention, so only biases coincidentally
# match. Treat library's init as independent and parity-tested at run time.
SNNTORCHNET_FAMILY = {"standard-snn", "cuba"}


def verify_init_match(models: list[str], seed: int,
                      dt_trains: list[float]) -> dict:
    """Preflight: CUBANet-family models must start from the same random
    weights, modulo each model's per-step drive scaling. standard-snn and
    cuba share the CUBANet class and with matched seed allocate every
    tensor in the same order, giving bit-identical raw weights. cuba then
    gets a one-shot multiplier (init_scales_for) per training dt so its
    per-step drive matches standard-snn at that dt. snntorch-library
    uses nn.Linear's own kaiming_uniform_ and is reported but not asserted
    — its role is an external parity reference, not a bit-match."""
    nets: dict[str, object] = {}
    # Only the CUBANet-family + snntorch-library go through the kaiming-init
    # path this preflight inspects. coba / ping use --w-in / --w-in-sparsity
    # and a different class (PINGNet) — independent by design; we don't build
    # them here.
    preflight_models = [m for m in models if m in SNNTORCHNET_FAMILY
                        or m == "snntorch-library"]
    for m in preflight_models:
        torch.manual_seed(seed)
        nets[m] = build_net(m, kaiming_init=True, hidden_sizes=[1024])
    report: dict[str, object] = {
        "family": sorted(SNNTORCHNET_FAMILY & set(models)),
        "seed": seed,
        "scales_per_regime": {
            str(dt): {m: {"weight": init_scales_for(m, dt)[0],
                          "bias": init_scales_for(m, dt)[1]}
                      for m in models}
            for dt in dt_trains
        },
        "params": {},
        "independent": sorted(set(models) - SNNTORCHNET_FAMILY),
    }
    family = [m for m in models if m in SNNTORCHNET_FAMILY]
    if len(family) >= 2:
        ref_name = family[0]
        ref_sd = nets[ref_name].state_dict()
        report["ref"] = ref_name
        for m in family[1:]:
            sd = nets[m].state_dict()
            if set(sd.keys()) != set(ref_sd.keys()):
                raise SystemExit(
                    f"init-match: state_dict keys differ between {ref_name!r} "
                    f"and {m!r}: {set(sd.keys()) ^ set(ref_sd.keys())}")
            for k, v in ref_sd.items():
                if not torch.equal(v, sd[k]):
                    max_abs = (v - sd[k]).abs().max().item()
                    raise SystemExit(
                        f"init-match: parameter {k!r} differs between "
                        f"{ref_name!r} and {m!r}: max_abs={max_abs:g}")
        for k, v in ref_sd.items():
            report["params"][k] = list(v.shape)
        print(f"[init-match] pre-scale weights match across {family} at seed={seed}")
    if report["independent"]:
        print(f"[init-match] independent init (nn.Linear kaiming): "
              f"{report['independent']}")
    for dt in dt_trains:
        # init-scale compensation is only meaningful for the CUBANet-family
        # models; coba/ping use --w-in / --w-in-sparsity instead.
        scaled_models = [m for m in models if MODEL_CONFIG.get(m, {}).get("__init_scale")]
        if not scaled_models:
            continue
        scales = ", ".join(
            f"{m}(W×{init_scales_for(m, dt)[0]:.3f} "
            f"b×{init_scales_for(m, dt)[1]:.3f})"
            for m in scaled_models)
        print(f"[init-match] dt={dt}: per-model init_scale: {scales}")
    return report


def _regime_key(dt_train: float) -> str:
    """Filesystem-safe label for a training regime (e.g. 0.1 → 'dt0.1')."""
    return f"dt{dt_train:g}"


def train_model(model: str, dt_train: float,
                dispatcher: BatchDispatcher) -> Path:
    """Queue a train cell at dt_train. Runs immediately for local, batched
    on drain() for Modal."""
    out_dir = ARTIFACTS / _regime_key(dt_train) / model / "train"
    config = MODEL_CONFIG[model]
    build_as = config["__build_as"]
    print(f"[{model} @ dt={dt_train}] training → {out_dir.relative_to(REPO)}"
          + (f"  [modal:{dispatcher.modal_gpu}]" if dispatcher.modal_gpu else ""))
    osc_args = [
        "train",
        "--model", build_as,
        "--dataset", "mnist",
        "--max-samples", str(TIER_CONFIG[TIER]["max_samples"]),
        "--epochs", str(TIER_CONFIG[TIER]["epochs"]),
        "--t-ms", str(T_MS),
        "--dt", str(dt_train),
        "--seed", str(SEED),
        "--observe", "video",
        "--frame-rate", "1",
        "--out-dir", str(out_dir),
        "--wipe-dir",
    ]
    if config["__init_scale"]:
        sw, sb = init_scales_for(model, dt_train)
        osc_args += ["--init-scale-weight", f"{sw}",
                     "--init-scale-bias", f"{sb}"]
        print(f"  init_scale W×{sw:.3f} b×{sb:.3f}")
    for k, v in config.items():
        if k.startswith("__"):
            continue
        if v is True:
            osc_args.append(k)
        elif v is not None:
            osc_args += [k, v]
    # PINGNet family (ping, coba) at dt=0.1 (6000 BPTT steps × 1024 hidden
    # × COBA state) OOMs both T4 (14.56 GiB) and A10G (24 GiB); bump to
    # A100 (80 GiB) when dispatching to Modal on a smaller GPU.
    gpu_override = None
    if (dispatcher.modal_gpu in ("T4", "L4", "A10G")
            and build_as == "ping" and dt_train <= 0.25):
        gpu_override = "A100"
        print(f"  [modal] upgrading {model}@dt={dt_train} from "
              f"{dispatcher.modal_gpu} to A100 (memory)")
    dispatcher.submit(osc_args, out_dir, gpu_override=gpu_override)
    return out_dir


def _mode_key(encoder_mode: str) -> str:
    """Filesystem-safe label for an encoder mode (e.g. 'count-pool' → 'count_pool')."""
    return encoder_mode.replace("-", "_")


def _sweep_grid_for(dt_train: float, encoder_mode: str) -> list[float]:
    """Per-mode eval-dt grid. The count-preserving modes only cover their
    paper-valid direction (upsample: eval-dt <= train-dt, §2.1 Fig 1B;
    downsample: eval-dt >= train-dt, §2.3). Resample covers both
    directions. All three include eval-dt == train-dt as the identity
    anchor."""
    full = DT_SWEEPS[dt_train]
    if encoder_mode == "upsample":
        return [d for d in full if d <= dt_train + 1e-9]
    if encoder_mode == "downsample":
        return [d for d in full if d >= dt_train - 1e-9]
    if encoder_mode == "resample":
        return list(full)
    raise ValueError(f"unknown encoder mode {encoder_mode!r}")


def sweep_model(model: str, dt_train: float, train_dir: Path,
                encoder_mode: str, dispatcher: BatchDispatcher) -> Path:
    """Queue a dt-sweep cell. Runs immediately for local, batched on drain()
    for Modal. The train_dir must already exist (drain the train phase first)."""
    sweep_dir = ARTIFACTS / _regime_key(dt_train) / model / f"sweep_{_mode_key(encoder_mode)}"
    sweep_grid = _sweep_grid_for(dt_train, encoder_mode)
    print(f"[{model} @ dt={dt_train}, mode={encoder_mode}] dt-sweep → "
          f"{sweep_dir.relative_to(REPO)}")
    # Sweep videos are only emitted for resample mode because it's the
    # only transport that covers the full eval-dt grid in every regime;
    # upsample and downsample each cover half and would produce
    # degenerate 2-frame videos at one of the train-dt settings.
    osc_args = [
        "infer",
        "--from-dir", str(train_dir),
        "--dt-sweep", *[str(d) for d in sweep_grid],
        "--frozen-inputs-mode", encoder_mode,
        "--max-samples", str(TIER_CONFIG[TIER]["max_samples"]),
        "--out-dir", str(sweep_dir),
        "--wipe-dir",
    ]
    if encoder_mode == "resample":
        osc_args += ["--observe", "video"]
    # PINGNet family inference at small dt hits the same memory wall as
    # training; see train_model above for the A100 rationale.
    build_as = MODEL_CONFIG[model]["__build_as"]
    gpu_override = None
    if (dispatcher.modal_gpu in ("T4", "L4", "A10G")
            and build_as == "ping" and dt_train <= 0.25):
        gpu_override = "A100"
    dispatcher.submit(osc_args, sweep_dir, gpu_override=gpu_override)
    return sweep_dir


def load_metrics(run_dir: Path) -> dict:
    return json.loads((run_dir / "metrics.json").read_text())


def load_config(run_dir: Path) -> dict:
    return json.loads((run_dir / "config.json").read_text())


def load_sweep(sweep_dir: Path) -> dict:
    return json.loads((sweep_dir / "results.json").read_text())


def run_date(run_dir: Path) -> str:
    fmt = "%A, %B %-d %Y at %H:%M"
    metrics = load_metrics(run_dir)
    if "run_finished_at" in metrics:
        dt_utc = datetime.fromisoformat(metrics["run_finished_at"])
        return dt_utc.astimezone().strftime(fmt)
    mtime = (run_dir / "metrics.json").stat().st_mtime
    return datetime.fromtimestamp(mtime).strftime(fmt)


def _stamp_figure(fig, notebook_run_id: str) -> None:
    fig.text(0.995, 0.005, notebook_run_id, ha="right", va="bottom",
             fontsize=7, color=theme.LABEL, family="monospace")


def plot_training_curves(regime_train_dirs: dict[float, dict[str, Path]],
                         out_path: Path, notebook_run_id: str) -> None:
    """Grid: one row per training regime, columns = (loss, accuracy)."""
    dt_trains = sorted(regime_train_dirs.keys())
    n = len(dt_trains)
    fig, axes = plt.subplots(n, 2, figsize=(8, 4.5 * max(n, 1) / 2),
                             squeeze=False)
    for i, dt_train in enumerate(dt_trains):
        ax_loss, ax_acc = axes[i]
        for model, run_dir in regime_train_dirs[dt_train].items():
            metrics = load_metrics(run_dir)
            epochs = [e["ep"] for e in metrics["epochs"]]
            loss = [e["loss"] for e in metrics["epochs"]]
            acc = [e["acc"] for e in metrics["epochs"]]
            ax_loss.plot(epochs, loss, marker="o",
                         color=MODEL_COLORS[model], label=MODEL_LABELS[model])
            ax_acc.plot(epochs, acc, marker="o",
                        color=MODEL_COLORS[model], label=MODEL_LABELS[model])
        ax_loss.set_xlabel("epoch")
        ax_loss.set_ylabel("train loss")
        ax_loss.set_title(f"train loss (train dt = {dt_train} ms)")
        ax_loss.grid(alpha=0.3)
        ax_loss.legend(frameon=False, fontsize=8)
        ax_acc.set_xlabel("epoch")
        ax_acc.set_ylabel("test accuracy (%)")
        ax_acc.set_title(f"test accuracy (train dt = {dt_train} ms)")
        ax_acc.grid(alpha=0.3)
        ax_acc.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    _stamp_figure(fig, notebook_run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _matrix_axes(n_rows: int, n_cols: int):
    """Grid: one row per encoder mode, one column per training regime. Shared
    y across all panels (accuracy or firing-rate, both unitful) and shared x
    per column so the three encoder-mode rows put their eval-dt points on the
    same scale — upsample (eval-dt ≤ train-dt) and downsample (eval-dt ≥
    train-dt) only populate half of their panel, but the reader can compare
    positions across rows directly."""
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(8, 4.5 * n_rows / 2),
                             sharex="col", sharey=True, squeeze=False)
    return fig, axes


ROW_MODES = ["count-preserving", "resample"]
# Marker per encoder mode in the count-preserving row. Same silhouette
# size, different geometry so upsample vs downsample is readable at a
# glance without implying a direction via the glyph (that's what the
# red train-dt line is for):
#   "o" = upsample    (filled circle)
#   "s" = downsample  (filled square)
#   "o" = resample    (filled circle on its own row, no ambiguity)
MODE_MARKERS = {"upsample": "o", "downsample": "s", "resample": "o"}


def _plot_model_mode(ax, sweep_dir, color, label, marker, key="acc"):
    """Plot one (model, encoder mode) series on ax; returns the list of
    (dt, value) points actually plotted, skipping entries missing the key."""
    if sweep_dir is None or not sweep_dir.exists():
        return []
    blob = load_sweep(sweep_dir)
    dts, vals = [], []
    for r in blob["sweep"]:
        v = r.get(key) if key != "acc" else r["acc"]
        if v is None:
            continue
        dts.append(r["dt"])
        vals.append(v)
    if not dts:
        return []
    ax.plot(dts, vals, marker=marker, color=color, label=label,
            linewidth=1.4, markersize=6)
    return list(zip(dts, vals))


def _mode_legend_handles():
    """Handles annotating which marker means which direction — drawn once
    on the count-preserving row so the reader can decode circle vs square
    without hunting for the caption."""
    from matplotlib.lines import Line2D
    return [
        Line2D([0], [0], color=theme.DIM, marker=MODE_MARKERS["upsample"],
               linestyle="-", markersize=6,
               label="upsample (eval-dt ≤ train-dt)"),
        Line2D([0], [0], color=theme.DIM, marker=MODE_MARKERS["downsample"],
               linestyle="-", markersize=6,
               label="downsample (eval-dt ≥ train-dt)"),
    ]


def plot_firing_rates(regime_sweep_dirs: dict[float, dict[str, dict[str, Path]]],
                      out_path: Path, notebook_run_id: str) -> None:
    """Mean hidden-layer firing rate vs eval-dt, 2×N layout:
    row 0 = count-preserving (upsample + downsample on one axis, marker
    shape encodes direction), row 1 = resample."""
    dt_trains = sorted(regime_sweep_dirs.keys())
    fig, axes = _matrix_axes(len(ROW_MODES), len(dt_trains))
    any_data = False
    for j, dt_train in enumerate(dt_trains):
        # Row 0: count-preserving — upsample and downsample, per model,
        # same colour and connected by the underlying line.
        ax = axes[0, j]
        for model in regime_sweep_dirs[dt_train]:
            mode_dirs = regime_sweep_dirs[dt_train][model]
            pts_up = _plot_model_mode(ax, mode_dirs.get("upsample"),
                                      MODEL_COLORS[model], MODEL_LABELS[model],
                                      MODE_MARKERS["upsample"], key="hid_rate_hz")
            pts_dn = _plot_model_mode(ax, mode_dirs.get("downsample"),
                                      MODEL_COLORS[model], None,
                                      MODE_MARKERS["downsample"], key="hid_rate_hz")
            if pts_up or pts_dn:
                any_data = True
        ax.axvline(dt_train, color=theme.DANGER, linestyle="--", linewidth=1,
                   label=f"train dt={dt_train}")
        ax.set_title(f"train dt = {dt_train} ms")
        if j == 0:
            ax.set_ylabel("count-preserving\nhidden rate (Hz)")
            ax.legend(frameon=False, fontsize=7, loc="upper right")
            # Direction-marker legend in a second slot so the < / > glyphs
            # get their decoding without polluting the model legend.
            ax.add_artist(ax.legend(handles=_mode_legend_handles(),
                                    frameon=False, fontsize=6,
                                    loc="lower right"))
            ax.legend(frameon=False, fontsize=7, loc="upper right")
        ax.grid(alpha=0.3)

        # Row 1: resample — one circle marker per point.
        ax = axes[1, j]
        for model in regime_sweep_dirs[dt_train]:
            pts = _plot_model_mode(ax,
                                   regime_sweep_dirs[dt_train][model].get("resample"),
                                   MODEL_COLORS[model], MODEL_LABELS[model],
                                   MODE_MARKERS["resample"], key="hid_rate_hz")
            if pts:
                any_data = True
        ax.axvline(dt_train, color=theme.DANGER, linestyle="--", linewidth=1)
        ax.set_xlabel("eval dt (ms)")
        if j == 0:
            ax.set_ylabel("resample\nhidden rate (Hz)")
        ax.grid(alpha=0.3)
    if not any_data:
        print("[warn] no firing-rate data in sweep results; skipping firing_rates.png")
        plt.close(fig)
        return
    fig.suptitle("Δt-stability: hidden firing rate vs eval-dt")
    fig.tight_layout()
    _stamp_figure(fig, notebook_run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_dt_sweep(regime_sweep_dirs: dict[float, dict[str, dict[str, Path]]],
                  out_path: Path, notebook_run_id: str) -> None:
    """Money plot — accuracy vs eval-dt, 2×N layout: row 0 = count-preserving
    (upsample < + downsample > on one axis), row 1 = resample."""
    dt_trains = sorted(regime_sweep_dirs.keys())
    fig, axes = _matrix_axes(len(ROW_MODES), len(dt_trains))
    for j, dt_train in enumerate(dt_trains):
        ax = axes[0, j]
        for model in regime_sweep_dirs[dt_train]:
            mode_dirs = regime_sweep_dirs[dt_train][model]
            _plot_model_mode(ax, mode_dirs.get("upsample"),
                             MODEL_COLORS[model], MODEL_LABELS[model],
                             MODE_MARKERS["upsample"])
            _plot_model_mode(ax, mode_dirs.get("downsample"),
                             MODEL_COLORS[model], None,
                             MODE_MARKERS["downsample"])
        ax.axvline(dt_train, color=theme.DANGER, linestyle="--", linewidth=1,
                   label=f"train dt={dt_train}")
        ax.set_ylim(0, 100)
        ax.set_title(f"train dt = {dt_train} ms")
        if j == 0:
            ax.set_ylabel("count-preserving\ntest acc (%)")
            ax.legend(frameon=False, fontsize=7, loc="lower right")
            ax.add_artist(ax.legend(handles=_mode_legend_handles(),
                                    frameon=False, fontsize=6,
                                    loc="lower left"))
            ax.legend(frameon=False, fontsize=7, loc="lower right")
        ax.grid(alpha=0.3)

        ax = axes[1, j]
        for model in regime_sweep_dirs[dt_train]:
            _plot_model_mode(ax,
                             regime_sweep_dirs[dt_train][model].get("resample"),
                             MODEL_COLORS[model], MODEL_LABELS[model],
                             MODE_MARKERS["resample"])
        ax.axvline(dt_train, color=theme.DANGER, linestyle="--", linewidth=1)
        ax.set_ylim(0, 100)
        ax.set_xlabel("eval dt (ms)")
        if j == 0:
            ax.set_ylabel("resample\ntest acc (%)")
        ax.grid(alpha=0.3)
    fig.suptitle("Δt-stability: accuracy vs eval-dt")
    fig.tight_layout()
    _stamp_figure(fig, notebook_run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _format_duration(seconds: float) -> str:
    s = int(round(seconds))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60:02d}s"
    return f"{s // 3600}h {(s % 3600) // 60:02d}m"


def _render_stamp_png(notebook_run_id: str, stamp_path: Path) -> None:
    fig = plt.figure(figsize=(2.8, 0.28), dpi=150)
    fig.patch.set_alpha(0.0)
    fig.text(0.97, 0.5, notebook_run_id, ha="right", va="center",
             fontsize=10, color="white", family="monospace",
             bbox=dict(facecolor="black", alpha=0.55, pad=3, edgecolor="none"))
    stamp_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stamp_path, transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _copy_with_stamp(src: Path, dst: Path, stamp_path: Path) -> None:
    sh.ffmpeg(
        "-y", "-i", str(src), "-i", str(stamp_path),
        "-filter_complex", "[0:v][1:v]overlay=W-w-10:H-h-10",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "veryfast", "-crf", "20",
        "-movflags", "+faststart",
        str(dst),
        _out=sys.stdout, _err=sys.stderr,
    )
    print(f"wrote {dst.relative_to(REPO)}")


def copy_videos(regime_train_dirs: dict[float, dict[str, Path]],
                regime_sweep_dirs: dict[float, dict[str, dict[str, Path]]],
                out_dir: Path, notebook_run_id: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp_path = out_dir / "_stamp.png"
    _render_stamp_png(notebook_run_id, stamp_path)
    for dt_train, train_dirs in regime_train_dirs.items():
        for model, run_dir in train_dirs.items():
            src = training_video_path(run_dir)
            if not src.exists():
                raise SystemExit(f"missing training video: {src}")
            dst = out_dir / f"training_{_regime_key(dt_train)}_{model}.mp4"
            _copy_with_stamp(src, dst, stamp_path)
    # Only resample emits sweep videos (see sweep_model).
    for dt_train, model_modes in regime_sweep_dirs.items():
        for model, mode_dirs in model_modes.items():
            sweep_dir = mode_dirs.get("resample")
            if sweep_dir is None:
                continue
            src = sweep_dir / "dt_sweep.mp4"
            if not src.exists():
                print(f"[warn] missing sweep video: {src}")
                continue
            dst = out_dir / f"dt_sweep_{_regime_key(dt_train)}_{model}.mp4"
            _copy_with_stamp(src, dst, stamp_path)
    stamp_path.unlink(missing_ok=True)


def write_numbers(regime_train_dirs: dict[float, dict[str, Path]],
                  regime_sweep_dirs: dict[float, dict[str, dict[str, Path]]],
                  out_path: Path, notebook_run_id: str,
                  duration_s: float, init_match: dict | None = None) -> dict:
    first_regime = next(iter(regime_train_dirs.values()))
    first_cfg = load_config(next(iter(first_regime.values())))
    summary: dict[str, dict] = {
        "notebook_run_id": notebook_run_id,
        "git_sha": first_cfg.get("git_sha"),
        "duration_s": round(duration_s, 1),
        "duration": _format_duration(duration_s),
        "init_match": init_match or {},
        "config": {
            "tier": TIER,
            "dataset": "mnist",
            "max_samples": TIER_CONFIG[TIER]["max_samples"],
            "epochs": TIER_CONFIG[TIER]["epochs"],
            "t_ms": first_cfg["t_ms"],
            "dt_trains": DT_TRAINS,
            "dt_sweep": DT_SWEEP,
            "dt_sweeps": {str(k): v for k, v in DT_SWEEPS.items()},
            "encoder_modes": ENCODER_MODES,
            "n_hidden": first_cfg["n_hidden"],
            "batch_size": first_cfg["batch_size"],
            "lr": first_cfg["lr"],
            "kaiming_init": True,
            "seed": SEED,
        },
        "regimes": {},
    }
    for dt_train in sorted(regime_train_dirs.keys()):
        train_dirs = regime_train_dirs[dt_train]
        model_modes = regime_sweep_dirs[dt_train]
        runs: dict[str, dict] = {}
        for model in train_dirs:
            metrics = load_metrics(train_dirs[model])
            cfg = load_config(train_dirs[model])
            sw, sb = init_scales_for(model, dt_train)
            per_mode: dict[str, dict] = {}
            for mode, sweep_dir in model_modes[model].items():
                sweep = load_sweep(sweep_dir)
                ref = next((r for r in sweep["sweep"] if r["dt"] == dt_train), None)
                accs = [r["acc"] for r in sweep["sweep"]]
                rates = [r["hid_rate_hz"] for r in sweep["sweep"]
                         if r.get("hid_rate_hz") is not None]
                per_mode[mode] = {
                    "sweep": sweep["sweep"],
                    "ref_acc": ref["acc"] if ref else None,
                    "sweep_min_acc": min(accs) if accs else None,
                    "sweep_max_acc": max(accs) if accs else None,
                    "ref_hid_rate_hz": ref.get("hid_rate_hz") if ref else None,
                    "sweep_min_hid_rate_hz": min(rates) if rates else None,
                    "sweep_max_hid_rate_hz": max(rates) if rates else None,
                }
            runs[model] = {
                "label": MODEL_LABELS[model],
                "run_date": run_date(train_dirs[model]),
                "run_id": cfg.get("run_id"),
                "git_sha": cfg.get("git_sha"),
                "init_scale_weight": sw,
                "init_scale_bias": sb,
                "best_acc": metrics["best_acc"],
                "best_epoch": metrics["best_epoch"],
                "final_acc": metrics["epochs"][-1]["acc"],
                "final_loss": metrics["epochs"][-1]["loss"],
                "final_rate_e": metrics["epochs"][-1].get("rate_e"),
                "total_elapsed_s": metrics["total_elapsed_s"],
                "encoder_modes": per_mode,
            }
        summary["regimes"][str(dt_train)] = {
            "dt_train": dt_train,
            "regime_key": _regime_key(dt_train),
            "runs": runs,
        }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def _primary_mode_for(dt_train: float) -> str:
    """Which count-preserving mode covers most of this regime's eval-dt
    sweep. Matches the MDX's snn{1,2}Cp selection so prose and machine
    criteria read from the same series."""
    return "downsample" if dt_train <= 0.5 else "upsample"


def _rate_ratio(mode_entry: dict) -> float | None:
    lo = mode_entry.get("sweep_min_hid_rate_hz")
    hi = mode_entry.get("sweep_max_hid_rate_hz")
    if not isinstance(lo, (int, float)) or not isinstance(hi, (int, float)):
        return None
    # Floor min at 0.5 Hz so near-silent sweep points don't collapse the
    # denominator to zero. For the flat-models check this makes no
    # practical difference (hi is small, ratio stays small); for the
    # sag-models check it means a 0→N Hz fan still registers as ~2N×
    # rather than being dropped as no-data.
    return hi / max(lo, 0.5)


def evaluate_success(figures_dir: Path, summary: dict) -> list[dict]:
    """Machine-checked version of the prose in nb010.mdx § Success criteria.

    Two layers of gate:

    (1) Per-run training-health checks, matching nb005–nb009 shape
        (shared _per_model.evaluate_success): every one of the 5 models ×
        2 dts must have final_acc above the tier's chance-floor, hidden
        rate_e in the 1–200 Hz band, and no late-epoch collapse. Without
        these, a sweep where every run sat at chance could still pass
        the sweep-level ratio checks below.

    (2) Sweep-level structural checks: (a) the three headline figures
        rendered, (b) cuba/coba/ping hold hidden rate flat across
        eval-dt on the regime's primary count-preserving mode
        (ratio ≤ FLAT_MAX), (c) standard-snn and snntorch-library sag
        on the same axis (ratio ≥ SAG_MIN), (d) pinglab parity —
        standard-snn best-acc within PARITY_TOL pts of snntorch-library
        in both regimes.

    Each is a hard gate; the prose paragraph above the table keeps the
    nuance.
    """
    dataset_root = figures_dir.parents[2]  # src/docs/public
    criteria: list[dict] = []

    # ── Per-run training-health (layer 1) ──
    # Matches the per-model runner thresholds in _per_model.py.
    PER_RUN_RATE_MIN_HZ = 1.0
    PER_RUN_RATE_MAX_HZ = 200.0
    PER_RUN_COLLAPSE_TOL_PP = 5.0
    # Tier floors match _per_model.DEFAULT_MIN_ACC. Default to the
    # lowest if tier is not recorded (older runs).
    TIER_FLOORS = {"extra small": 15.0, "small": 30.0, "medium": 50.0,
                   "large": 70.0, "extra large": 70.0}
    tier = summary.get("tier") or summary.get("config", {}).get("tier")
    floor = TIER_FLOORS.get(tier, 15.0)

    regimes_all = summary.get("regimes", {})
    acc_fails: list[str] = []
    rate_fails: list[str] = []
    collapse_fails: list[str] = []
    acc_detail: list[str] = []
    rate_detail: list[str] = []
    collapse_detail: list[str] = []
    for dt_key, regime in regimes_all.items():
        dt_train = regime.get("dt_train", float(dt_key))
        for m, r in regime.get("runs", {}).items():
            tag = f"{m}@dt{dt_train}"
            best = r.get("best_acc")
            final = r.get("final_acc")
            rate = r.get("final_rate_e")
            if isinstance(final, (int, float)):
                acc_detail.append(f"{tag} {final:.0f}%")
                if final < floor:
                    acc_fails.append(f"{tag}:{final:.0f}%<{floor:.0f}%")
            else:
                acc_fails.append(f"{tag}:no-data")
            if isinstance(rate, (int, float)):
                rate_detail.append(f"{tag} {rate:.1f}Hz")
                if not (PER_RUN_RATE_MIN_HZ <= rate <= PER_RUN_RATE_MAX_HZ):
                    rate_fails.append(f"{tag}:{rate:.1f}Hz")
            else:
                rate_fails.append(f"{tag}:no-data")
            if isinstance(best, (int, float)) and isinstance(final, (int, float)):
                delta = best - final
                collapse_detail.append(f"{tag} Δ={delta:.0f}pp")
                if delta > PER_RUN_COLLAPSE_TOL_PP:
                    collapse_fails.append(f"{tag}:Δ={delta:.0f}pp")

    criteria.append({
        "label": f"all runs final-acc ≥ {floor:.0f}% ({tier or 'unknown'} tier floor)",
        "passed": not acc_fails,
        "detail": "; ".join(acc_detail) if acc_detail else "no data",
    })
    criteria.append({
        "label": f"all runs hidden rate in band ({PER_RUN_RATE_MIN_HZ:g}–{PER_RUN_RATE_MAX_HZ:g} Hz)",
        "passed": not rate_fails,
        "detail": "; ".join(rate_detail) if rate_detail else "no data",
    })
    criteria.append({
        "label": f"no collapse (final ≥ best − {PER_RUN_COLLAPSE_TOL_PP:.0f}pp) for all runs",
        "passed": not collapse_fails,
        "detail": "; ".join(collapse_detail) if collapse_detail else "no data",
    })

    # ── Sweep-level structural checks (layer 2) ──

    for fname, label in (
        ("training_curves.png", "training curves rendered"),
        ("dt_sweep.png", "dt-sweep accuracy plot rendered"),
        ("firing_rates.png", "firing-rate plot rendered"),
    ):
        p = figures_dir / fname
        ok = p.exists() and p.stat().st_size > 0
        criteria.append({
            "label": label,
            "passed": bool(ok),
            "detail": f"{p.name} ({p.stat().st_size} bytes)" if ok
                      else f"missing {p.name}",
            "detail_href": "/" + str(p.relative_to(dataset_root)) if ok else None,
        })

    regimes = summary.get("regimes", {})
    flat_models = ("cuba", "coba", "ping")
    sag_models = ("standard-snn", "snntorch-library")
    # Thresholds are deliberately loose relative to the prose: they gate
    # against regression ("ping holding gamma across eval-dt collapsed",
    # "snnTorch paths became flat") rather than enforcing the tier-small
    # numeric headline, which varies run-to-run.
    FLAT_MAX = 3.0
    SAG_MIN = 10.0
    PARITY_TOL = 15.0

    flat_fails: list[str] = []
    sag_fails: list[str] = []
    parity_fails: list[str] = []
    flat_detail: list[str] = []
    sag_detail: list[str] = []
    parity_detail: list[str] = []

    for dt_key, regime in regimes.items():
        dt_train = regime.get("dt_train", float(dt_key))
        mode = _primary_mode_for(dt_train)
        runs = regime.get("runs", {})

        for m in flat_models:
            entry = runs.get(m, {}).get("encoder_modes", {}).get(mode, {})
            ratio = _rate_ratio(entry)
            if ratio is None:
                flat_fails.append(f"{m}@dt={dt_train}:no-data")
                continue
            flat_detail.append(f"{m}@dt{dt_train} {ratio:.2f}×")
            if ratio > FLAT_MAX:
                flat_fails.append(f"{m}@dt={dt_train}:{ratio:.2f}×>{FLAT_MAX}×")

        for m in sag_models:
            entry = runs.get(m, {}).get("encoder_modes", {}).get(mode, {})
            ratio = _rate_ratio(entry)
            if ratio is None:
                sag_fails.append(f"{m}@dt={dt_train}:no-data")
                continue
            sag_detail.append(f"{m}@dt{dt_train} {ratio:.1f}×")
            if ratio < SAG_MIN:
                sag_fails.append(f"{m}@dt={dt_train}:{ratio:.1f}×<{SAG_MIN}×")

        snn_acc = runs.get("standard-snn", {}).get("best_acc")
        lib_acc = runs.get("snntorch-library", {}).get("best_acc")
        if isinstance(snn_acc, (int, float)) and isinstance(lib_acc, (int, float)):
            diff = abs(snn_acc - lib_acc)
            parity_detail.append(f"dt{dt_train} Δ={diff:.1f}pt")
            if diff > PARITY_TOL:
                parity_fails.append(f"dt={dt_train}:Δ={diff:.1f}>{PARITY_TOL}pt")
        else:
            parity_fails.append(f"dt={dt_train}:no-data")

    criteria.append({
        "label": f"cuba/coba/ping rate-flat (ratio ≤ {FLAT_MAX:g}×) on count-preserving sweep",
        "passed": not flat_fails,
        "detail": "; ".join(flat_detail) if flat_detail else "no data",
    })
    criteria.append({
        "label": f"snnTorch paths sag (ratio ≥ {SAG_MIN:g}×) on count-preserving sweep",
        "passed": not sag_fails,
        "detail": "; ".join(sag_detail) if sag_detail else "no data",
    })
    criteria.append({
        "label": f"pinglab parity: standard-snn vs snntorch-library best-acc within {PARITY_TOL:g} pt",
        "passed": not parity_fails,
        "detail": "; ".join(parity_detail) if parity_detail else "no data",
    })
    return criteria


def _print_and_gate(success_criteria: list[dict]) -> None:
    for c in success_criteria:
        mark = "pass" if c["passed"] else "FAIL"
        print(f"  [{mark}] {c['label']} — {c['detail']}")
    if any(not c["passed"] for c in success_criteria):
        sys.exit(1)


def evaluate_only() -> None:
    numbers_path = FIGURES / "numbers.json"
    if not numbers_path.exists():
        raise SystemExit(f"--evaluate-success-only requires existing "
                         f"{numbers_path.relative_to(REPO)}")
    summary = json.loads(numbers_path.read_text())
    summary["success_criteria"] = evaluate_success(FIGURES, summary)
    numbers_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"rewrote {numbers_path.relative_to(REPO)} (success_criteria only)")
    _print_and_gate(summary["success_criteria"])


def main() -> None:
    global TIER
    if "--evaluate-success-only" in sys.argv:
        evaluate_only()
        return
    skip_training = "--skip-training" in sys.argv
    wipe_dir = "--no-wipe-dir" not in sys.argv
    modal_gpu = parse_modal_gpu(sys.argv)
    TIER = parse_tier(sys.argv, choices=TIER_CONFIG.keys(), default=DEFAULT_TIER)
    t_start = time.monotonic()
    notebook_run_id = next_run_id(SLUG)
    print(f"notebook_run_id = {notebook_run_id} tier={TIER}"
          + ("  [skip-training]" if skip_training else ""))
    if wipe_dir:
        if skip_training:
            # Preserve train dirs; wipe sweep subdirs + figures so they regenerate.
            for dt_train in DT_TRAINS:
                for model in MODELS:
                    for mode in ENCODER_MODES:
                        sd = ARTIFACTS / _regime_key(dt_train) / model / f"sweep_{_mode_key(mode)}"
                        if sd.exists():
                            print(f"[wipe] {sd.relative_to(REPO)}")
                            shutil.rmtree(sd)
            if FIGURES.exists():
                print(f"[wipe] {FIGURES.relative_to(REPO)}")
                shutil.rmtree(FIGURES)
        else:
            for d in (ARTIFACTS, FIGURES):
                if d.exists():
                    print(f"[wipe] {d.relative_to(REPO)}")
                    shutil.rmtree(d)
    FIGURES.mkdir(parents=True, exist_ok=True)
    persist_run_id(SLUG, notebook_run_id)

    init_match = verify_init_match(MODELS, SEED, DT_TRAINS)

    dispatcher = BatchDispatcher(modal_gpu, REPO, OSCILLOSCOPE)

    regime_train_dirs: dict[float, dict[str, Path]] = {}
    # regime_sweep_dirs[dt_train][model][encoder_mode] = sweep_dir
    regime_sweep_dirs: dict[float, dict[str, dict[str, Path]]] = {}

    # Phase 1: enqueue every train cell across all dt regimes, then
    # drain once so Modal runs them all in parallel under a single
    # app.run() lifecycle. (Local mode runs each submit() synchronously
    # so drain() is a no-op.)
    for dt_train in DT_TRAINS:
        print(f"\n=== regime: train dt = {dt_train} ms ===")
        if skip_training:
            td = {m: ARTIFACTS / _regime_key(dt_train) / m / "train" for m in MODELS}
            for m, d in td.items():
                if not (d / "weights.pth").exists():
                    raise SystemExit(
                        f"--skip-training requires existing train weights at {d}")
        else:
            td = {m: train_model(m, dt_train, dispatcher) for m in MODELS}
        regime_train_dirs[dt_train] = td
    dispatcher.drain()

    for dt_train, td in regime_train_dirs.items():
        for m, d in td.items():
            if not (d / "metrics.json").exists():
                raise SystemExit(f"training did not produce {d / 'metrics.json'}")
            if not training_video_path(d).exists():
                raise SystemExit(f"training did not produce {training_video_path(d)}")

    # Phase 2: enqueue every dt-sweep cell, then drain.
    for dt_train in DT_TRAINS:
        td = regime_train_dirs[dt_train]
        sd: dict[str, dict[str, Path]] = {}
        for m in MODELS:
            sd[m] = {mode: sweep_model(m, dt_train, td[m], mode, dispatcher)
                     for mode in ENCODER_MODES}
        regime_sweep_dirs[dt_train] = sd
    dispatcher.drain()

    for dt_train, sd in regime_sweep_dirs.items():
        for m, modes in sd.items():
            for mode, d in modes.items():
                if not (d / "results.json").exists():
                    raise SystemExit(f"dt-sweep did not produce {d / 'results.json'}")

    plot_training_curves(regime_train_dirs, FIGURES / "training_curves.png",
                         notebook_run_id)
    print(f"wrote {(FIGURES / 'training_curves.png').relative_to(REPO)}")
    plot_dt_sweep(regime_sweep_dirs, FIGURES / "dt_sweep.png", notebook_run_id)
    print(f"wrote {(FIGURES / 'dt_sweep.png').relative_to(REPO)}")
    plot_firing_rates(regime_sweep_dirs, FIGURES / "firing_rates.png",
                      notebook_run_id)
    fr = FIGURES / "firing_rates.png"
    if fr.exists():
        print(f"wrote {fr.relative_to(REPO)}")
    copy_videos(regime_train_dirs, regime_sweep_dirs, FIGURES, notebook_run_id)

    numbers_path = FIGURES / "numbers.json"
    duration_s = time.monotonic() - t_start
    summary = write_numbers(regime_train_dirs, regime_sweep_dirs, numbers_path,
                            notebook_run_id, duration_s,
                            init_match=init_match)
    summary["success_criteria"] = evaluate_success(FIGURES, summary)
    numbers_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {numbers_path.relative_to(REPO)}")
    for dt_train_s, regime in summary["regimes"].items():
        print(f"  regime dt={dt_train_s}:")
        for model, s in regime["runs"].items():
            print(f"    {model}: best={s['best_acc']}%  "
                  f"final={s['final_acc']}%  "
                  f"elapsed={s['total_elapsed_s']:.0f}s")
            for mode, m in s["encoder_modes"].items():
                print(f"      [{mode}] ref={m['ref_acc']}%  "
                      f"sweep=[{m['sweep_min_acc']}..{m['sweep_max_acc']}]%")
    print(f"  total duration: {summary['duration']}")
    _print_and_gate(summary["success_criteria"])


if __name__ == "__main__":
    main()
    sys.exit(0)
