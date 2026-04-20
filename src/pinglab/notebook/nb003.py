"""Notebook runner for entry 003 — cuba vs snntorch-clone Δt-stability.

Trains *snntorch-clone*, *snntorch-library*, and *cuba* at two training
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

Notebook entry: src/docs/src/pages/notebook/nb003.mdx
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

from _run_id import next_run_id, persist as persist_run_id  # noqa: E402
from config import build_net  # noqa: E402

SLUG = "nb003"
ARTIFACTS = REPO / "src" / "artifacts" / "notebook" / SLUG
FIGURES = REPO / "src" / "docs" / "public" / "figures" / "notebook" / SLUG
OSCILLOSCOPE = REPO / "src" / "pinglab" / "oscilloscope.py"

MODELS = ["snntorch-clone", "snntorch-library", "cuba"]
# Two input-transport modes for the dt sweep, matching the paper
# (Parthasarathy et al. §2.1 / Fig 1B / §2.3). The FrozenEncoder anchors
# at train-dt and transports count-preservingly under `zero-pad` (zeros
# between spikes to finer eval-dt per Fig 1B, sum-pool to coarser per
# §2.3) or draws fresh per target under `resample`.
ENCODER_MODES = ["zero-pad", "resample"]
MAX_SAMPLES = 500
EPOCHS = 5
T_MS = 600.0
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
TIER = "small"  # see src/docs/src/pages/llm-conventions.md § 8 Run sizing tiers
TAU_MEM_MS = 10.0  # matches models.py SNN_TAU_MEM_MS

# Per-step drive compensation for cuba at training dt. cuba's update is
#   mem = β·mem + (1-β)/dt · (W·s) + (1-β) · b      (β = exp(-dt/τ_mem))
# so at training-dt the spike drive is scaled by (1-β)/dt ≈ 0.099 and the
# bias drive by (1-β) ≈ 0.025, both ≪ 1. Starting cuba from the same random
# weights as snntorch-clone leaves it ~10× below threshold for spike drive
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
    so it starts at the same per-step effective drive as snntorch-clone at
    training-dt."""
    if model == "cuba":
        return cuba_init_scales(dt_train)
    return (1.0, 1.0)

MODEL_LABELS = {
    "snntorch-clone":   "snntorch-clone",
    "snntorch-library": "snntorch-library",
    "cuba":             "cuba",
}
MODEL_COLORS = {
    "snntorch-clone":   "#1f77b4",
    "snntorch-library": "#ff7f0e",
    "cuba":             "#2ca02c",
}


def training_video_path(out_dir: Path) -> Path:
    return out_dir / "training.mp4"


# Models sharing the CUBANet class — within this family, matched seed +
# --kaiming-init produces bit-identical raw weights because every parameter is
# allocated in the same order. snntorch-library goes through nn.Linear, which
# uses a different kaiming_uniform_ convention, so only biases coincidentally
# match. Treat library's init as independent and parity-tested at run time.
SNNTORCHNET_FAMILY = {"snntorch-clone", "cuba"}


def verify_init_match(models: list[str], seed: int,
                      dt_trains: list[float]) -> dict:
    """Preflight: CUBANet-family models must start from the same random
    weights, modulo each model's per-step drive scaling. snntorch-clone and
    cuba share the CUBANet class and with matched seed allocate every
    tensor in the same order, giving bit-identical raw weights. cuba then
    gets a one-shot multiplier (init_scales_for) per training dt so its
    per-step drive matches snntorch-clone at that dt. snntorch-library
    uses nn.Linear's own kaiming_uniform_ and is reported but not asserted
    — its role is an external parity reference, not a bit-match."""
    nets: dict[str, object] = {}
    for m in models:
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
        scales = ", ".join(
            f"{m}(W×{init_scales_for(m, dt)[0]:.3f} "
            f"b×{init_scales_for(m, dt)[1]:.3f})"
            for m in models)
        print(f"[init-match] dt={dt}: per-model init_scale: {scales}")
    return report


def _regime_key(dt_train: float) -> str:
    """Filesystem-safe label for a training regime (e.g. 0.1 → 'dt0.1')."""
    return f"dt{dt_train:g}"


def train_model(model: str, dt_train: float) -> Path:
    """Train at dt_train with per-epoch video observation."""
    out_dir = ARTIFACTS / _regime_key(dt_train) / model / "train"
    sw, sb = init_scales_for(model, dt_train)
    print(f"[{model} @ dt={dt_train}] training → {out_dir.relative_to(REPO)} "
          f"(init_scale W×{sw:.3f} b×{sb:.3f})")
    sh.uv(
        "run", "python", str(OSCILLOSCOPE), "train",
        "--model", model,
        "--kaiming-init",
        "--init-scale-weight", f"{sw}",
        "--init-scale-bias", f"{sb}",
        "--dataset", "mnist",
        "--max-samples", str(MAX_SAMPLES),
        "--epochs", str(EPOCHS),
        "--t-ms", str(T_MS),
        "--dt", str(dt_train),
        "--seed", str(SEED),
        "--observe", "video",
        "--frame-rate", "1",
        "--out-dir", str(out_dir),
        "--wipe-dir",
        _cwd=str(REPO),
        _out=sys.stdout,
        _err=sys.stderr,
    )
    metrics_path = out_dir / "metrics.json"
    if not metrics_path.exists():
        raise SystemExit(f"training did not produce {metrics_path}")
    if not training_video_path(out_dir).exists():
        raise SystemExit(f"training did not produce {training_video_path(out_dir)}")
    return out_dir


def _mode_key(encoder_mode: str) -> str:
    """Filesystem-safe label for an encoder mode (e.g. 'count-pool' → 'count_pool')."""
    return encoder_mode.replace("-", "_")


def sweep_model(model: str, dt_train: float, train_dir: Path,
                encoder_mode: str) -> Path:
    """Run dt-sweep inference against trained weights with a chosen frozen-input
    transport mode. Same underlying spike pattern across dt values (how it's
    transported to coarser dt depends on mode) so the LIF dynamics are the
    only thing that differs between sweep points."""
    sweep_dir = ARTIFACTS / _regime_key(dt_train) / model / f"sweep_{_mode_key(encoder_mode)}"
    sweep_grid = DT_SWEEPS[dt_train]
    print(f"[{model} @ dt={dt_train}, mode={encoder_mode}] dt-sweep → "
          f"{sweep_dir.relative_to(REPO)}")
    # Videos are only emitted for the zero-pad mode to keep wall-clock and
    # disk within a small-tier budget; resample shares identical plotting
    # paths so its videos add no additional scientific content.
    cmd = [
        "run", "python", str(OSCILLOSCOPE), "infer",
        "--from-dir", str(train_dir),
        "--dt-sweep", *[str(d) for d in sweep_grid],
        "--frozen-inputs-mode", encoder_mode,
        "--max-samples", str(MAX_SAMPLES),
        "--out-dir", str(sweep_dir),
        "--wipe-dir",
    ]
    if encoder_mode == "zero-pad":
        cmd += ["--observe", "video"]
    sh.uv(*cmd, _cwd=str(REPO), _out=sys.stdout, _err=sys.stderr)
    results_path = sweep_dir / "results.json"
    if not results_path.exists():
        raise SystemExit(f"dt-sweep did not produce {results_path}")
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
             fontsize=7, color="#888888", family="monospace")


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
    y across all panels (accuracy or firing-rate, both unitful)."""
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(8, 4.5 * n_rows / 2),
                             sharey=True, squeeze=False)
    return fig, axes


def plot_firing_rates(regime_sweep_dirs: dict[float, dict[str, dict[str, Path]]],
                      out_path: Path, notebook_run_id: str) -> None:
    """Mean hidden-layer firing rate vs eval-dt, matrix layout:
    rows = encoder mode, cols = training regime."""
    dt_trains = sorted(regime_sweep_dirs.keys())
    fig, axes = _matrix_axes(len(ENCODER_MODES), len(dt_trains))
    any_data = False
    for i, mode in enumerate(ENCODER_MODES):
        for j, dt_train in enumerate(dt_trains):
            ax = axes[i, j]
            for model, mode_dirs in regime_sweep_dirs[dt_train].items():
                sweep_dir = mode_dirs.get(mode)
                if sweep_dir is None:
                    continue
                blob = load_sweep(sweep_dir)
                dts, rates = [], []
                for r in blob["sweep"]:
                    rate = r.get("hid_rate_hz")
                    if rate is None:
                        continue
                    dts.append(r["dt"])
                    rates.append(rate)
                if not dts:
                    continue
                any_data = True
                ax.plot(dts, rates, marker="o",
                        color=MODEL_COLORS[model], label=MODEL_LABELS[model])
            ax.axvline(dt_train, color="#cc4444", linestyle="--", linewidth=1,
                       label=f"train dt={dt_train}")
            if i == 0:
                ax.set_title(f"train dt = {dt_train} ms")
            if i == len(ENCODER_MODES) - 1:
                ax.set_xlabel("eval dt (ms)")
            if j == 0:
                ax.set_ylabel(f"{mode}\nhidden rate (Hz)")
            ax.grid(alpha=0.3)
            if i == 0 and j == 0:
                ax.legend(frameon=False, fontsize=7)
    if not any_data:
        print("[warn] no firing-rate data in sweep results; skipping firing_rates.png")
        plt.close(fig)
        return
    fig.suptitle("Δt-stability: hidden firing rate vs eval-dt (per encoder mode)")
    fig.tight_layout()
    _stamp_figure(fig, notebook_run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_dt_sweep(regime_sweep_dirs: dict[float, dict[str, dict[str, Path]]],
                  out_path: Path, notebook_run_id: str) -> None:
    """Money plot — accuracy vs eval-dt, matrix layout: rows = encoder mode,
    cols = training regime. Each panel marks its training dt."""
    dt_trains = sorted(regime_sweep_dirs.keys())
    fig, axes = _matrix_axes(len(ENCODER_MODES), len(dt_trains))
    for i, mode in enumerate(ENCODER_MODES):
        for j, dt_train in enumerate(dt_trains):
            ax = axes[i, j]
            for model, mode_dirs in regime_sweep_dirs[dt_train].items():
                sweep_dir = mode_dirs.get(mode)
                if sweep_dir is None:
                    continue
                blob = load_sweep(sweep_dir)
                dts = [r["dt"] for r in blob["sweep"]]
                accs = [r["acc"] for r in blob["sweep"]]
                ax.plot(dts, accs, marker="o",
                        color=MODEL_COLORS[model], label=MODEL_LABELS[model])
            ax.axvline(dt_train, color="#cc4444", linestyle="--", linewidth=1,
                       label=f"train dt={dt_train}")
            ax.set_ylim(0, 100)
            if i == 0:
                ax.set_title(f"train dt = {dt_train} ms")
            if i == len(ENCODER_MODES) - 1:
                ax.set_xlabel("eval dt (ms)")
            if j == 0:
                ax.set_ylabel(f"{mode}\ntest acc (%)")
            ax.grid(alpha=0.3)
            if i == 0 and j == 0:
                ax.legend(frameon=False, fontsize=7)
    fig.suptitle("Δt-stability: accuracy vs eval-dt (per encoder mode)")
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
    # Only zero-pad emits sweep videos (see sweep_model).
    for dt_train, model_modes in regime_sweep_dirs.items():
        for model, mode_dirs in model_modes.items():
            sweep_dir = mode_dirs.get("zero-pad")
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
        "duration_s": round(duration_s, 1),
        "duration": _format_duration(duration_s),
        "init_match": init_match or {},
        "config": {
            "tier": TIER,
            "dataset": "mnist",
            "max_samples": MAX_SAMPLES,
            "epochs": EPOCHS,
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


def main() -> None:
    sweep_only = "--sweep-only" in sys.argv
    wipe_dir = "--no-wipe-dir" not in sys.argv
    t_start = time.monotonic()
    notebook_run_id = next_run_id(SLUG)
    print(f"notebook_run_id = {notebook_run_id}"
          + ("  [sweep-only]" if sweep_only else ""))
    if wipe_dir:
        if sweep_only:
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

    regime_train_dirs: dict[float, dict[str, Path]] = {}
    # regime_sweep_dirs[dt_train][model][encoder_mode] = sweep_dir
    regime_sweep_dirs: dict[float, dict[str, dict[str, Path]]] = {}
    for dt_train in DT_TRAINS:
        print(f"\n=== regime: train dt = {dt_train} ms ===")
        if sweep_only:
            td = {m: ARTIFACTS / _regime_key(dt_train) / m / "train" for m in MODELS}
            for m, d in td.items():
                if not (d / "weights.pth").exists():
                    raise SystemExit(
                        f"--sweep-only requires existing train weights at {d}")
        else:
            td = {m: train_model(m, dt_train) for m in MODELS}
        sd: dict[str, dict[str, Path]] = {}
        for m in MODELS:
            sd[m] = {mode: sweep_model(m, dt_train, td[m], mode)
                     for mode in ENCODER_MODES}
        regime_train_dirs[dt_train] = td
        regime_sweep_dirs[dt_train] = sd

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


if __name__ == "__main__":
    main()
    sys.exit(0)
