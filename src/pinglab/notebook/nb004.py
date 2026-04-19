"""Notebook runner for entry 004 — cuba vs snntorch-clone Δt-stability.

Trains *snntorch-clone* and *cuba* at a canonical training *dt*, then runs
inference at a sweep of *dt* values for each. *cuba* applies $(1-\beta)/dt$
drive scaling and should hold accuracy across eval-*dt*; *snntorch-clone*
deliberately does not, so its accuracy should sag as eval-*dt* departs
from train-*dt*. This is the first rung-to-rung ablation on the [CUBA
ladder](/models/#the-ladder).

Writes:
  * training_curves.png — test accuracy & train loss per epoch (both models)
  * dt_sweep.png — accuracy vs eval-dt overlay (the money plot)
  * firing_rates.png — mean hidden-layer firing rate vs eval-dt (overlay)
  * training_snntorch-clone.mp4, training_cuba.mp4 — per-epoch videos
  * numbers.json — config + per-model best/final + sweep results

Notebook entry: src/docs/src/pages/notebook/nb004.mdx
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

SLUG = "nb004"
ARTIFACTS = REPO / "src" / "artifacts" / "notebook" / SLUG
FIGURES = REPO / "src" / "docs" / "public" / "figures" / "notebook" / SLUG
OSCILLOSCOPE = REPO / "src" / "pinglab" / "oscilloscope.py"

MODELS = ["snntorch-clone", "snntorch-library", "cuba"]
MAX_SAMPLES = 200
EPOCHS = 3
T_MS = 600.0
DT_TRAIN = 0.1
# Sweep grid: below train-dt (finer), at train-dt (control), and above (coarser).
# Integer ratios only, so FrozenEncoder OR-pool downsampling stays valid.
DT_SWEEP = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
SEED = 42
TIER = "extra small"  # see src/docs/src/pages/llm-conventions.md § 8 Run sizing tiers
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

_cuba_w, _cuba_b = cuba_init_scales(DT_TRAIN)
MODEL_INIT_SCALE = {
    "snntorch-clone":   (1.0, 1.0),
    "snntorch-library": (1.0, 1.0),  # canonical discretisation, same as clone
    "cuba":             (_cuba_w, _cuba_b),
}

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


# Models sharing the SNNTorchNet class — within this family, matched seed +
# --kaiming-init produces bit-identical raw weights because every parameter is
# allocated in the same order. snntorch-library goes through nn.Linear, which
# uses a different kaiming_uniform_ convention, so only biases coincidentally
# match. Treat library's init as independent and parity-tested at run time.
SNNTORCHNET_FAMILY = {"snntorch-clone", "cuba"}


def verify_init_match(models: list[str], seed: int) -> dict:
    """Preflight: SNNTorchNet-family models must start from the same random
    weights, modulo each model's per-step drive scaling. snntorch-clone and
    cuba share the SNNTorchNet class and with matched seed allocate every
    tensor in the same order, giving bit-identical raw weights. cuba then
    gets a one-shot multiplier (MODEL_INIT_SCALE) so its per-step drive
    matches snntorch-clone at training dt. snntorch-library uses nn.Linear's
    own kaiming_uniform_ and is reported but not asserted — its role is an
    external parity reference, not a bit-match."""
    nets: dict[str, object] = {}
    for m in models:
        torch.manual_seed(seed)
        nets[m] = build_net(m, kaiming_init=True, hidden_sizes=[1024])
    report: dict[str, object] = {
        "family": sorted(SNNTORCHNET_FAMILY & set(models)),
        "seed": seed,
        "scale": {m: {"weight": MODEL_INIT_SCALE[m][0],
                      "bias": MODEL_INIT_SCALE[m][1]} for m in models},
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
    scales = ", ".join(
        f"{m}(W×{MODEL_INIT_SCALE[m][0]:.3f} b×{MODEL_INIT_SCALE[m][1]:.3f})"
        for m in models)
    print(f"[init-match] per-model init_scale: {scales}")
    return report


def train_model(model: str) -> Path:
    """Train at DT_TRAIN with per-epoch video observation."""
    out_dir = ARTIFACTS / model / "train"
    sw, sb = MODEL_INIT_SCALE[model]
    print(f"[{model}] training → {out_dir.relative_to(REPO)} "
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
        "--dt", str(DT_TRAIN),
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


def sweep_model(model: str, train_dir: Path) -> Path:
    """Run dt-sweep inference against trained weights. Frozen-inputs so every
    dt sees the same underlying spike pattern (OR-pooled to coarser dt), which
    isolates the LIF dynamics as the only thing changing across the sweep.
    --observe video emits dt_sweep.mp4 — one SCOPE_FRAME per dt, the same
    network rendered across timesteps sizes."""
    sweep_dir = ARTIFACTS / model / "sweep"
    print(f"[{model}] dt-sweep → {sweep_dir.relative_to(REPO)}")
    sh.uv(
        "run", "python", str(OSCILLOSCOPE), "infer",
        "--from-dir", str(train_dir),
        "--dt-sweep", *[str(d) for d in DT_SWEEP],
        "--frozen-inputs",
        "--max-samples", str(MAX_SAMPLES),
        "--observe", "video",
        "--out-dir", str(sweep_dir),
        "--wipe-dir",
        _cwd=str(REPO),
        _out=sys.stdout,
        _err=sys.stderr,
    )
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


def plot_training_curves(train_dirs: dict[str, Path], out_path: Path,
                         notebook_run_id: str) -> None:
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(8, 4.5))
    for model, run_dir in train_dirs.items():
        metrics = load_metrics(run_dir)
        epochs = [e["ep"] for e in metrics["epochs"]]
        loss = [e["loss"] for e in metrics["epochs"]]
        acc = [e["acc"] for e in metrics["epochs"]]
        label = MODEL_LABELS[model]
        color = MODEL_COLORS[model]
        ax_loss.plot(epochs, loss, marker="o", color=color, label=label)
        ax_acc.plot(epochs, acc, marker="o", color=color, label=label)
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("train loss")
    ax_loss.set_title("Training loss")
    ax_loss.grid(alpha=0.3)
    ax_loss.legend(frameon=False)
    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylabel("test accuracy (%)")
    ax_acc.set_title("Test accuracy")
    ax_acc.grid(alpha=0.3)
    ax_acc.legend(frameon=False)
    fig.tight_layout()
    _stamp_figure(fig, notebook_run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_firing_rates(sweep_dirs: dict[str, Path], out_path: Path,
                      notebook_run_id: str) -> None:
    """Mean hidden-layer firing rate vs eval-dt. Reveals whether the
    Δt-stability gap in *dt_sweep.png* corresponds to a change in mean
    activity level (canonical path: drive scales with 1/dt, so rate
    should track with eval-dt) or stays flat (cuba: per-ms drive
    invariant, rate should be ~constant)."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    any_data = False
    for model, sweep_dir in sweep_dirs.items():
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
    if not any_data:
        print("[warn] no firing-rate data in sweep results; skipping firing_rates.png")
        plt.close(fig)
        return
    ax.axvline(DT_TRAIN, color="#cc4444", linestyle="--", linewidth=1,
               label=f"train dt={DT_TRAIN}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("eval dt (ms, log scale)")
    ax.set_ylabel("mean hidden firing rate (Hz, log scale)")
    ax.set_title("Δt-stability: hidden firing rate vs eval-dt (frozen inputs)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(frameon=False)
    fig.tight_layout()
    _stamp_figure(fig, notebook_run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_dt_sweep(sweep_dirs: dict[str, Path], out_path: Path,
                  notebook_run_id: str) -> None:
    """Money plot — accuracy vs eval-dt overlay for both models, with the
    training dt marked."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for model, sweep_dir in sweep_dirs.items():
        blob = load_sweep(sweep_dir)
        dts = [r["dt"] for r in blob["sweep"]]
        accs = [r["acc"] for r in blob["sweep"]]
        ax.plot(dts, accs, marker="o",
                color=MODEL_COLORS[model], label=MODEL_LABELS[model])
    ax.axvline(DT_TRAIN, color="#cc4444", linestyle="--", linewidth=1,
               label=f"train dt={DT_TRAIN}")
    ax.set_xscale("log")
    ax.set_xlabel("eval dt (ms, log scale)")
    ax.set_ylabel("test accuracy (%)")
    ax.set_title("Δt-stability: accuracy vs eval-dt (frozen inputs)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3, which="both")
    ax.legend(frameon=False)
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


def copy_videos(train_dirs: dict[str, Path], sweep_dirs: dict[str, Path],
                out_dir: Path, notebook_run_id: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp_path = out_dir / "_stamp.png"
    _render_stamp_png(notebook_run_id, stamp_path)
    for model, run_dir in train_dirs.items():
        src = training_video_path(run_dir)
        if not src.exists():
            raise SystemExit(f"missing training video: {src}")
        _copy_with_stamp(src, out_dir / f"training_{model}.mp4", stamp_path)
    for model, sweep_dir in sweep_dirs.items():
        src = sweep_dir / "dt_sweep.mp4"
        if not src.exists():
            # --observe video may have been skipped; don't hard-fail.
            print(f"[warn] missing sweep video: {src}")
            continue
        _copy_with_stamp(src, out_dir / f"dt_sweep_{model}.mp4", stamp_path)
    stamp_path.unlink(missing_ok=True)


def write_numbers(train_dirs: dict[str, Path], sweep_dirs: dict[str, Path],
                  out_path: Path, notebook_run_id: str,
                  duration_s: float, init_match: dict | None = None) -> dict:
    first_cfg = load_config(next(iter(train_dirs.values())))
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
            "dt_train": DT_TRAIN,
            "dt_sweep": DT_SWEEP,
            "n_hidden": first_cfg["n_hidden"],
            "batch_size": first_cfg["batch_size"],
            "lr": first_cfg["lr"],
            "kaiming_init": True,
            "seed": SEED,
        },
        "runs": {},
    }
    for model in train_dirs:
        metrics = load_metrics(train_dirs[model])
        cfg = load_config(train_dirs[model])
        sweep = load_sweep(sweep_dirs[model])
        ref = next((r for r in sweep["sweep"] if r["dt"] == DT_TRAIN), None)
        accs = [r["acc"] for r in sweep["sweep"]]
        rates = [r["hid_rate_hz"] for r in sweep["sweep"]
                 if r.get("hid_rate_hz") is not None]
        summary["runs"][model] = {
            "label": MODEL_LABELS[model],
            "run_date": run_date(train_dirs[model]),
            "run_id": cfg.get("run_id"),
            "git_sha": cfg.get("git_sha"),
            "best_acc": metrics["best_acc"],
            "best_epoch": metrics["best_epoch"],
            "final_acc": metrics["epochs"][-1]["acc"],
            "final_loss": metrics["epochs"][-1]["loss"],
            "total_elapsed_s": metrics["total_elapsed_s"],
            "sweep": sweep["sweep"],
            "ref_acc": ref["acc"] if ref else None,
            "sweep_min_acc": min(accs) if accs else None,
            "sweep_max_acc": max(accs) if accs else None,
            "ref_hid_rate_hz": ref.get("hid_rate_hz") if ref else None,
            "sweep_min_hid_rate_hz": min(rates) if rates else None,
            "sweep_max_hid_rate_hz": max(rates) if rates else None,
        }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    wipe_dir = "--no-wipe-dir" not in sys.argv
    t_start = time.monotonic()
    notebook_run_id = next_run_id(SLUG)
    print(f"notebook_run_id = {notebook_run_id}")
    if wipe_dir:
        for d in (ARTIFACTS, FIGURES):
            if d.exists():
                print(f"[wipe] {d.relative_to(REPO)}")
                shutil.rmtree(d)
    FIGURES.mkdir(parents=True, exist_ok=True)
    persist_run_id(SLUG, notebook_run_id)

    init_match = verify_init_match(MODELS, SEED)

    train_dirs = {m: train_model(m) for m in MODELS}
    sweep_dirs = {m: sweep_model(m, train_dirs[m]) for m in MODELS}

    plot_training_curves(train_dirs, FIGURES / "training_curves.png", notebook_run_id)
    print(f"wrote {(FIGURES / 'training_curves.png').relative_to(REPO)}")
    plot_dt_sweep(sweep_dirs, FIGURES / "dt_sweep.png", notebook_run_id)
    print(f"wrote {(FIGURES / 'dt_sweep.png').relative_to(REPO)}")
    plot_firing_rates(sweep_dirs, FIGURES / "firing_rates.png", notebook_run_id)
    fr = FIGURES / "firing_rates.png"
    if fr.exists():
        print(f"wrote {fr.relative_to(REPO)}")
    copy_videos(train_dirs, sweep_dirs, FIGURES, notebook_run_id)

    numbers_path = FIGURES / "numbers.json"
    duration_s = time.monotonic() - t_start
    summary = write_numbers(train_dirs, sweep_dirs, numbers_path,
                            notebook_run_id, duration_s,
                            init_match=init_match)
    print(f"wrote {numbers_path.relative_to(REPO)}")
    for model, s in summary["runs"].items():
        print(f"  {model}: best={s['best_acc']}%  final={s['final_acc']}%  "
              f"ref={s['ref_acc']}%  sweep=[{s['sweep_min_acc']}..{s['sweep_max_acc']}]%  "
              f"elapsed={s['total_elapsed_s']:.0f}s")
    print(f"  total duration: {summary['duration']}")


if __name__ == "__main__":
    main()
    sys.exit(0)
