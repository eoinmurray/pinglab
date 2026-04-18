"""Notebook runner for entry 003 — snnTorch calibration.

Runs a small, matched training of the in-repo `snntorch-clone` model and the
`snntorch-library` parity reference, then writes a training-curve comparison
figure and a numbers.json summary into the notebook's figures dir.

Notebook entry: src/docs/src/pages/notebook/nb003.mdx
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import sh

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src" / "pinglab"))

import matplotlib.pyplot as plt  # noqa: E402

from _run_id import next_run_id, persist as persist_run_id  # noqa: E402

SLUG = "nb003"
ARTIFACTS = REPO / "src" / "artifacts" / "notebook" / SLUG
FIGURES = REPO / "src" / "docs" / "public" / "figures" / "notebook" / SLUG
OSCILLOSCOPE = REPO / "src" / "pinglab" / "oscilloscope.py"

MODELS = ["snntorch-clone", "snntorch-library"]
MAX_SAMPLES = 5000
EPOCHS = 40
T_MS = 200.0
SEED = 42
TIER = "large"  # see src/docs/src/pages/llm-context.md § 8 Run sizing tiers

MODEL_LABELS = {
    "snntorch-clone": "pinglab snntorch-clone",
    "snntorch-library": "snnTorch library",
}
MODEL_COLORS = {
    "snntorch-clone": "#1f77b4",
    "snntorch-library": "#d62728",
}


def training_video_path(out_dir: Path) -> Path:
    return out_dir / "training.mp4"


def train_model(model: str) -> Path:
    """Always retrain from scratch — no caching.

    A cache keyed on file existence drifts when the repro script's constants
    (SEED, MAX_SAMPLES, EPOCHS, TIER) change without the artifacts dir being
    wiped: training is skipped but numbers.json.config is rewritten from the
    new constants, so the published figures no longer correspond to the
    published config. Always retraining (the `--wipe-dir` on the CLI below
    handles the per-model artifacts dir) keeps the artifacts honest.
    """
    out_dir = ARTIFACTS / model
    print(f"[{model}] training → {out_dir.relative_to(REPO)}")
    sh.uv(
        "run", "python", str(OSCILLOSCOPE), "train",
        "--model", model,
        "--kaiming-init",
        "--dataset", "mnist",
        "--max-samples", str(MAX_SAMPLES),
        "--epochs", str(EPOCHS),
        "--t-ms", str(T_MS),
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


def load_metrics(run_dir: Path) -> dict:
    return json.loads((run_dir / "metrics.json").read_text())


def load_config(run_dir: Path) -> dict:
    return json.loads((run_dir / "config.json").read_text())


def run_date(run_dir: Path) -> str:
    """Long-form date + HH:MM local time the training run produced its metrics.json.

    Prefers the canonical `run_finished_at` field (ISO-8601 UTC, written
    by oscilloscope.py at end of train). Falls back to file mtime for
    pre-field runs. The field survives git checkout / file copy, which
    mtime does not.
    """
    fmt = "%A, %B %-d %Y at %H:%M"
    metrics = load_metrics(run_dir)
    if "run_finished_at" in metrics:
        # ISO-8601 UTC → local long-form date + time
        dt_utc = datetime.fromisoformat(metrics["run_finished_at"])
        return dt_utc.astimezone().strftime(fmt)
    mtime = (run_dir / "metrics.json").stat().st_mtime
    return datetime.fromtimestamp(mtime).strftime(fmt)


def _stamp_figure(fig, notebook_run_id: str) -> None:
    """Stamp the notebook_run_id in the bottom-right corner of a matplotlib figure."""
    fig.text(0.995, 0.005, notebook_run_id, ha="right", va="bottom",
             fontsize=7, color="#888888", family="monospace")


def plot_firing_rates(run_dirs: dict[str, Path], out_path: Path,
                      notebook_run_id: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for model, run_dir in run_dirs.items():
        metrics = load_metrics(run_dir)
        epochs = [e["ep"] for e in metrics["epochs"]]
        rate_e = [e["rate_e"] for e in metrics["epochs"]]
        ax.plot(epochs, rate_e, marker="o",
                color=MODEL_COLORS[model], label=MODEL_LABELS[model])
    ax.set_xlabel("epoch")
    ax.set_ylabel("hidden-layer firing rate (Hz)")
    ax.set_title("Firing rate vs epoch")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    _stamp_figure(fig, notebook_run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_training_curves(run_dirs: dict[str, Path], out_path: Path,
                         notebook_run_id: str) -> None:
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(8, 4.5))
    for model, run_dir in run_dirs.items():
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


def _format_duration(seconds: float) -> str:
    s = int(round(seconds))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60:02d}s"
    return f"{s // 3600}h {(s % 3600) // 60:02d}m"


def write_numbers(run_dirs: dict[str, Path], out_path: Path,
                  notebook_run_id: str, duration_s: float) -> dict:
    # Pull shared hyperparameters from the first run's config.json (matched across models).
    first_cfg = load_config(next(iter(run_dirs.values())))
    summary: dict[str, dict] = {
        "notebook_run_id": notebook_run_id,
        "duration_s": round(duration_s, 1),
        "duration": _format_duration(duration_s),
        "config": {
            "tier": TIER,
            "dataset": "mnist",
            "max_samples": MAX_SAMPLES,
            "epochs": EPOCHS,
            "t_ms": first_cfg["t_ms"],
            "dt": first_cfg["dt"],
            "n_hidden": first_cfg["n_hidden"],
            "batch_size": first_cfg["batch_size"],
            "lr": first_cfg["lr"],
            "kaiming_init": True,
            "seed": SEED,
        },
        "runs": {},
    }
    for model, run_dir in run_dirs.items():
        metrics = load_metrics(run_dir)
        cfg = load_config(run_dir)
        summary["runs"][model] = {
            "label": MODEL_LABELS[model],
            "run_date": run_date(run_dir),
            "run_id": cfg.get("run_id"),
            "git_sha": cfg.get("git_sha"),
            "best_acc": metrics["best_acc"],
            "best_epoch": metrics["best_epoch"],
            "final_acc": metrics["epochs"][-1]["acc"],
            "final_loss": metrics["epochs"][-1]["loss"],
            "total_elapsed_s": metrics["total_elapsed_s"],
        }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def _render_stamp_png(notebook_run_id: str, stamp_path: Path) -> None:
    """Render a small transparent PNG with the notebook_run_id, for ffmpeg overlay."""
    import matplotlib as _mpl
    fig = plt.figure(figsize=(2.8, 0.28), dpi=150)
    fig.patch.set_alpha(0.0)
    fig.text(0.97, 0.5, notebook_run_id, ha="right", va="center",
             fontsize=10, color="white", family="monospace",
             bbox=dict(facecolor="black", alpha=0.55, pad=3, edgecolor="none"))
    stamp_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stamp_path, transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def copy_training_videos(run_dirs: dict[str, Path], out_dir: Path,
                         notebook_run_id: str) -> None:
    """Copy each run's training.mp4 (assembled from per-epoch observe frames)
    into the figures dir, overlaying the notebook_run_id in the bottom-right
    corner via ffmpeg. Each video shows the oscilloscope with accumulated
    accuracy / grad-flow / rate-history sidebars evolving over training.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp_path = out_dir / "_stamp.png"
    _render_stamp_png(notebook_run_id, stamp_path)
    for model, run_dir in run_dirs.items():
        src = training_video_path(run_dir)
        if not src.exists():
            raise SystemExit(f"missing training video: {src}")
        dst = out_dir / f"training_{model}.mp4"
        # Overlay stamp 10px from right and bottom edges.
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
    stamp_path.unlink(missing_ok=True)


def main() -> None:
    wipe_dir = "--no-wipe-dir" not in sys.argv
    t_start = time.monotonic()
    # Notebook-level run id stamps this repro invocation (spans all model runs).
    notebook_run_id = next_run_id(SLUG)
    print(f"notebook_run_id = {notebook_run_id}")
    if wipe_dir:
        for d in (ARTIFACTS, FIGURES):
            if d.exists():
                print(f"[wipe] {d.relative_to(REPO)}")
                shutil.rmtree(d)
    FIGURES.mkdir(parents=True, exist_ok=True)
    persist_run_id(SLUG, notebook_run_id)
    run_dirs = {m: train_model(m) for m in MODELS}
    fig_path = FIGURES / "training_curves.png"
    plot_training_curves(run_dirs, fig_path, notebook_run_id)
    print(f"wrote {fig_path.relative_to(REPO)}")
    rates_path = FIGURES / "firing_rates.png"
    plot_firing_rates(run_dirs, rates_path, notebook_run_id)
    print(f"wrote {rates_path.relative_to(REPO)}")
    copy_training_videos(run_dirs, FIGURES, notebook_run_id)
    numbers_path = FIGURES / "numbers.json"
    duration_s = time.monotonic() - t_start
    summary = write_numbers(run_dirs, numbers_path, notebook_run_id, duration_s)
    print(f"wrote {numbers_path.relative_to(REPO)}")
    for model, s in summary["runs"].items():
        print(f"  {model}: best={s['best_acc']}%  final={s['final_acc']}%  "
              f"elapsed={s['total_elapsed_s']:.0f}s")
    print(f"  total duration: {summary['duration']}")


if __name__ == "__main__":
    main()
    sys.exit(0)
