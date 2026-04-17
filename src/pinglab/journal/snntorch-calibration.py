"""Repro for the snnTorch calibration journal entry.

Runs a small, matched training of the in-repo `snntorch` model and the
`snntorch-library` parity reference, then writes a training-curve comparison
figure and a numbers.json summary into the journal's figures dir.

Journal entry: src/docs/src/pages/journal/snntorch-calibration.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import sh

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src" / "pinglab"))

import matplotlib.pyplot as plt  # noqa: E402

SLUG = "snntorch-calibration"
ARTIFACTS = REPO / "src" / "artifacts" / "journal" / SLUG
FIGURES = REPO / "src" / "docs" / "public" / "figures" / "journal" / SLUG
OSCILLOSCOPE = REPO / "src" / "pinglab" / "oscilloscope.py"

MODELS = ["snntorch", "snntorch-library"]
MAX_SAMPLES = 200
EPOCHS = 3

MODEL_LABELS = {
    "snntorch": "pinglab snntorch",
    "snntorch-library": "snnTorch library",
}
MODEL_COLORS = {
    "snntorch": "#1f77b4",
    "snntorch-library": "#d62728",
}


def training_video_path(out_dir: Path) -> Path:
    return out_dir / "training.mp4"


def train_model(model: str) -> Path:
    out_dir = ARTIFACTS / model
    metrics_path = out_dir / "metrics.json"
    if metrics_path.exists() and training_video_path(out_dir).exists():
        print(f"[{model}] metrics + training video exist — reusing")
        return out_dir
    print(f"[{model}] training → {out_dir.relative_to(REPO)}")
    sh.uv(
        "run", "python", str(OSCILLOSCOPE), "train",
        "--model", model,
        "--kaiming-init",
        "--dataset", "mnist",
        "--max-samples", str(MAX_SAMPLES),
        "--epochs", str(EPOCHS),
        "--observe", "video",
        "--frame-rate", "1",
        "--out-dir", str(out_dir),
        "--wipe-dir",
        _cwd=str(REPO),
        _out=sys.stdout,
        _err=sys.stderr,
    )
    if not metrics_path.exists():
        raise SystemExit(f"training did not produce {metrics_path}")
    if not training_video_path(out_dir).exists():
        raise SystemExit(f"training did not produce {training_video_path(out_dir)}")
    return out_dir


def load_metrics(run_dir: Path) -> dict:
    return json.loads((run_dir / "metrics.json").read_text())


def plot_training_curves(run_dirs: dict[str, Path], out_path: Path) -> None:
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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_numbers(run_dirs: dict[str, Path], out_path: Path) -> dict:
    summary: dict[str, dict] = {
        "config": {
            "dataset": "mnist",
            "max_samples": MAX_SAMPLES,
            "epochs": EPOCHS,
            "kaiming_init": True,
        },
        "runs": {},
    }
    for model, run_dir in run_dirs.items():
        metrics = load_metrics(run_dir)
        summary["runs"][model] = {
            "best_acc": metrics["best_acc"],
            "best_epoch": metrics["best_epoch"],
            "final_acc": metrics["epochs"][-1]["acc"],
            "final_loss": metrics["epochs"][-1]["loss"],
            "total_elapsed_s": metrics["total_elapsed_s"],
        }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def copy_training_videos(run_dirs: dict[str, Path], out_dir: Path) -> None:
    """Copy each run's training.mp4 (assembled from per-epoch observe frames)
    into the figures dir. Each video shows the oscilloscope with accumulated
    accuracy / grad-flow / rate-history sidebars evolving over training.
    """
    import shutil
    out_dir.mkdir(parents=True, exist_ok=True)
    for model, run_dir in run_dirs.items():
        src = training_video_path(run_dir)
        if not src.exists():
            raise SystemExit(f"missing training video: {src}")
        dst = out_dir / f"training_{model}.mp4"
        shutil.copy2(src, dst)
        print(f"wrote {dst.relative_to(REPO)}")


def main() -> None:
    run_dirs = {m: train_model(m) for m in MODELS}
    fig_path = FIGURES / "training_curves.png"
    plot_training_curves(run_dirs, fig_path)
    print(f"wrote {fig_path.relative_to(REPO)}")
    copy_training_videos(run_dirs, FIGURES)
    numbers_path = FIGURES / "numbers.json"
    summary = write_numbers(run_dirs, numbers_path)
    print(f"wrote {numbers_path.relative_to(REPO)}")
    for model, s in summary["runs"].items():
        print(f"  {model}: best={s['best_acc']}%  final={s['final_acc']}%  "
              f"elapsed={s['total_elapsed_s']:.0f}s")


if __name__ == "__main__":
    main()
    sys.exit(0)
