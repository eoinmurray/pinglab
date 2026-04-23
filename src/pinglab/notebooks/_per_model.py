"""Shared orchestration for single-model training notebooks (nb005–nb009).

Each per-model runner declares its SLUG, MODEL, and a build_osc_args(tier)
callable that returns the oscilloscope CLI argument list. This module
handles the common plumbing: tier parsing, dir wipe, dispatcher, plots,
video copy, numbers.json — identical across all five.
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import sh

from _modal import BatchDispatcher, parse_modal_gpu
from _run_id import next_run_id, persist as persist_run_id
from _tier import parse_tier
from pinglab import theme

TIER_CONFIG = {
    "extra small": dict(max_samples=100,   epochs=1),
    "small":       dict(max_samples=500,   epochs=5),
    "medium":      dict(max_samples=2000,  epochs=10),
    "large":       dict(max_samples=5000,  epochs=40),
    "extra large": dict(max_samples=10000, epochs=40),
}
DEFAULT_TIER = "small"
T_MS = 200.0
DT_TRAIN = 0.1
SEED = 42


def load_metrics(run_dir: Path) -> dict:
    return json.loads((run_dir / "metrics.json").read_text())


def load_config(run_dir: Path) -> dict:
    return json.loads((run_dir / "config.json").read_text())


def run_date(run_dir: Path) -> str:
    fmt = "%A, %B %-d %Y at %H:%M"
    metrics = load_metrics(run_dir)
    if "run_finished_at" in metrics:
        return datetime.fromisoformat(metrics["run_finished_at"]).astimezone().strftime(fmt)
    return datetime.fromtimestamp((run_dir / "metrics.json").stat().st_mtime).strftime(fmt)


def _stamp_figure(fig, notebook_run_id: str) -> None:
    fig.text(0.995, 0.005, notebook_run_id, ha="right", va="bottom",
             fontsize=7, color=theme.LABEL, family="monospace")


def _format_duration(seconds: float) -> str:
    s = int(round(seconds))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60:02d}s"
    return f"{s // 3600}h {(s % 3600) // 60:02d}m"


def plot_training_curves(run_dir: Path, out_path: Path, model: str,
                         notebook_run_id: str) -> None:
    metrics = load_metrics(run_dir)
    epochs = [e["ep"] for e in metrics["epochs"]]
    loss = [e["loss"] for e in metrics["epochs"]]
    acc = [e["acc"] for e in metrics["epochs"]]
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 4.5))
    ax_loss.plot(epochs, loss, marker="o", color=theme.CAT_BLUE, label=model)
    ax_acc.plot(epochs, acc, marker="o", color=theme.CAT_BLUE, label=model)
    ax_loss.set_xlabel("epoch"); ax_loss.set_ylabel("train loss")
    ax_loss.set_title(f"{model} — train loss"); ax_loss.grid(alpha=0.3)
    ax_loss.legend(frameon=False, fontsize=8)
    ax_acc.set_xlabel("epoch"); ax_acc.set_ylabel("test accuracy (%)")
    ax_acc.set_title(f"{model} — test accuracy"); ax_acc.grid(alpha=0.3)
    ax_acc.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    _stamp_figure(fig, notebook_run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_firing_rates(run_dir: Path, out_path: Path, model: str,
                      notebook_run_id: str) -> None:
    metrics = load_metrics(run_dir)
    init_rate = metrics.get("init", {}).get("rate_e") or 0.0
    epochs = [0] + [e["ep"] for e in metrics["epochs"]]
    rates = [init_rate] + [e.get("rate_e", 0.0) for e in metrics["epochs"]]
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    ax.plot(epochs, rates, marker="o", color=theme.CAT_GREEN, label=model)
    ax.set_xlabel("epoch"); ax.set_ylabel("mean hidden firing rate (Hz)")
    ax.set_title(f"{model} — hidden firing rate per epoch"); ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    _stamp_figure(fig, notebook_run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _render_stamp_png(notebook_run_id: str, stamp_path: Path) -> None:
    fig = plt.figure(figsize=(2.8, 0.28), dpi=150)
    fig.patch.set_alpha(0.0)
    fig.text(0.97, 0.5, notebook_run_id, ha="right", va="center",
             fontsize=10, color="white", family="monospace",
             bbox=dict(facecolor="black", alpha=0.55, pad=3, edgecolor="none"))
    stamp_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stamp_path, transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def copy_training_video(run_dir: Path, out_dir: Path,
                        notebook_run_id: str) -> None:
    src = run_dir / "training.mp4"
    if not src.exists():
        raise SystemExit(f"missing training video: {src}")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp_path = out_dir / "_stamp.png"
    _render_stamp_png(notebook_run_id, stamp_path)
    dst = out_dir / "training.mp4"
    sh.ffmpeg(
        "-y", "-i", str(src), "-i", str(stamp_path),
        "-filter_complex", "[0:v][1:v]overlay=W-w-10:H-h-10",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "veryfast", "-crf", "20",
        "-movflags", "+faststart",
        str(dst),
        _out=sys.stdout, _err=sys.stderr,
    )
    print(f"wrote {dst}")
    stamp_path.unlink(missing_ok=True)


def write_numbers(run_dir: Path, out_path: Path, model: str, tier: str,
                  notebook_run_id: str, duration_s: float) -> dict:
    metrics = load_metrics(run_dir)
    cfg = load_config(run_dir)
    summary = {
        "notebook_run_id": notebook_run_id,
        "git_sha": cfg.get("git_sha"),
        "duration_s": round(duration_s, 1),
        "duration": _format_duration(duration_s),
        "tier": tier,
        "config": {
            "tier": tier,
            "dataset": "mnist",
            "model": model,
            "max_samples": TIER_CONFIG[tier]["max_samples"],
            "epochs": TIER_CONFIG[tier]["epochs"],
            "t_ms": cfg["t_ms"],
            "dt": DT_TRAIN,
            "n_hidden": cfg["n_hidden"],
            "batch_size": cfg["batch_size"],
            "seed": SEED,
        },
        "run": {
            "model": model,
            "run_date": run_date(run_dir),
            "run_id": cfg.get("run_id"),
            "git_sha": cfg.get("git_sha"),
            "lr": cfg.get("lr"),
            "best_acc": metrics["best_acc"],
            "best_epoch": metrics["best_epoch"],
            "final_acc": metrics["epochs"][-1]["acc"],
            "final_loss": metrics["epochs"][-1]["loss"],
            "total_elapsed_s": metrics["total_elapsed_s"],
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def run(slug: str, model: str, build_osc_args: Callable[[str, Path], list[str]],
        gpu_needs_a100: bool = False) -> None:
    """Run one per-model notebook. build_osc_args(tier, out_dir) returns the
    full `oscilloscope train …` argument list for the given tier."""
    repo = Path(__file__).resolve().parents[3]
    artifacts = repo / "src" / "artifacts" / "notebooks" / slug
    figures = repo / "src" / "docs" / "public" / "figures" / "notebooks" / slug
    oscilloscope = repo / "src" / "pinglab" / "oscilloscope.py"

    skip_training = "--skip-training" in sys.argv
    wipe_dir = "--no-wipe-dir" not in sys.argv
    modal_gpu = parse_modal_gpu(sys.argv)
    tier = parse_tier(sys.argv, choices=TIER_CONFIG.keys(), default=DEFAULT_TIER)
    t_start = time.monotonic()
    notebook_run_id = next_run_id(slug)
    print(f"notebook_run_id = {notebook_run_id} tier={tier} model={model}"
          + ("  [skip-training]" if skip_training else ""))

    if wipe_dir:
        if skip_training:
            if figures.exists():
                print(f"[wipe] {figures.relative_to(repo)}")
                shutil.rmtree(figures)
        else:
            for d in (artifacts, figures):
                if d.exists():
                    print(f"[wipe] {d.relative_to(repo)}")
                    shutil.rmtree(d)
    figures.mkdir(parents=True, exist_ok=True)
    persist_run_id(slug, notebook_run_id)

    dispatcher = BatchDispatcher(modal_gpu, repo, oscilloscope)

    run_dir = artifacts / "train"
    if skip_training:
        if not (run_dir / "weights.pth").exists():
            raise SystemExit(f"--skip-training requires existing weights at {run_dir}")
    else:
        osc_args = build_osc_args(tier, run_dir)
        gpu_override = None
        if gpu_needs_a100 and modal_gpu in ("T4", "L4", "A10G"):
            gpu_override = "A100"
            print(f"  [modal] upgrading {model} from {modal_gpu} to A100 (memory)")
        dispatcher.submit(osc_args, run_dir, gpu_override=gpu_override)
    dispatcher.drain()

    if not (run_dir / "metrics.json").exists():
        raise SystemExit(f"training did not produce {run_dir / 'metrics.json'}")
    if not (run_dir / "training.mp4").exists():
        raise SystemExit(f"training did not produce {run_dir / 'training.mp4'}")

    plot_training_curves(run_dir, figures / "training_curves.png", model, notebook_run_id)
    print(f"wrote {figures / 'training_curves.png'}")
    plot_firing_rates(run_dir, figures / "firing_rates.png", model, notebook_run_id)
    print(f"wrote {figures / 'firing_rates.png'}")
    copy_training_video(run_dir, figures, notebook_run_id)

    duration_s = time.monotonic() - t_start
    summary = write_numbers(run_dir, figures / "numbers.json", model, tier,
                            notebook_run_id, duration_s)
    print(f"wrote {figures / 'numbers.json'}")
    s = summary["run"]
    print(f"  {model}: best={s['best_acc']}%  final={s['final_acc']}%  "
          f"elapsed={s['total_elapsed_s']:.0f}s")
    print(f"  total duration: {summary['duration']}")


def osc_base_args(out_dir: Path, tier: str, build_as: str) -> list[str]:
    """Common prefix of every per-model train invocation."""
    return [
        "train",
        "--model", build_as,
        "--dataset", "mnist",
        "--max-samples", str(TIER_CONFIG[tier]["max_samples"]),
        "--epochs", str(TIER_CONFIG[tier]["epochs"]),
        "--t-ms", str(T_MS),
        "--dt", str(DT_TRAIN),
        "--seed", str(SEED),
        "--observe", "video",
        "--frame-rate", "1",
        "--out-dir", str(out_dir),
        "--wipe-dir",
    ]
