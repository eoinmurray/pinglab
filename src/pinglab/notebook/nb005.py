"""Notebook runner for entry 005 — SHD (Spiking Heidelberg Digits).

SHD is a natively event-based dataset: 20 classes of spoken digits
(English + German), 700 input channels (cochleogram spikes), ~1 s trials.
Unlike sMNIST (nb004), there is no artificial temporal encoding — each
trial *is* a spike train, which lines up directly with the dt and
encoding questions this notebook is built around.

This is the first-cell baseline: one *cuba* model, two hidden layers,
at *extra_small* tier, to validate the SHD pipeline + depth-2 ladder
train end-to-end and reach non-trivial accuracy. The cell matrix (more
models, depth sweeps, dt regimes) comes after.

Writes:
  * training_curves.png — train loss & test accuracy per epoch
  * training_cuba.mp4 — per-epoch snapshot video
  * numbers.json — config + baseline cell results

Notebook entry: src/docs/src/pages/notebook/nb005.mdx
"""
from __future__ import annotations

import json
import math
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import sh

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src" / "pinglab"))

import matplotlib.pyplot as plt  # noqa: E402

from _run_id import next_run_id, persist as persist_run_id  # noqa: E402

SLUG = "nb005"
ARTIFACTS = REPO / "src" / "artifacts" / "notebook" / SLUG
FIGURES = REPO / "src" / "docs" / "public" / "figures" / "notebook" / SLUG
OSCILLOSCOPE = REPO / "src" / "pinglab" / "oscilloscope.py"

# Tier config — see src/docs/src/pages/styleguide.md § 9 Run sizing tiers
TIER = "medium"
TIER_CONFIG = {
    "extra_small": dict(max_samples=200, epochs=3),
    "small":       dict(max_samples=500, epochs=5),
    "medium":      dict(max_samples=2000, epochs=10),
    "large":       dict(max_samples=5000, epochs=40),
}

# SHD trials are ~1 s; at dt=1 ms that's 1000 steps.
T_MS = 1000.0
DT = 1.0
SEED = 42
# ≥2 hidden layers — matches nb004 sMNIST rationale for depth on
# classification tasks with many input channels.
HIDDEN = [128, 128]
TAU_MEM_MS = 10.0  # matches models.py SNN_TAU_MEM_MS

MODEL = "cuba"
MODEL_COLOR = "#2ca02c"


def cuba_init_scales(dt: float, tau: float = TAU_MEM_MS) -> tuple[float, float]:
    """Per-step drive compensation for cuba — see nb003."""
    beta = math.exp(-dt / tau)
    return dt / (1.0 - beta), 1.0 / (1.0 - beta)


def train_baseline() -> Path:
    tier = TIER_CONFIG[TIER]
    out_dir = ARTIFACTS / MODEL
    sw, sb = cuba_init_scales(DT)
    print(f"[cuba SHD] training → {out_dir.relative_to(REPO)} "
          f"(init_scale W×{sw:.3f} b×{sb:.3f})")
    sh.uv(
        "run", "python", str(OSCILLOSCOPE), "train",
        "--model", MODEL,
        "--dataset", "shd",
        "--n-hidden", *[str(h) for h in HIDDEN],
        "--max-samples", str(tier["max_samples"]),
        "--epochs", str(tier["epochs"]),
        "--t-ms", str(T_MS),
        "--dt", str(DT),
        "--lr", "0.01",
        "--kaiming-init",
        "--init-scale-weight", "3.0",
        "--init-scale-bias", f"{sb}",
        "--no-dales-law",
        "--readout", "li",
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
    video_path = out_dir / "training.mp4"
    if not video_path.exists():
        raise SystemExit(f"training did not produce {video_path}")
    return out_dir


def _stamp_figure(fig, run_id: str) -> None:
    fig.text(0.995, 0.005, run_id, ha="right", va="bottom",
             fontsize=7, color="#888888", family="monospace")


def _render_stamp_png(run_id: str, stamp_path: Path) -> None:
    fig = plt.figure(figsize=(2.8, 0.28), dpi=150)
    fig.patch.set_alpha(0.0)
    fig.text(0.97, 0.5, run_id, ha="right", va="center",
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


def plot_training_curves(metrics: dict, out_path: Path, run_id: str) -> None:
    epochs = [e["ep"] for e in metrics["epochs"]]
    loss = [e["loss"] for e in metrics["epochs"]]
    acc = [e["acc"] for e in metrics["epochs"]]
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(8.0, 4.5))
    ax_loss.plot(epochs, loss, marker="o", color=MODEL_COLOR, label=MODEL)
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("train loss")
    ax_loss.set_title("train loss")
    ax_loss.grid(alpha=0.3)
    ax_loss.legend(frameon=False, fontsize=8)
    ax_acc.plot(epochs, acc, marker="o", color=MODEL_COLOR, label=MODEL)
    ax_acc.axhline(5.0, color="#cc4444", linestyle="--", linewidth=1,
                   label="chance (5%)")
    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylabel("test accuracy (%)")
    ax_acc.set_ylim(0, max(15.0, max(acc) * 1.2))
    ax_acc.set_title("test accuracy")
    ax_acc.grid(alpha=0.3)
    ax_acc.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    _stamp_figure(fig, run_id)
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


def main() -> None:
    wipe_dir = "--no-wipe-dir" not in sys.argv
    t_start = time.monotonic()
    run_id = next_run_id(SLUG)
    print(f"[nb005] run_id={run_id} tier={TIER}")
    if wipe_dir:
        for d in (ARTIFACTS, FIGURES):
            if d.exists():
                print(f"[wipe] {d.relative_to(REPO)}")
                shutil.rmtree(d)
    FIGURES.mkdir(parents=True, exist_ok=True)
    persist_run_id(SLUG, run_id)

    run_dir = train_baseline()
    metrics = json.loads((run_dir / "metrics.json").read_text())
    cfg = json.loads((run_dir / "config.json").read_text())

    plot_training_curves(metrics, FIGURES / "training_curves.png", run_id)
    print(f"wrote {(FIGURES / 'training_curves.png').relative_to(REPO)}")

    stamp_path = FIGURES / "_stamp.png"
    _render_stamp_png(run_id, stamp_path)
    _copy_with_stamp(run_dir / "training.mp4",
                     FIGURES / f"training_{MODEL}.mp4", stamp_path)
    stamp_path.unlink(missing_ok=True)

    best_acc = metrics.get("best_acc", max(e["acc"] for e in metrics["epochs"]))
    final_acc = metrics["epochs"][-1]["acc"]
    final_loss = metrics["epochs"][-1]["loss"]
    duration_s = time.monotonic() - t_start

    numbers = {
        "notebook_run_id": run_id,
        "git_sha": cfg.get("git_sha"),
        "tier": TIER,
        "duration_s": round(duration_s, 1),
        "duration": _format_duration(duration_s),
        "config": {
            "model": MODEL,
            "dataset": "shd",
            "hidden": HIDDEN,
            "t_ms": T_MS,
            "dt": DT,
            "seed": SEED,
            **TIER_CONFIG[TIER],
        },
        "cells": {
            "cuba_baseline": {
                "model": MODEL,
                "hidden": HIDDEN,
                "best_acc": best_acc,
                "final_acc": final_acc,
                "final_loss": final_loss,
                "run_dir": str(run_dir.relative_to(REPO)),
            },
        },
        "run_finished_at": datetime.utcnow().isoformat() + "Z",
    }
    (FIGURES / "numbers.json").write_text(json.dumps(numbers, indent=2) + "\n")
    print(f"[nb005] best_acc={best_acc} final_acc={final_acc} "
          f"final_loss={final_loss:.4f}")
    print(f"[nb005] wrote numbers.json → {FIGURES.relative_to(REPO)}")
    print(f"[nb005] duration: {_format_duration(duration_s)}")


if __name__ == "__main__":
    main()
