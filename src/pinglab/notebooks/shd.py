"""Notebook runner for entry 004 — SHD (Spiking Heidelberg Digits).

SHD is a natively event-based dataset: 20 classes of spoken digits
(English + German), 700 input channels (cochleogram spikes), ~1 s trials.
There is no artificial temporal encoding — each trial *is* a spike
train, which lines up directly with the dt and encoding questions this
notebook is built around.

Runs a single baseline cell (cuba + rate readout) at the configured tier.

Writes:
  * shd_digits.png — 4×5 dataset-at-a-glance raster grid
  * training_curves.png — loss + accuracy per epoch
  * training_cuba.mp4 — per-epoch hidden-activity video
  * numbers.json — config + cell summary

Notebook entry: src/docs/src/pages/notebooks/shd.mdx
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
import numpy as np  # noqa: E402

from _modal import append_modal_args, parse_modal_gpu  # noqa: E402
from _tier import parse_tier  # noqa: E402
from _run_id import next_run_id, persist as persist_run_id  # noqa: E402

SLUG = "shd"
ARTIFACTS = REPO / "src" / "artifacts" / "notebooks" / SLUG
FIGURES = REPO / "src" / "docs" / "public" / "figures" / "notebooks" / SLUG
OSCILLOSCOPE = REPO / "src" / "pinglab" / "oscilloscope.py"

DEFAULT_TIER = "medium"
TIER = DEFAULT_TIER  # overridable via --tier <name>
TIER_CONFIG = {
    "extra_small": dict(max_samples=200, epochs=3),
    "small":       dict(max_samples=500, epochs=5),
    "medium":      dict(max_samples=2000, epochs=10),
    "large":       dict(max_samples=5000, epochs=40),
    "huge":        dict(max_samples=10000, epochs=80),
}

T_MS = 1000.0
DT = 2.0
SEED = 42
HIDDEN = [128]
TAU_MEM_MS = 20.0
TAU_SYN_MS = 10.0

MODEL = "cuba-exp"
MODEL_COLOR = "#2ca02c"

def cuba_init_scales(dt: float, tau: float = TAU_MEM_MS) -> tuple[float, float]:
    """Per-step drive compensation for cuba — see dt-stability."""
    beta = math.exp(-dt / tau)
    return dt / (1.0 - beta), 1.0 / (1.0 - beta)


def train_cell(
    name: str,
    lr: float,
    isw: float,
    isb: float,
    hidden: list[int],
    readout: str,
    observe_video: bool,
    modal_gpu: str | None = None,
) -> Path:
    """Train one cell. Returns the run dir containing metrics.json."""
    tier = TIER_CONFIG[TIER]
    out_dir = ARTIFACTS / name
    print(f"[cell {name}] lr={lr} isw={isw} isb={isb:.3f} hidden={hidden} "
          f"readout={readout} → {out_dir.relative_to(REPO)}"
          + (f"  [modal:{modal_gpu}]" if modal_gpu else ""))
    args = [
        "run", "python", str(OSCILLOSCOPE), "train",
        "--model", MODEL,
        "--dataset", "shd",
        "--n-hidden", *[str(h) for h in hidden],
        "--max-samples", str(tier["max_samples"]),
        "--epochs", str(tier["epochs"]),
        "--t-ms", str(T_MS),
        "--dt", str(DT),
        "--tau-mem", str(TAU_MEM_MS),
        "--tau-syn", str(TAU_SYN_MS),
        "--lr", str(lr),
        # Adamax (Cramer 2022, Zenke Spytorch). L∞-norm second moment
        # makes the preconditioner robust to single outlier batches —
        # directly addresses the "one bad grad poisons Adam" failure
        # mode that produced the huge-tier one-way-door at ep 38.
        "--optimizer", "adamax",
        # ReduceLROnPlateau (factor 0.5, patience 5). Large tier plateaued
        # around ep 20 then diverged at ep 28 with a constant 1e-3 lr; the
        # scheduler should halve the step once acc stops climbing and keep
        # the network in the stable region.
        "--adaptive-lr",
        "--batch-size", "256",
        "--kaiming-init",
        "--init-scale-weight", str(isw),
        "--init-scale-bias", f"{isb}",
        "--no-dales-law",
        # Feedforward only while we test slope=10 in isolation. Re-enable
        # W_rec once slope=10 alone is stable on adamax.
        "--w-rec", "0.0", "0.0",
        "--readout", readout,
        # Cramer et al. (2022) SHD RSNN recipe — two-sided firing-rate
        # regulariser on per-neuron trial spike counts.
        # grad-clip: loosen from M.GRAD_CLIP=1.0 default. At τ_syn=10 ms with
        # 500-step BPTT the raw grad_norm lands 50k–1.5M; global-norm=1.0
        # projects every update onto the unit ball, destroying Adam's
        # preconditioner. Setting clip to 100 lets Adam's second-moment
        # estimate absorb the scale.
        "--grad-clip", "100",
        "--fr-reg-lower-theta", "0.01",
        "--fr-reg-lower-strength", "1.0",
        "--fr-reg-upper-theta", "100",
        "--fr-reg-upper-strength", "0.06",
        "--seed", str(SEED),
        "--out-dir", str(out_dir),
        "--wipe-dir",
    ]
    if observe_video:
        args += ["--observe", "video", "--frame-rate", "1"]
    args = append_modal_args(args, modal_gpu)
    sh.uv(*args, _cwd=str(REPO), _out=sys.stdout, _err=sys.stderr)
    metrics_path = out_dir / "metrics.json"
    if not metrics_path.exists():
        raise SystemExit(f"training did not produce {metrics_path}")
    if observe_video and not (out_dir / "training.mp4").exists():
        raise SystemExit(f"training did not produce {out_dir / 'training.mp4'}")
    return out_dir


def _cell_summary(name: str, spec: dict, run_dir: Path) -> dict:
    metrics = json.loads((run_dir / "metrics.json").read_text())
    best_acc = metrics.get("best_acc", max(e["acc"] for e in metrics["epochs"]))
    final = metrics["epochs"][-1]
    return {
        "name": name,
        "model": MODEL,
        **spec,
        "best_acc": best_acc,
        "final_acc": final["acc"],
        "final_loss": final["loss"],
        "run_dir": str(run_dir.relative_to(REPO)),
    }


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


SHD_CLASS_NAMES = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "null", "eins", "zwei", "drei", "vier", "fünf", "sechs", "sieben", "acht", "neun",
]


def plot_shd_digits(out_path: Path, run_id: str, seed: int = SEED) -> None:
    """4×5 grid of one spike raster per SHD class (0..19).

    Drops into the entry as a dataset-at-a-glance panel: English digits
    (0–9) and their German counterparts (10–19) lined up so the
    cross-language structure is visible at a glance.
    """
    from oscilloscope import _load_shd  # noqa: E402

    X, y = _load_shd(dt_ms=DT, t_ms=T_MS, max_samples=200)
    rng = np.random.default_rng(seed)
    picks: list[int] = []
    for cls in range(20):
        idxs = np.where(y == cls)[0]
        if len(idxs) == 0:
            continue
        picks.append(int(rng.choice(idxs)))

    # Compute tight data bounds so empty margins don't dominate.
    x_max = 0.0
    y_min_val = 700.0
    y_max_val = 0.0
    for idx in picks:
        t_sp, ch_sp = np.nonzero(X[idx])
        if len(t_sp) == 0:
            continue
        x_max = max(x_max, float(t_sp.max()) * DT)
        y_min_val = min(y_min_val, float(ch_sp.min()))
        y_max_val = max(y_max_val, float(ch_sp.max()))
    x_hi = float(np.ceil(x_max / 50.0) * 50.0) if x_max > 0 else T_MS
    y_lo = max(0.0, float(np.floor(y_min_val / 50.0) * 50.0))
    y_hi = min(700.0, float(np.ceil(y_max_val / 50.0) * 50.0))

    fig, axes = plt.subplots(4, 5, figsize=(8.0, 4.5))
    ink = "#1a1a1a"
    rule = "#bdbdbd"
    for row_idx, row in enumerate(axes):
        for col_idx, ax in enumerate(row):
            flat_idx = row_idx * 5 + col_idx
            if flat_idx >= len(picks):
                ax.set_visible(False)
                continue
            idx = picks[flat_idx]
            cls = int(y[idx])
            raster = X[idx]
            t_spikes, ch_spikes = np.nonzero(raster)
            ax.scatter(t_spikes * DT, ch_spikes, s=0.4, color=ink, marker=".",
                       linewidths=0, rasterized=True, alpha=0.85)
            ax.set_title(f"{SHD_CLASS_NAMES[cls]}  ·  {cls:02d}",
                         fontsize=8, color=ink, loc="left", pad=3,
                         fontfamily="serif")
            ax.set_xlim(0.0, x_hi)
            ax.set_ylim(y_lo, y_hi)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor(rule)
                spine.set_linewidth(0.5)

    # Single axis legend — scale bar on the bottom-left panel only.
    anchor = axes[-1, 0]
    anchor.set_xticks([0, int(x_hi)])
    anchor.set_yticks([int(y_lo), int(y_hi)])
    anchor.tick_params(axis="both", labelsize=6, color=rule, length=2,
                       width=0.5, pad=2)
    anchor.set_xlabel("time (ms)", fontsize=6, color=ink, labelpad=1)
    anchor.set_ylabel("channel", fontsize=6, color=ink, labelpad=2)

    fig.suptitle("SHD · one example per class  (0–9 English, 10–19 German)",
                 fontsize=9, color=ink, y=0.985, fontfamily="serif")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95), pad=0.2)
    fig.subplots_adjust(hspace=0.35, wspace=0.06)
    _stamp_figure(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


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


def run_baseline(run_id: str, modal_gpu: str | None, skip_training: bool) -> dict:
    """Single cell at the current best-known config (post-sweep winner)."""
    # Raw Kaiming init — no per-step drive compensation. The previous
    # isw=6 / isb derived from 1/(1-β_mem) pushed neurons into a regime
    # where β=10 produced 1e10+ grads even with adamax; dropping it
    # lets the model learn under Cramer's native weight scale.
    spec = dict(lr=1e-3, isw=1.0, isb=1.0, hidden=HIDDEN, readout="li")
    if skip_training:
        run_dir = ARTIFACTS / "cuba_baseline"
        if not (run_dir / "metrics.json").exists():
            raise SystemExit(f"--skip-training requires existing {run_dir}/metrics.json")
    else:
        run_dir = train_cell("cuba_baseline", **spec,
                             observe_video=True, modal_gpu=modal_gpu)
    summary = _cell_summary("cuba_baseline", spec, run_dir)

    metrics = json.loads((run_dir / "metrics.json").read_text())
    plot_training_curves(metrics, FIGURES / "training_curves.png", run_id)
    print(f"wrote {(FIGURES / 'training_curves.png').relative_to(REPO)}")

    stamp_path = FIGURES / "_stamp.png"
    _render_stamp_png(run_id, stamp_path)
    _copy_with_stamp(run_dir / "training.mp4",
                     FIGURES / f"training_{MODEL}.mp4", stamp_path)
    stamp_path.unlink(missing_ok=True)
    return summary


def main() -> None:
    global TIER
    wipe_dir = "--no-wipe-dir" not in sys.argv
    skip_training = "--skip-training" in sys.argv
    modal_gpu = parse_modal_gpu(sys.argv)
    TIER = parse_tier(sys.argv, choices=TIER_CONFIG.keys(), default=DEFAULT_TIER)

    t_start = time.monotonic()
    run_id = next_run_id(SLUG)
    print(f"[shd] run_id={run_id} tier={TIER}"
          + ("  [skip-training]" if skip_training else "")
          + (f"  [modal:{modal_gpu}]" if modal_gpu else ""))
    if wipe_dir:
        wipe_targets = (FIGURES,) if skip_training else (ARTIFACTS, FIGURES)
        for d in wipe_targets:
            if d.exists():
                print(f"[wipe] {d.relative_to(REPO)}")
                shutil.rmtree(d)
    FIGURES.mkdir(parents=True, exist_ok=True)
    persist_run_id(SLUG, run_id)

    # Dataset-at-a-glance panel — first thing we render.
    plot_shd_digits(FIGURES / "shd_digits.png", run_id)
    print(f"wrote {(FIGURES / 'shd_digits.png').relative_to(REPO)}")

    summary = run_baseline(run_id, modal_gpu, skip_training)
    cells_dict = {"cuba_baseline": summary}
    winner = summary

    run_dir = REPO / summary["run_dir"]
    cfg = json.loads((run_dir / "config.json").read_text())

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
        "cells": cells_dict,
        "winner": winner,
        "run_finished_at": datetime.utcnow().isoformat() + "Z",
    }
    (FIGURES / "numbers.json").write_text(json.dumps(numbers, indent=2) + "\n")
    print(f"[shd] best_acc={winner['best_acc']} "
          f"final_acc={winner['final_acc']} "
          f"final_loss={winner['final_loss']:.4f}")
    print(f"[shd] wrote numbers.json → {FIGURES.relative_to(REPO)}")
    print(f"[shd] duration: {_format_duration(duration_s)}")


if __name__ == "__main__":
    main()
