"""Notebook runner for entry 005 — SHD (Spiking Heidelberg Digits).

SHD is a natively event-based dataset: 20 classes of spoken digits
(English + German), 700 input channels (cochleogram spikes), ~1 s trials.
Unlike sMNIST (nb004), there is no artificial temporal encoding — each
trial *is* a spike train, which lines up directly with the dt and
encoding questions this notebook is built around.

Two invocation modes:

  * default — single baseline cell (cuba + li readout), as before
  * --sweep — grid over (lr, hidden architecture) at fixed
    init_scale_weight, cuba + rate readout. Winner picked by best_acc
    and promoted as numbers["winner"].

Writes:
  * training_curves.png — winner cell (or baseline in default mode)
  * training_cuba.mp4 — winner cell (or baseline)
  * sweep_lr_hidden.png — heatmap (sweep mode only)
  * numbers.json — config + cells dict + winner

Notebook entry: src/docs/src/pages/notebook/nb005.mdx
"""
from __future__ import annotations

import json
import math
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import sh

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src" / "pinglab"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from _modal import append_modal_args, parse_modal_gpu  # noqa: E402
from _run_id import next_run_id, persist as persist_run_id  # noqa: E402

SLUG = "nb005"
ARTIFACTS = REPO / "src" / "artifacts" / "notebook" / SLUG
FIGURES = REPO / "src" / "docs" / "public" / "figures" / "notebook" / SLUG
OSCILLOSCOPE = REPO / "src" / "pinglab" / "oscilloscope.py"

DEFAULT_TIER = "medium"
TIER = DEFAULT_TIER  # overridable via --tier <name>
TIER_CONFIG = {
    "extra_small": dict(max_samples=200, epochs=3),
    "small":       dict(max_samples=500, epochs=5),
    "medium":      dict(max_samples=2000, epochs=10),
    "large":       dict(max_samples=5000, epochs=40),
}

T_MS = 1000.0
DT = 1.0
SEED = 42
HIDDEN = [128, 128]
TAU_MEM_MS = 10.0

MODEL = "cuba"
MODEL_COLOR = "#2ca02c"

# Sweep axes — grid over (lr, hidden architecture), cuba + rate readout.
# isw fixed at 1.0 (step 1 winner); isb = 1/(1-β) from nb004.
SWEEP_LRS = [3e-2, 1e-1]
SWEEP_HIDDENS = [[128, 128], [256, 256], [512, 512], [256, 256, 256]]
SWEEP_ISW = 1.0


def cuba_init_scales(dt: float, tau: float = TAU_MEM_MS) -> tuple[float, float]:
    """Per-step drive compensation for cuba — see nb003."""
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
        "--lr", str(lr),
        "--kaiming-init",
        "--init-scale-weight", str(isw),
        "--init-scale-bias", f"{isb}",
        "--no-dales-law",
        "--readout", readout,
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


def _hidden_label(hidden: list[int]) -> str:
    """'128×2', '256×3' when layers are uniform, else '128-256-128'."""
    if len(set(hidden)) == 1:
        return f"{hidden[0]}×{len(hidden)}"
    return "-".join(str(h) for h in hidden)


def plot_lr_hidden_heatmap(cells: list[dict], out_path: Path, run_id: str) -> None:
    """Best-acc heatmap over (lr, hidden architecture)."""
    hidden_keys = [tuple(h) for h in SWEEP_HIDDENS]
    acc = np.full((len(hidden_keys), len(SWEEP_LRS)), np.nan)
    for c in cells:
        try:
            i = hidden_keys.index(tuple(c["hidden"]))
            j = SWEEP_LRS.index(c["lr"])
        except ValueError:
            continue
        acc[i, j] = c["best_acc"]

    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    im = ax.imshow(acc, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(SWEEP_LRS)))
    ax.set_xticklabels([f"{lr:g}" for lr in SWEEP_LRS])
    ax.set_yticks(range(len(hidden_keys)))
    ax.set_yticklabels([_hidden_label(list(h)) for h in hidden_keys])
    ax.set_xlabel("learning rate")
    ax.set_ylabel("hidden architecture")
    ax.set_title(f"cuba + rate on SHD — best test accuracy (%), {TIER} tier")
    for i in range(acc.shape[0]):
        for j in range(acc.shape[1]):
            v = acc[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                        color="white" if v < np.nanmean(acc) else "black",
                        fontsize=10)
    fig.colorbar(im, ax=ax, label="best acc (%)")
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


def _parse_tier(argv: list[str], default: str = DEFAULT_TIER) -> str:
    if "--tier" not in argv:
        return default
    idx = argv.index("--tier")
    if idx + 1 >= len(argv):
        raise SystemExit("--tier requires a value")
    tier = argv[idx + 1]
    if tier not in TIER_CONFIG:
        raise SystemExit(f"--tier: unknown tier {tier!r}, choose from {list(TIER_CONFIG)}")
    return tier


def _parse_max_workers(argv: list[str], default: int = 4) -> int:
    if "--max-workers" not in argv:
        return default
    idx = argv.index("--max-workers")
    if idx + 1 >= len(argv):
        raise SystemExit("--max-workers requires a value")
    return int(argv[idx + 1])


def run_baseline(run_id: str, modal_gpu: str | None) -> dict:
    """Single cell at the current best-known config (post-sweep winner)."""
    _, isb = cuba_init_scales(DT)
    spec = dict(lr=0.03, isw=1.0, isb=isb, hidden=HIDDEN, readout="rate")
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


def run_sweep(run_id: str, modal_gpu: str | None, max_workers: int) -> tuple[list[dict], dict]:
    """Grid over (lr, hidden architecture); cuba + rate readout. Returns (cells, winner)."""
    _, isb = cuba_init_scales(DT)
    specs = []
    for hidden in SWEEP_HIDDENS:
        for lr in SWEEP_LRS:
            hlabel = _hidden_label(hidden).replace("×", "x")
            name = f"sweep_lr{lr:g}_h{hlabel}".replace(".", "p")
            specs.append((name, dict(lr=lr, isw=SWEEP_ISW, isb=isb,
                                     hidden=list(hidden), readout="rate")))

    def _run(item):
        name, spec = item
        run_dir = train_cell(name, **spec, observe_video=False,
                             modal_gpu=modal_gpu)
        return _cell_summary(name, spec, run_dir)

    cells: list[dict] = []
    if modal_gpu and max_workers > 1:
        print(f"[sweep] fan-out {len(specs)} cells on {max_workers} workers "
              f"(modal:{modal_gpu})")
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for summary in ex.map(_run, specs):
                cells.append(summary)
                print(f"[sweep] {summary['name']}: best_acc={summary['best_acc']}")
    else:
        print(f"[sweep] sequential {len(specs)} cells")
        for item in specs:
            summary = _run(item)
            cells.append(summary)
            print(f"[sweep] {summary['name']}: best_acc={summary['best_acc']}")

    winner = max(cells, key=lambda c: c["best_acc"])
    print(f"[sweep] winner: {winner['name']} best_acc={winner['best_acc']}")

    # Heatmap + training curves from winner.
    plot_lr_hidden_heatmap(cells, FIGURES / "sweep_lr_hidden.png", run_id)
    print(f"wrote {(FIGURES / 'sweep_lr_hidden.png').relative_to(REPO)}")

    winner_dir = REPO / winner["run_dir"]
    metrics = json.loads((winner_dir / "metrics.json").read_text())
    plot_training_curves(metrics, FIGURES / "training_curves.png", run_id)
    print(f"wrote {(FIGURES / 'training_curves.png').relative_to(REPO)}")

    # Winner doesn't have a video (sweep cells skip --observe video) —
    # rerun the winner once with video on so the entry's Figure in
    # Appendix › Videos stays populated.
    winner_video_spec = {k: winner[k] for k in ("lr", "isw", "isb", "hidden", "readout")}
    winner_dir = train_cell(winner["name"] + "_video", **winner_video_spec,
                            observe_video=True, modal_gpu=modal_gpu)
    stamp_path = FIGURES / "_stamp.png"
    _render_stamp_png(run_id, stamp_path)
    _copy_with_stamp(winner_dir / "training.mp4",
                     FIGURES / f"training_{MODEL}.mp4", stamp_path)
    stamp_path.unlink(missing_ok=True)

    return cells, winner


def main() -> None:
    global TIER
    wipe_dir = "--no-wipe-dir" not in sys.argv
    sweep = "--sweep" in sys.argv
    modal_gpu = parse_modal_gpu(sys.argv)
    TIER = _parse_tier(sys.argv)
    max_workers = _parse_max_workers(sys.argv, default=len(SWEEP_LRS) * len(SWEEP_HIDDENS) if sweep else 1)

    t_start = time.monotonic()
    run_id = next_run_id(SLUG)
    mode = "sweep" if sweep else "baseline"
    print(f"[nb005] run_id={run_id} tier={TIER} mode={mode}"
          + (f"  [modal:{modal_gpu}]" if modal_gpu else ""))
    if wipe_dir:
        for d in (ARTIFACTS, FIGURES):
            if d.exists():
                print(f"[wipe] {d.relative_to(REPO)}")
                shutil.rmtree(d)
    FIGURES.mkdir(parents=True, exist_ok=True)
    persist_run_id(SLUG, run_id)

    cells_dict: dict = {}
    winner: dict | None = None
    if sweep:
        cells, winner = run_sweep(run_id, modal_gpu, max_workers)
        for c in cells:
            cells_dict[c["name"]] = c
    else:
        summary = run_baseline(run_id, modal_gpu)
        cells_dict["cuba_baseline"] = summary
        winner = summary

    # Pull git_sha from any cell's config.json (all commits same SHA).
    any_run_dir = REPO / next(iter(cells_dict.values()))["run_dir"]
    cfg = json.loads((any_run_dir / "config.json").read_text())

    duration_s = time.monotonic() - t_start
    numbers = {
        "notebook_run_id": run_id,
        "git_sha": cfg.get("git_sha"),
        "mode": mode,
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
            "sweep_lrs": SWEEP_LRS if sweep else None,
            "sweep_hiddens": SWEEP_HIDDENS if sweep else None,
            "sweep_isw": SWEEP_ISW if sweep else None,
        },
        "cells": cells_dict,
        "winner": winner,
        "run_finished_at": datetime.utcnow().isoformat() + "Z",
    }
    (FIGURES / "numbers.json").write_text(json.dumps(numbers, indent=2) + "\n")
    print(f"[nb005] winner best_acc={winner['best_acc']} "
          f"final_acc={winner['final_acc']} "
          f"final_loss={winner['final_loss']:.4f}")
    print(f"[nb005] wrote numbers.json → {FIGURES.relative_to(REPO)}")
    print(f"[nb005] duration: {_format_duration(duration_s)}")


if __name__ == "__main__":
    main()
