"""Notebook runner for entry 022 — coba with and without Dale's law.

Trains the coba recipe (COBANet with --ei-strength 0, no active I-loop)
twice: once with Dale's law enforced on the trainable feedforward
weights (W_ff ≥ 0, the default — the cell is excitatory in, only) and
once with signed weights allowed (W_ff free, the standard mathematical
SNN). The recurrent E-I weights inside COBANet are fixed buffers either
way, so this isolates the effect of Dale's law on the *input*
projection.

Two cells: coba/dales, coba/no-dales. Same calibrated recipe in every
other respect (lr, w_in, surrogate slope, readout, batch size) as
nb011, so any accuracy or firing-rate difference is attributable to
the Dale's-law constraint alone.

Notebook entry: src/docs/src/pages/notebooks/nb022.mdx
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src" / "pinglab"))

from _modal import BatchDispatcher, parse_modal_gpu  # noqa: E402
from _run_id import next_run_id, persist as persist_run_id  # noqa: E402
from _tier import parse_tier  # noqa: E402
from pinglab import theme  # noqa: E402

SLUG = "nb022"
ARTIFACTS = REPO / "src" / "artifacts" / "notebooks" / SLUG
FIGURES = REPO / "src" / "docs" / "public" / "figures" / "notebooks" / SLUG
OSCILLOSCOPE = REPO / "src" / "pinglab" / "oscilloscope/__main__.py"

TIER_CONFIG = {
    "extra small": dict(max_samples=100, epochs=1),
    "small": dict(max_samples=500, epochs=5),
    "medium": dict(max_samples=2000, epochs=10),
    "large": dict(max_samples=5000, epochs=40),
    "extra large": dict(max_samples=10000, epochs=40),
}
DEFAULT_TIER = "small"
T_MS = 200.0
DT_TRAIN = 0.1
SEED = 42

# Two cells: one with Dale's law on, one off. Same recipe otherwise.
CELLS = ["dales", "no_dales"]

# coba recipe from nb011 / nb021.
COBA_RECIPE: dict[str, str | bool | None] = {
    "__build_as": "ping",
    "--ei-strength": "0",
    "--v-grad-dampen": "1000",
    "--w-in": "0.3",
    "--w-in-sparsity": "0.95",
    "--readout": "mem-mean",
    "--surrogate-slope": "1",
    "--readout-w-out-scale": "100",
    "--lr": "0.0004",
    "--batch-size": "256",
}

CELL_COLORS = {
    "dales": theme.AMBER,
    "no_dales": theme.ELECTRIC_CYAN,
}
CELL_LABEL = {
    "dales": "Dale's law (W_ff ≥ 0)",
    "no_dales": "signed weights",
}

MIN_ACC_BY_TIER = {
    "extra small": 15.0,
    "small": 30.0,
    "medium": 50.0,
    "large": 70.0,
    "extra large": 70.0,
}


def cell_dir(cell: str) -> Path:
    return ARTIFACTS / f"coba__{cell}"


def build_train_args(cell: str, tier: str, out_dir: Path) -> list[str]:
    args = [
        "train",
        "--model", COBA_RECIPE["__build_as"],
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
    for k, v in COBA_RECIPE.items():
        if k.startswith("__"):
            continue
        if v is True:
            args.append(k)
        elif v is not None:
            args += [k, v]
    # Dale's-law flag. The CLI default is --dales-law (on); we pass
    # --no-dales-law explicitly for the signed-weights cell.
    if cell == "no_dales":
        args.append("--no-dales-law")
    return args


def load_metrics(run_dir: Path) -> dict:
    return json.loads((run_dir / "metrics.json").read_text())


def load_config(run_dir: Path) -> dict:
    return json.loads((run_dir / "config.json").read_text())


def _stamp(fig, run_id: str) -> None:
    fig.text(
        0.995, 0.005, run_id,
        ha="right", va="bottom",
        fontsize=theme.SIZE_CAPTION,
        color=theme.LABEL, family="monospace",
    )


def plot_accuracy(rows: list[dict], out_path: Path, run_id: str) -> None:
    """Side-by-side bars: final accuracy with vs without Dale's law."""
    theme.apply()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    xs = [0, 1]
    accs = [next(r for r in rows if r["cell"] == c)["final_acc"] for c in CELLS]
    rates = [next(r for r in rows if r["cell"] == c)["rate_e"] for c in CELLS]
    ax.bar(
        xs,
        accs,
        width=0.6,
        color=[CELL_COLORS[c] for c in CELLS],
        edgecolor=theme.INK_BLACK,
    )
    for x, a, r in zip(xs, accs, rates):
        ax.text(x, a + 1.5, f"{a:.1f}%", ha="center", va="bottom",
                fontsize=theme.SIZE_BASE, color=theme.INK_BLACK)
        ax.text(x, 2, f"{r:.1f} Hz", ha="center", va="bottom",
                fontsize=theme.SIZE_ANNOTATION, color=theme.LABEL)
    ax.set_xticks(xs)
    ax.set_xticklabels([CELL_LABEL[c] for c in CELLS])
    ax.set_ylabel("test accuracy (%, final epoch)")
    ax.set_title("coba with and without Dale's law")
    ax.set_ylim(0, max(100, max(accs) + 10))
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    _stamp(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_weight_hist(rows: list[dict], out_path: Path, run_id: str) -> None:
    """Histogram of trained input weights W_ff[0] for each cell.

    Under Dale's law every entry is clamped non-negative at use; without
    it the optimiser is free to put weight mass on either sign. This
    plot shows the resulting distribution.
    """
    theme.apply()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(-0.6, 0.6, 61)
    for c in CELLS:
        weights_path = cell_dir(c) / "weights.pth"
        state = torch.load(weights_path, map_location="cpu", weights_only=False)
        # The first feedforward weight matrix is W_ff.0 (input → hidden).
        w0 = state.get("W_ff.0")
        if w0 is None:
            # Fallback: scan for the first feedforward weight.
            for k in state:
                if k.startswith("W_ff"):
                    w0 = state[k]
                    break
        flat = w0.detach().cpu().numpy().flatten()
        ax.hist(
            flat, bins=bins,
            color=CELL_COLORS[c], alpha=0.55,
            edgecolor=theme.INK_BLACK, linewidth=0.5,
            label=CELL_LABEL[c],
        )
    ax.axvline(0.0, color=theme.LABEL, lw=0.8, ls=":", alpha=0.7)
    ax.set_xlabel("W_ff[0] entry value")
    ax.set_ylabel("count")
    ax.set_title("Distribution of trained input weights")
    ax.legend(fontsize=theme.SIZE_ANNOTATION)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    _stamp(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def evaluate_success(rows: list[dict], tier: str, figures: Path) -> list[dict]:
    floor = float(MIN_ACC_BY_TIER[tier])
    figs_root = figures.parents[2]

    def artifact(name: str, label: str) -> dict:
        path = figures / name
        ok = path.exists() and path.stat().st_size > 0
        href = "/" + str(path.relative_to(figs_root)) if ok else None
        return {
            "label": label,
            "passed": bool(ok),
            "detail": f"{path.name} ({path.stat().st_size} bytes)"
            if ok else f"missing {path.name}",
            "detail_href": href,
        }

    crits: list[dict] = [
        artifact("accuracy.png", "accuracy figure rendered"),
        artifact("weight_hist.png", "weight histogram rendered"),
    ]
    for c in CELLS:
        crits.append(
            artifact(
                f"training__{c}.mp4",
                f"{c}: training video",
            )
        )
    for c in CELLS:
        r = next(row for row in rows if row["cell"] == c)
        crits.append(
            {
                "label": f"{c} acc ≥ {floor:.0f}% ({tier} floor)",
                "passed": bool(r["best_acc"] >= floor),
                "detail": f"{c}={r['best_acc']:.2f}%",
            }
        )
    return crits


def _format_duration(seconds: float) -> str:
    s = int(round(seconds))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60:02d}s"
    return f"{s // 3600}h {(s % 3600) // 60:02d}m"


def copy_video(run_dir: Path, out_path: Path) -> None:
    src = run_dir / "training.mp4"
    if not src.exists():
        raise SystemExit(f"missing training video: {src}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, out_path)
    print(f"wrote {out_path}")


def main() -> None:
    tier = parse_tier(sys.argv, choices=TIER_CONFIG.keys(), default=DEFAULT_TIER)
    modal_gpu = parse_modal_gpu(sys.argv)
    skip_training = "--skip-training" in sys.argv
    wipe_dir = "--no-wipe-dir" not in sys.argv

    t_start = time.monotonic()
    notebook_run_id = next_run_id(SLUG)
    print(
        f"notebook_run_id = {notebook_run_id} tier={tier} "
        f"cells={len(CELLS)}"
        + ("  [skip-training]" if skip_training else "")
    )

    if wipe_dir:
        if skip_training:
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

    if not skip_training:
        dispatcher = BatchDispatcher(modal_gpu, REPO, OSCILLOSCOPE)
        for c in CELLS:
            out = cell_dir(c)
            gpu_override = None
            # The recipe builds as "ping" under the hood (ei-strength=0
            # makes it coba). Modal sometimes needs A100 for the ping
            # build at dt=0.1; auto-upgrade T4/L4/A10G.
            if modal_gpu in ("T4", "L4", "A10G"):
                gpu_override = "A100"
            print(
                f"[train] coba/{c} → {out.relative_to(REPO)}"
                + (f"  [modal:{modal_gpu}]" if modal_gpu else "")
            )
            dispatcher.submit(
                build_train_args(c, tier, out),
                out,
                gpu_override=gpu_override,
            )
        dispatcher.drain()

    rows: list[dict] = []
    for c in CELLS:
        run_dir = cell_dir(c)
        if not (run_dir / "metrics.json").exists():
            raise SystemExit(f"missing metrics: {run_dir / 'metrics.json'}")
        metrics = load_metrics(run_dir)
        last = metrics["epochs"][-1]
        rows.append(
            {
                "cell": c,
                "best_acc": float(metrics["best_acc"]),
                "best_epoch": int(metrics["best_epoch"]),
                "final_acc": float(last["acc"]),
                "rate_e": float(last.get("rate_e") or 0.0),
            }
        )
        copy_video(run_dir, FIGURES / f"training__{c}.mp4")

    print("  results:")
    for r in rows:
        print(
            f"    coba/{r['cell']:<9}  "
            f"acc(final)={r['final_acc']:6.2f}%  best={r['best_acc']:6.2f}%  "
            f"rate_e={r['rate_e']:6.1f} Hz"
        )

    plot_accuracy(rows, FIGURES / "accuracy.png", notebook_run_id)
    print(f"wrote {FIGURES / 'accuracy.png'}")
    plot_weight_hist(rows, FIGURES / "weight_hist.png", notebook_run_id)
    print(f"wrote {FIGURES / 'weight_hist.png'}")

    duration_s = time.monotonic() - t_start
    train_cfg = load_config(cell_dir(CELLS[0]))
    crits = evaluate_success(rows, tier, FIGURES)
    summary = {
        "notebook_run_id": notebook_run_id,
        "git_sha": train_cfg.get("git_sha"),
        "duration_s": round(duration_s, 1),
        "duration": _format_duration(duration_s),
        "tier": tier,
        "config": {
            "tier": tier,
            "dataset": "mnist",
            "cells": CELLS,
            "max_samples": TIER_CONFIG[tier]["max_samples"],
            "epochs": TIER_CONFIG[tier]["epochs"],
            "t_ms": T_MS,
            "dt": DT_TRAIN,
            "seed": SEED,
        },
        "results": rows,
        "success_criteria": crits,
    }
    (FIGURES / "numbers.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {FIGURES / 'numbers.json'}")
    print(f"  total duration: {summary['duration']}")

    for c in crits:
        mark = "pass" if c["passed"] else "FAIL"
        print(f"  [{mark}] {c['label']} — {c['detail']}")
    if any(not c["passed"] for c in crits):
        sys.exit(1)


if __name__ == "__main__":
    main()
