"""Notebook runner for entry 004 — exp-Euler vs fwd-Euler parity + CM_BACK_SCALE retune.

Trains *ping* at matched config under both COBA integrators and across a
range of gradient dampening factors. Answers two coupled questions:

  1. **Parity.** Does exp-Euler reach the same final accuracy as the
     previous forward-Euler default at matched cm_back_scale?
  2. **Retune.** Does the unified-integrator change warrant a different
     cm_back_scale default?

Experiment matrix (default tier=extra_small):
  integrator × cm_back_scale ∈ {expeuler, fwd} × {200, 500, 1000, 2000}

Writes:
  * training_curves.png   — loss & accuracy per epoch, one line per cell
  * accuracy_bar.png      — best test accuracy per cell
  * numbers.json          — config + per-cell best/final + gradient stats

Notebook entry: src/docs/src/pages/notebook/nb004.mdx
"""
from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import sh

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src" / "pinglab"))

import matplotlib.pyplot as plt  # noqa: E402

from _run_id import next_run_id, persist as persist_run_id  # noqa: E402

SLUG = "nb004"
ARTIFACTS = REPO / "src" / "artifacts" / "notebook" / SLUG
FIGURES = REPO / "src" / "docs" / "public" / "figures" / "notebook" / SLUG
OSCILLOSCOPE = REPO / "src" / "pinglab" / "oscilloscope.py"

# Tier config — see src/docs/src/pages/llm-conventions.md § 8 Run sizing tiers
TIER = "extra_small"
TIER_CONFIG = {
    "extra_small": dict(max_samples=200, epochs=3, t_ms=600.0),
    "small":       dict(max_samples=500, epochs=5, t_ms=600.0),
    "medium":      dict(max_samples=2000, epochs=10, t_ms=600.0),
    "large":       dict(max_samples=5000, epochs=40, t_ms=200.0),
}

INTEGRATORS = ["expeuler", "fwd"]
CM_BACK_SCALES = [200.0, 500.0, 1000.0, 2000.0]
DT = 0.25
SEED = 42

INTEGRATOR_COLORS = {"expeuler": "#1f77b4", "fwd": "#ff7f0e"}


def _cell_key(integrator: str, cm_back: float) -> str:
    return f"{integrator}_cm{cm_back:g}"


def train_cell(integrator: str, cm_back: float) -> Path:
    """Train PING at one (integrator, cm_back) cell. Returns run dir."""
    tier = TIER_CONFIG[TIER]
    out_dir = ARTIFACTS / _cell_key(integrator, cm_back)
    print(f"[{integrator} cm={cm_back}] training → {out_dir.relative_to(REPO)}")
    sh.uv(
        "run", "python", str(OSCILLOSCOPE), "train",
        "--model", "ping",
        "--ei-strength", "0.5",
        "--coba-integrator", integrator,
        "--cm-back-scale", str(cm_back),
        "--dataset", "mnist",
        "--max-samples", str(tier["max_samples"]),
        "--epochs", str(tier["epochs"]),
        "--t-ms", str(tier["t_ms"]),
        "--dt", str(DT),
        "--lr", "0.0001",
        "--adaptive-lr",
        "--w-in", "1.2",
        "--w-in-sparsity", "0.95",
        "--seed", str(SEED),
        "--out-dir", str(out_dir),
        "--wipe-dir",
        _cwd=str(REPO),
        _out=sys.stdout,
        _err=sys.stderr,
    )
    if not (out_dir / "metrics.json").exists():
        raise SystemExit(f"training did not produce {out_dir}/metrics.json")
    return out_dir


def load_metrics(run_dir: Path) -> dict:
    return json.loads((run_dir / "metrics.json").read_text())


def plot_training_curves(cells: dict[tuple[str, float], Path],
                         out_path: Path, run_id: str) -> None:
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(8, 4.5))
    for (integrator, cm_back), run_dir in cells.items():
        m = load_metrics(run_dir)
        epochs = [e["ep"] for e in m["epochs"]]
        loss = [e["loss"] for e in m["epochs"]]
        acc = [e["acc"] for e in m["epochs"]]
        label = f"{integrator} cm={cm_back:g}"
        color = INTEGRATOR_COLORS[integrator]
        alpha = 0.4 + 0.6 * (CM_BACK_SCALES.index(cm_back) / max(1, len(CM_BACK_SCALES) - 1))
        ax_loss.plot(epochs, loss, marker="o", color=color, alpha=alpha, label=label)
        ax_acc.plot(epochs, acc, marker="o", color=color, alpha=alpha, label=label)
    for ax, ylabel, title in [
        (ax_loss, "train loss", "Training loss"),
        (ax_acc, "test accuracy (%)", "Test accuracy"),
    ]:
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    fig.text(0.995, 0.005, run_id, ha="right", va="bottom",
             fontsize=7, color="#888888", family="monospace")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_accuracy_bar(cells: dict[tuple[str, float], Path],
                      out_path: Path, run_id: str) -> None:
    """Bar chart: best test accuracy per (integrator, cm_back) cell."""
    import numpy as np
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.35
    x = np.arange(len(CM_BACK_SCALES))
    for i, integrator in enumerate(INTEGRATORS):
        accs = []
        for cm in CM_BACK_SCALES:
            m = load_metrics(cells[(integrator, cm)])
            accs.append(m.get("best_acc", max(e["acc"] for e in m["epochs"])))
        ax.bar(x + (i - 0.5) * width, accs, width,
               color=INTEGRATOR_COLORS[integrator], label=integrator)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{cm:g}" for cm in CM_BACK_SCALES])
    ax.set_xlabel("cm_back_scale")
    ax.set_ylabel("best test accuracy (%)")
    ax.set_title("PING parity + cm_back_scale sweep")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.text(0.995, 0.005, run_id, ha="right", va="bottom",
             fontsize=7, color="#888888", family="monospace")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def collect_numbers(cells: dict[tuple[str, float], Path], run_id: str) -> dict:
    out: dict = {
        "notebook_run_id": run_id,
        "tier": TIER,
        "config": {
            "integrators": INTEGRATORS,
            "cm_back_scales": CM_BACK_SCALES,
            "dt": DT,
            "seed": SEED,
            **TIER_CONFIG[TIER],
        },
        "cells": {},
    }
    for (integrator, cm), run_dir in cells.items():
        m = load_metrics(run_dir)
        key = _cell_key(integrator, cm)
        out["cells"][key] = {
            "integrator": integrator,
            "cm_back_scale": cm,
            "best_acc": m.get("best_acc", max(e["acc"] for e in m["epochs"])),
            "final_acc": m["epochs"][-1]["acc"],
            "final_loss": m["epochs"][-1]["loss"],
            "run_dir": str(run_dir.relative_to(REPO)),
        }
    return out


def wipe_default(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-wipe-dir", action="store_true",
                    help="Skip wiping ARTIFACTS + FIGURES before run.")
    args = ap.parse_args()

    if not args.no_wipe_dir:
        wipe_default(ARTIFACTS)
        wipe_default(FIGURES)

    run_id = next_run_id(SLUG)
    persist_run_id(SLUG, run_id)
    print(f"[nb004] run_id={run_id} tier={TIER}")

    cells: dict[tuple[str, float], Path] = {}
    for integrator in INTEGRATORS:
        for cm in CM_BACK_SCALES:
            cells[(integrator, cm)] = train_cell(integrator, cm)

    plot_training_curves(cells, FIGURES / "training_curves.png", run_id)
    plot_accuracy_bar(cells, FIGURES / "accuracy_bar.png", run_id)
    numbers = collect_numbers(cells, run_id)
    numbers["run_finished_at"] = datetime.utcnow().isoformat() + "Z"
    (FIGURES / "numbers.json").write_text(json.dumps(numbers, indent=2))
    print(f"[nb004] wrote figures + numbers.json → {FIGURES.relative_to(REPO)}")


if __name__ == "__main__":
    main()
