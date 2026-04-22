"""Notebook runner for entry 004 — sMNIST.

Sequential MNIST: each MNIST digit is presented one row (28 pixels) at a
time over 280 ms (28 rows × 10 ms/row), so the network has to integrate
information across time to classify — a memory/dynamics test rather than
an input-encoding test.

This is the first-cell baseline: one *cuba* model, two hidden layers, at
extra_small tier, just to confirm the sMNIST pipeline + depth-≥2 config
trains end-to-end and reaches non-trivial accuracy. Once this works the
cell matrix (more models, depth sweeps, dt regimes) can be built on top.

Notebook entry: src/docs/src/pages/notebook/nb004.mdx
"""
from __future__ import annotations

import json
import math
import shutil
import sys
from datetime import datetime
from pathlib import Path

import sh

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src" / "pinglab"))

from _modal import append_modal_args, parse_modal_gpu  # noqa: E402
from _run_id import next_run_id, persist as persist_run_id  # noqa: E402

SLUG = "nb004"
ARTIFACTS = REPO / "src" / "artifacts" / "notebook" / SLUG
FIGURES = REPO / "src" / "docs" / "public" / "figures" / "notebook" / SLUG
OSCILLOSCOPE = REPO / "src" / "pinglab" / "oscilloscope.py"

# Tier config — see src/docs/src/pages/styleguide.md § 8 Run sizing tiers
TIER = "medium"
TIER_CONFIG = {
    "extra_small": dict(max_samples=200, epochs=3),
    "small":       dict(max_samples=500, epochs=5),
    "medium":      dict(max_samples=2000, epochs=10),
    "large":       dict(max_samples=5000, epochs=40),
}

# sMNIST presents each row (28 px) for 10 ms → 280 ms per sample.
T_MS = 280.0
DT = 1.0
SEED = 42
# ≥2 hidden layers required: single-hidden-layer SNNs bottleneck at the
# 28-input interface on sMNIST and fail to learn.
HIDDEN = [64, 64]
TAU_MEM_MS = 10.0  # matches models.py SNN_TAU_MEM_MS


def cuba_init_scales(dt: float, tau: float = TAU_MEM_MS) -> tuple[float, float]:
    """Per-step drive compensation for cuba — see nb003."""
    beta = math.exp(-dt / tau)
    return dt / (1.0 - beta), 1.0 / (1.0 - beta)


def train_baseline(modal_gpu: str | None = None) -> Path:
    tier = TIER_CONFIG[TIER]
    out_dir = ARTIFACTS / "cuba"
    sw, sb = cuba_init_scales(DT)
    print(f"[cuba sMNIST] training → {out_dir.relative_to(REPO)} "
          f"(init_scale W×{sw:.3f} b×{sb:.3f})"
          + (f"  [modal:{modal_gpu}]" if modal_gpu else ""))
    args = [
        "run", "python", str(OSCILLOSCOPE), "train",
        "--model", "cuba",
        "--dataset", "smnist",
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
        "--seed", str(SEED),
        "--out-dir", str(out_dir),
        "--wipe-dir",
    ]
    args = append_modal_args(args, modal_gpu)
    sh.uv(*args, _cwd=str(REPO), _out=sys.stdout, _err=sys.stderr)
    metrics_path = out_dir / "metrics.json"
    if not metrics_path.exists():
        raise SystemExit(f"training did not produce {metrics_path}")
    return out_dir


def main() -> None:
    wipe_dir = "--no-wipe-dir" not in sys.argv
    modal_gpu = parse_modal_gpu(sys.argv)
    run_id = next_run_id(SLUG)
    print(f"[nb004] run_id={run_id} tier={TIER}"
          + (f"  [modal:{modal_gpu}]" if modal_gpu else ""))
    if wipe_dir:
        for d in (ARTIFACTS, FIGURES):
            if d.exists():
                print(f"[wipe] {d.relative_to(REPO)}")
                shutil.rmtree(d)
    FIGURES.mkdir(parents=True, exist_ok=True)
    persist_run_id(SLUG, run_id)

    run_dir = train_baseline(modal_gpu=modal_gpu)
    metrics = json.loads((run_dir / "metrics.json").read_text())
    cfg = json.loads((run_dir / "config.json").read_text())
    best_acc = metrics.get("best_acc", max(e["acc"] for e in metrics["epochs"]))
    final_acc = metrics["epochs"][-1]["acc"]
    final_loss = metrics["epochs"][-1]["loss"]

    numbers = {
        "notebook_run_id": run_id,
        "git_sha": cfg.get("git_sha"),
        "tier": TIER,
        "config": {
            "model": "cuba",
            "dataset": "smnist",
            "hidden": HIDDEN,
            "t_ms": T_MS,
            "dt": DT,
            "seed": SEED,
            **TIER_CONFIG[TIER],
        },
        "cells": {
            "cuba_baseline": {
                "model": "cuba",
                "hidden": HIDDEN,
                "best_acc": best_acc,
                "final_acc": final_acc,
                "final_loss": final_loss,
                "run_dir": str(run_dir.relative_to(REPO)),
            },
        },
        "run_finished_at": datetime.utcnow().isoformat() + "Z",
    }
    (FIGURES / "numbers.json").write_text(json.dumps(numbers, indent=2))
    print(f"[nb004] best_acc={best_acc} final_acc={final_acc} "
          f"final_loss={final_loss:.4f}")
    print(f"[nb004] wrote numbers.json → {FIGURES.relative_to(REPO)}")


if __name__ == "__main__":
    main()
