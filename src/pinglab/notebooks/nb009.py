"""Notebook runner for entry 006 — snntorch-library single-model training.

Trains the snntorch-library model (external snnTorch reference path) at
dt=0.1 ms on MNIST and publishes training curves, hidden firing rates,
a training video, and numbers.json. Companion to the other per-model
runners (nb007, nb010–nb012) and to the five-model Δt-stability sweep
in nb013.

Notebook entry: src/docs/src/pages/notebooks/nb009.mdx
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src" / "pinglab"))

from _per_model import run, osc_base_args  # noqa: E402

SLUG = "nb009"
MODEL = "snntorch-library"


def build_osc_args(tier: str, out_dir: Path) -> list[str]:
    # --readout mem-mean: snnTorch tutorial 5 pattern (output spiking LIF
    # with mean(v_out) loss). Dense per-step gradient signal vs the
    # surrogate-gated spike-count alternative.
    # --surrogate-slope 1: bounds fast-sigmoid pseudo-derivative through
    # 2000 BPTT steps; slope=5 overflows fp32 in W_ff.0 grads.
    return osc_base_args(out_dir, tier, build_as=MODEL) + [
        "--kaiming-init",
        "--readout",
        "mem-mean",
        "--surrogate-slope",
        "1",
        "--lr",
        "0.01",
        "--batch-size",
        "256",
    ]


if __name__ == "__main__":
    run(SLUG, MODEL, build_osc_args)
