"""Notebook runner for entry 009 — ping single-model training.

Trains the full PING model (conductance-based synapses with the
E→I→E feedback loop that produces gamma oscillations) on MNIST at
dt=0.1 ms and publishes training curves, hidden firing rates, a
training video, and numbers.json. Companion to the other per-model
runners (nb005–nb008) and to the five-model Δt-stability sweep in
nb003.

Notebook entry: src/docs/src/pages/notebooks/nb009.mdx
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src" / "pinglab"))

from _per_model import run, osc_base_args  # noqa: E402

SLUG = "nb009"
MODEL = "ping"


def build_osc_args(tier: str, out_dir: Path) -> list[str]:
    return osc_base_args(out_dir, tier, build_as=MODEL) + [
        "--ei-strength", "0.5",
        "--v-grad-dampen", "1000",
        "--w-in", "1.2",
        "--w-in-sparsity", "0.95",
        "--lr", "0.0001",
    ]


if __name__ == "__main__":
    run(SLUG, MODEL, build_osc_args, gpu_needs_a100=True)
