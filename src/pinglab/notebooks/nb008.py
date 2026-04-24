"""Notebook runner for entry 008 — coba single-model training.

Trains the coba model (conductance-based synapses, dispatched via
PINGNet with ei_strength=0 so the E→I→E loop is disabled) on MNIST at
dt=0.1 ms and publishes training curves, hidden firing rates, a
training video, and numbers.json. Companion to the other per-model
runners (nb005–nb007, nb009) and to the five-model Δt-stability sweep
in nb010.

Notebook entry: src/docs/src/pages/notebooks/nb008.mdx
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src" / "pinglab"))

from _per_model import run, osc_base_args  # noqa: E402

SLUG = "nb008"
MODEL = "coba"


def build_osc_args(tier: str, out_dir: Path) -> list[str]:
    # coba = ping with ei-strength=0; see models page and nb010 MODEL_CONFIG.
    return osc_base_args(out_dir, tier, build_as="ping") + [
        "--ei-strength", "0",
        "--v-grad-dampen", "1000",
        "--w-in", "0.3",
        "--w-in-sparsity", "0.95",
        "--lr", "0.0001",
    ]


if __name__ == "__main__":
    run(SLUG, MODEL, build_osc_args, gpu_needs_a100=True)
