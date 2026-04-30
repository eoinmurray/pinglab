"""Notebook runner for entry 008 — coba single-model training.

Trains the coba model (conductance-based synapses, dispatched via
COBANet with ei_strength=0 so the E→I→E loop is disabled) on MNIST at
dt=0.1 ms and publishes training curves, hidden firing rates, a
training video, and numbers.json. Companion to the other per-model
runners (nb006–nb009, nb011) and to the five-model Δt-stability sweep
in nb012.

Notebook entry: src/docs/src/pages/notebooks/nb010.mdx
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src" / "pinglab"))

from _per_model import run, osc_base_args  # noqa: E402

SLUG = "nb010"
MODEL = "coba"


def build_osc_args(tier: str, out_dir: Path) -> list[str]:
    # coba = ping with ei-strength=0; see models page and nb012 MODEL_CONFIG.
    # mem-mean readout + slope=1: snnTorch tutorial 5 pattern (see nb006).
    return osc_base_args(out_dir, tier, build_as="ping") + [
        "--ei-strength", "0",
        "--v-grad-dampen", "1000",
        "--w-in", "0.3",
        "--w-in-sparsity", "0.95",
        "--readout", "mem-mean",
        "--surrogate-slope", "1",
        # COBANet hidden firing rate is ~10× lower than CUBANet's
        # under matched recipes, so the output LIF gets ~10× less
        # drive. Scale W_out at init to compensate.
        "--readout-w-out-scale", "100",
        "--lr", "0.0004",
        "--batch-size", "256",
    ]


if __name__ == "__main__":
    run(SLUG, MODEL, build_osc_args, gpu_needs_a100=True)
