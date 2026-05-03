"""Notebook runner for entry 007 — cuba single-model training.

Trains the cuba model (current-based synapses, exp-Euler + ZOH integration
keeping dt explicit in the update) on MNIST at dt=0.1 ms and publishes
training curves, hidden firing rates, a training video, and numbers.json.
Companion to the other per-model runners (nb007, nb009, nb011, nb012) and
to the five-model Δt-stability sweep in nb013.

Notebook entry: src/docs/src/pages/notebooks/nb010.mdx
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src" / "pinglab"))

from _per_model import run, osc_base_args  # noqa: E402

SLUG = "nb010"
MODEL = "cuba"


def build_osc_args(tier: str, out_dir: Path) -> list[str]:
    # mem-mean readout + slope=1 — same rationale as nb007.
    return osc_base_args(out_dir, tier, build_as=MODEL) + [
        "--kaiming-init",
        "--readout",
        "mem-mean",
        "--surrogate-slope",
        "1",
        "--lr",
        "0.04",
        "--batch-size",
        "256",
    ]


if __name__ == "__main__":
    run(SLUG, MODEL, build_osc_args)
