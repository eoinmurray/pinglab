"""Notebook runner for entry 006 — snntorch-library single-model training.

Trains the snntorch-library model (external snnTorch reference path) at
dt=0.1 ms on MNIST and publishes training curves, hidden firing rates,
a training video, and numbers.json. Companion to the other per-model
runners (nb005, nb007–nb009) and to the five-model Δt-stability sweep
in nb003.

Notebook entry: src/docs/src/pages/notebooks/nb006.mdx
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src" / "pinglab"))

from _per_model import run, osc_base_args  # noqa: E402

SLUG = "nb006"
MODEL = "snntorch-library"


def build_osc_args(tier: str, out_dir: Path) -> list[str]:
    return osc_base_args(out_dir, tier, build_as=MODEL) + [
        "--kaiming-init",
        "--lr", "0.01",
    ]


if __name__ == "__main__":
    run(SLUG, MODEL, build_osc_args)
