"""Notebook runner for entry 005 — standard-snn single-model training.

Trains the standard-snn (LIF classifier, snntorch research baseline) at
dt=0.1 ms on MNIST and publishes training curves, hidden firing rates,
a training video, and numbers.json. Companion to the other per-model
runners (nb006–nb009) and to the five-model Δt-stability sweep in
nb010.

Notebook entry: src/docs/src/pages/notebooks/nb005.mdx
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src" / "pinglab"))

from _per_model import run, osc_base_args  # noqa: E402

SLUG = "nb005"
MODEL = "standard-snn"


def build_osc_args(tier: str, out_dir: Path) -> list[str]:
    # --readout li: leaky-integrator readout (bounded logits) instead of
    # the default "rate" readout, which accumulates spike counts over
    # T_steps=2000 and saturates CE so hard that backward produces
    # non-finite grads — every batch gets skipped, loss reports 0, and
    # the video shows identical frames across all epochs.
    # --surrogate-slope 1: default slope=5 × T_steps=2000 timesteps of
    # BPTT through the LIF stack overflows fp32 in W_ff.0 / b_ff.0 grads
    # (finite_max hits ~9e37, then NaN after further multiplies). slope=1
    # keeps the surrogate pseudo-derivative ≤ 1 so the gradient stays
    # bounded end-to-end; the grad-norm skip path no longer fires.
    return osc_base_args(out_dir, tier, build_as=MODEL) + [
        "--kaiming-init",
        "--readout", "li",
        "--surrogate-slope", "1",
        "--lr", "0.01",
        "--batch-size", "256",
    ]


if __name__ == "__main__":
    run(SLUG, MODEL, build_osc_args)
