"""Notebook runner for entry 007 — standard-snn-exp single-model training.

Same as nb006 (standard-snn / snntorch-style direct-LIF) but with an
exponential synapse on the input layer. Companion to the other
per-model runners; sits between nb006 (standard-snn, no exp synapse)
and nb008 (snntorch-library reference) on the ladder.

Recipe is otherwise identical to nb006 — readout=li, surrogate-slope=1,
kaiming init, lr=0.01, batch=256 — so the only knob that differs is
the synaptic kernel.

Notebook entry: src/docs/src/pages/notebooks/nb007.mdx
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src" / "pinglab"))

from _per_model import run, osc_base_args  # noqa: E402

SLUG = "nb007"
MODEL = "standard-snn-exp"


def build_osc_args(tier: str, out_dir: Path) -> list[str]:
    # Recipe identical to nb006; only the model name changes. The
    # standard-snn-exp registry entry instantiates CUBANet with
    # exponential_synapse=True (see config.py MODEL_REGISTRY).
    return osc_base_args(out_dir, tier, build_as=MODEL) + [
        "--kaiming-init",
        "--readout", "li",
        "--surrogate-slope", "1",
        "--lr", "0.01",
        "--batch-size", "256",
    ]


if __name__ == "__main__":
    run(SLUG, MODEL, build_osc_args)
