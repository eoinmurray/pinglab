"""Notebook runner for entry 007 — cuba single-model training.

Trains the cuba model (current-based synapses, snnTorch-style LIF with
one-shot (1-β)/dt init-scale compensation at train-dt) on MNIST at
dt=0.1 ms and publishes training curves, hidden firing rates, a
training video, and numbers.json. Companion to the other per-model
runners (nb005, nb006, nb008, nb009) and to the five-model Δt-stability
sweep in nb010.

Notebook entry: src/docs/src/pages/notebooks/nb007.mdx
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src" / "pinglab"))

from _per_model import run, osc_base_args, DT_TRAIN  # noqa: E402

SLUG = "nb007"
MODEL = "cuba"
TAU_MEM_MS = 10.0
# (1-β)/dt compensation at train-dt: one-shot init multiplier so cuba's
# initial per-step drive matches standard-snn's at this dt. See models
# page "CUBA" / nb010 init_scales_for.
_BETA = math.exp(-DT_TRAIN / TAU_MEM_MS)
INIT_SCALE_WEIGHT = DT_TRAIN / (1.0 - _BETA)
INIT_SCALE_BIAS = 1.0 / (1.0 - _BETA)


def build_osc_args(tier: str, out_dir: Path) -> list[str]:
    # See nb005 for why --readout li + --surrogate-slope 1 are required —
    # cuba shares the CUBANet class and therefore the saturated-rate-readout
    # + slope-5×2000-step BPTT overflow pathologies with standard-snn.
    return osc_base_args(out_dir, tier, build_as=MODEL) + [
        "--kaiming-init",
        "--readout", "li",
        "--surrogate-slope", "1",
        "--lr", "0.01",
        "--init-scale-weight", f"{INIT_SCALE_WEIGHT}",
        "--init-scale-bias", f"{INIT_SCALE_BIAS}",
    ]


if __name__ == "__main__":
    run(SLUG, MODEL, build_osc_args)
