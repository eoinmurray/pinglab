"""Independent exp086 stages. Importing never starts a run."""

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
# exp085 still uses flat simulator imports; keep that compatibility scoped here.
sys.path[:0] = [
    str(_REPO / "tools"),
    str(_REPO / "tools/snnsim"),
    str(_REPO / "experiments"),
]

from .measurements import (
    analyse_trajectory,
    choose_intermediate,
    circular_distance,
    instantaneous_frequency,
)
from .recipe import K_VALUES, PHASE_BINS, make_inputs

__all__ = [
    "K_VALUES",
    "PHASE_BINS",
    "make_inputs",
    "analyse_trajectory",
    "circular_distance",
    "instantaneous_frequency",
    "choose_intermediate",
]
