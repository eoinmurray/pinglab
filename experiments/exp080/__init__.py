"""EXP080 independent stages; compatibility exports are pure recipe helpers."""

from .measurements import analyze
from .recipe import (
    EPOCHS_STANDARD,
    RATES_HZ,
    SEEDS,
    USEFUL_ACCURACY,
    probe_single_spike,
    validate_simulator,
)

__all__ = [
    "EPOCHS_STANDARD",
    "RATES_HZ",
    "SEEDS",
    "USEFUL_ACCURACY",
    "probe_single_spike",
    "validate_simulator",
    "analyze",
]
