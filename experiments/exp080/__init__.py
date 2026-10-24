"""EXP080 independent stages; compatibility exports are pure recipe helpers.

This package exposes local scientific definitions for its independent stages.
Importing it does not create runs, load scientific artifacts or launch execution.
"""

from experiments.exp080.analyse import analyze
from experiments.exp080.recipe import (
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
