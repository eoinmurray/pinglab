"""Pure recipe and numerical exports; execute an explicit stage.

This package exposes local scientific definitions for its independent stages.
Importing it does not create runs, load scientific artifacts or launch execution.
"""

from experiments.exp038.custom.measurements import summarize_ei_points
from experiments.exp038.recipe import (
    ANALYSIS_PURPOSE,
    CHECKPOINT_POLICY,
    CHECKPOINT_ROLE,
    EVAL_MAX_SAMPLES,
    FR_STRENGTH_UPPER,
    MODELS,
    RATE_TARGET_GRID_HZ,
    SEEDS_BASELINE,
    cell_name,
    rate_target_display,
    seeds_for,
)

__all__ = [
    "ANALYSIS_PURPOSE",
    "CHECKPOINT_POLICY",
    "CHECKPOINT_ROLE",
    "EVAL_MAX_SAMPLES",
    "FR_STRENGTH_UPPER",
    "MODELS",
    "RATE_TARGET_GRID_HZ",
    "SEEDS_BASELINE",
    "cell_name",
    "rate_target_display",
    "seeds_for",
    "summarize_ei_points",
]
