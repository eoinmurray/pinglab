"""Pure recipe/measurement exports; execute an explicit stage.

This package exposes local scientific definitions for its independent stages.
Importing it does not create runs, load scientific artifacts or launch execution.
"""

from experiments.exp037.custom.measurements import (
    summarize_accuracy,
    summarize_frontier,
    summarize_perturbation_rows,
)
from experiments.exp037.recipe import (
    ANALYSIS_PURPOSE,
    CHECKPOINT_POLICY,
    CHECKPOINT_ROLE,
    EVAL_MAX_SAMPLES,
    FR_STRENGTH_UPPER,
    MODELS,
    PERTURB_ADD_LEVELS,
    PERTURB_DROP_LEVELS,
    PERTURB_RASTER_ADD_LEVELS,
    PERTURB_RASTER_DROP_LEVELS,
    RATE_TARGET_GRID_HZ,
    SEEDS_BASELINE,
    _parse_job,
    cell_name,
    infer_jobs,
    rate_target_display,
    seeds_for,
)

__all__ = [
    "summarize_frontier",
    "summarize_accuracy",
    "summarize_perturbation_rows",
    "ANALYSIS_PURPOSE",
    "CHECKPOINT_POLICY",
    "CHECKPOINT_ROLE",
    "EVAL_MAX_SAMPLES",
    "FR_STRENGTH_UPPER",
    "MODELS",
    "RATE_TARGET_GRID_HZ",
    "SEEDS_BASELINE",
    "PERTURB_DROP_LEVELS",
    "PERTURB_ADD_LEVELS",
    "PERTURB_RASTER_DROP_LEVELS",
    "PERTURB_RASTER_ADD_LEVELS",
    "cell_name",
    "rate_target_display",
    "seeds_for",
    "infer_jobs",
    "_parse_job",
]
