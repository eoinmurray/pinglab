"""exp049: explicit compute, analyse and present stages.

This package exposes local scientific definitions for its independent stages.
Importing it does not create runs, load scientific artifacts or launch execution.
"""

from experiments.exp049.custom.measurements import weight_summary
from experiments.exp049.recipe import (
    ANALYSIS_PURPOSE,
    CHECKPOINT_POLICY,
    CHECKPOINT_ROLE,
    COND_ORDER,
    EVAL_MAX_SAMPLES,
    SEEDS,
    cell_name,
)

__all__ = [
    "CHECKPOINT_POLICY",
    "CHECKPOINT_ROLE",
    "ANALYSIS_PURPOSE",
    "EVAL_MAX_SAMPLES",
    "COND_ORDER",
    "SEEDS",
    "cell_name",
    "weight_summary",
]
