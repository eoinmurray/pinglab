"""Scientific identities only; execute the three stages explicitly.

This package exposes local scientific definitions for its independent stages.
Importing it does not create runs, load scientific artifacts or launch execution.
"""

from experiments.exp044.recipe import (
    ANALYSIS_PURPOSE,
    CHECKPOINT_POLICY,
    CHECKPOINT_ROLE,
    DT_SWEEP_MS,
    EVAL_MAX_SAMPLES,
    SEEDS,
    cell_name,
    dt_label,
)

__all__ = [
    "ANALYSIS_PURPOSE",
    "CHECKPOINT_POLICY",
    "CHECKPOINT_ROLE",
    "DT_SWEEP_MS",
    "EVAL_MAX_SAMPLES",
    "SEEDS",
    "cell_name",
    "dt_label",
]
