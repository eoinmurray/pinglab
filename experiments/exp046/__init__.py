"""Recipe exports only; execution requires an explicit stage invocation.

This package exposes local scientific definitions for its independent stages.
Importing it does not create runs, load scientific artifacts or launch execution.
"""

from experiments.exp046.recipe import (
    ANALYSIS_PURPOSE,
    CHECKPOINT_POLICY,
    CHECKPOINT_ROLE,
    EVAL_MAX_SAMPLES,
    SEEDS,
    TAU_GABA_SWEEP_MS,
    cell_name,
    configuration,
)

__all__ = [
    "ANALYSIS_PURPOSE",
    "CHECKPOINT_POLICY",
    "CHECKPOINT_ROLE",
    "EVAL_MAX_SAMPLES",
    "SEEDS",
    "TAU_GABA_SWEEP_MS",
    "cell_name",
    "configuration",
]
