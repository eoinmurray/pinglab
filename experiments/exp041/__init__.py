"""Recipe exports only; stages require explicit invocation and source identities.

This package exposes local scientific definitions for its independent stages.
Importing it does not create runs, load scientific artifacts or launch execution.
"""

from experiments.exp041.recipe import (
    ANALYSIS_PURPOSE,
    CHECKPOINT_POLICY,
    CHECKPOINT_ROLE,
    EVAL_MAX_SAMPLES,
    SEEDS,
    TAU_GABA_SWEEP,
    cell_name,
    configuration,
)

__all__ = [
    "ANALYSIS_PURPOSE",
    "CHECKPOINT_POLICY",
    "CHECKPOINT_ROLE",
    "EVAL_MAX_SAMPLES",
    "TAU_GABA_SWEEP",
    "SEEDS",
    "cell_name",
    "configuration",
]
