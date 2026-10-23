"""Scientific recipe only; invoke compute, analyse and present explicitly.

This package exposes local scientific definitions for its independent stages.
Importing it does not create runs, load scientific artifacts or launch execution.
"""

from experiments.exp042.recipe import (
    ANALYSIS_PURPOSE,
    CELL_JITTER_SIGMAS_MS,
    CHECKPOINT_POLICY,
    CHECKPOINT_ROLE,
    EVAL_MAX_SAMPLES,
    JITTER_SIGMAS_MS,
    SEEDS,
    cell_name,
    configuration,
    jobs,
)

__all__ = [
    "ANALYSIS_PURPOSE",
    "CHECKPOINT_POLICY",
    "CHECKPOINT_ROLE",
    "SEEDS",
    "JITTER_SIGMAS_MS",
    "CELL_JITTER_SIGMAS_MS",
    "EVAL_MAX_SAMPLES",
    "cell_name",
    "configuration",
    "jobs",
]
