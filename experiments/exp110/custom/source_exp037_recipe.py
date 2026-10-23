"""Explicit absolute-rate and baseline-relative perturbation recipes.

The owning experiment calls these local definitions from its explicit stages.
Inputs and outputs retain that stage’s validated scientific contract; importing
this module does not start computation or select a scientific run.
"""

from experiments.exp110.custom.source_exp037_custom_bank_contract import (
    training_run_values,
)

MODELS = list(training_run_values("TR-02", "model"))
