"""Exp023 scientific interface. Run compute, analyse and present independently.

This package exposes local scientific definitions for its independent stages.
Importing it does not create runs, load scientific artifacts or launch execution.
"""

from experiments.exp023.recipe import (
    CELLS,
    DT_MS,
    FI_EI,
    FI_RATES_HZ,
    author_network,
    configuration,
    trials,
)

__all__ = [
    "CELLS",
    "DT_MS",
    "FI_EI",
    "FI_RATES_HZ",
    "author_network",
    "configuration",
    "trials",
]
