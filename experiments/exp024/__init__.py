"""Exp024 scientific interface. Execute analyse and present separately.

This package exposes local scientific definitions for its independent stages.
Importing it does not create runs, load scientific artifacts or launch execution.
"""

from experiments.exp024.recipe import MODELS, SEEDS, cell_name

__all__ = ["MODELS", "SEEDS", "cell_name"]
