"""Exp022 public scientific interface; execution lives in explicit stages.

This package exposes local scientific definitions for its independent stages.
Importing it does not create runs, load scientific artifacts or launch execution.
"""

import sys
from pathlib import Path

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2]),
    str(Path(__file__).resolve().parents[1]),
    str(Path(__file__).resolve().parents[2] / "tools"),
]
from experiments.exp022.recipe import *
