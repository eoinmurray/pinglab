"""Exp022 public scientific interface; execution lives in explicit stages."""

import importlib
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[2]),
                str(Path(__file__).resolve().parents[1]),
                str(Path(__file__).resolve().parents[2] / "tools")]
from .recipe import *  # noqa: F403


def __getattr__(name):
    # Existing scheduler hooks remain import-compatible without executing work.
    if name.startswith("__"):
        raise AttributeError(name)
    module = importlib.import_module("experiments.exp022.compute")
    try:
        return getattr(module, name)
    except AttributeError:
        raise AttributeError(name) from None
