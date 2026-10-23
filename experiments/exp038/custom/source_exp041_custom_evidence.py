"""Validate retained cells and measurements without running a simulator.

The owning experiment calls these local definitions from its explicit stages.
Inputs and outputs retain that stage’s validated scientific contract; importing
this module does not start computation or select a scientific run.
"""

import math

import numpy as np

from pingstore.contracts import PingstoreError, load_json


def finite(value, label: str, *, minimum=0.0, maximum=None) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        raise PingstoreError(f"invalid or missing {label}")
    return float(value)


def _same(actual, expected) -> bool:
    if isinstance(actual, bool) or isinstance(expected, bool):
        return type(actual) is type(expected) and actual == expected
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return bool(np.isclose(actual, expected))
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _same(a, b) for a, b in zip(actual, expected)
        )
    if isinstance(actual, dict) and isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            _same(actual[k], expected[k]) for k in actual
        )
    return actual == expected
