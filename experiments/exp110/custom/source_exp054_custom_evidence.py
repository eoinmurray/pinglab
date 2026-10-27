"""Lossless storage and validation of current and retained scientific evidence.

The owning experiment calls these local definitions from its explicit stages.
Inputs and outputs retain that stage’s validated scientific contract; importing
this module does not start computation or select a scientific run.
"""

import math

import numpy as np
from pingstore.library import PingstoreError, load_json


def read(directory):
    used = set()
    with np.load(directory / "arrays.npz", allow_pickle=False) as arrays:

        def unpack(value):
            if isinstance(value, dict) and set(value) == {"__array__"}:
                name = value["__array__"]
                if name not in arrays:
                    raise PingstoreError("missing exp054 array")
                a = arrays[name]
                if a.dtype.kind not in "biuf" or np.isinf(a).any():
                    raise PingstoreError("invalid exp054 numerical array")
                used.add(name)
                return a
            if value == {"__float__": "nan"}:
                return float("nan")
            if isinstance(value, dict):
                return {k: unpack(v) for k, v in value.items()}
            if isinstance(value, list):
                return [unpack(v) for v in value]
            if isinstance(value, float) and not math.isfinite(value):
                raise PingstoreError("invalid exp054 scalar")
            return value

        document = unpack(load_json(directory / "evidence.json"))
        if used != set(arrays.files):
            raise PingstoreError("unreferenced exp054 arrays")
    return document
