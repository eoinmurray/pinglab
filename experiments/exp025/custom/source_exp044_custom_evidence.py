"""Validate retained cells and measurements without running a simulator.

The owning experiment calls these local definitions from its explicit stages.
Inputs and outputs retain that stage’s validated scientific contract; importing
this module does not start computation or select a scientific run.
"""

from pathlib import Path

import numpy as np
from experiments.exp025.custom.source_exp044_recipe import duration_steps
from pingstore.contracts import PingstoreError


def snapshot(path: Path, dt: float, common: dict) -> dict:
    with np.load(path, allow_pickle=False) as data:
        if not np.isclose(float(data["dt"]), dt):
            raise PingstoreError("snapshot timestep mismatch")
        result = {key: np.array(data[key]) for key in ("spk_e", "spk_i")}
    steps = duration_steps(common["t_ms"], dt)
    for key, population in (("spk_e", "n_hidden"), ("spk_i", "n_inh")):
        array = result[key]
        if array.shape != (steps, common[population]) or not np.all(
            (array == 0) | (array == 1)
        ):
            raise PingstoreError(f"invalid snapshot shape or spikes for {key}")
    return result
