from __future__ import annotations

import numpy as np


def build_input_projection(
    *,
    N_E: int,
    num_fibers: int,
    targets_per_fiber: int,
    seed: int,
) -> np.ndarray:
    """
    Build a fixed random projection from input fibers to E neurons.

    Returns an array of shape (num_fibers, targets_per_fiber) with neuron IDs.
    """
    if targets_per_fiber > N_E:
        raise ValueError("targets_per_fiber cannot exceed N_E")

    rng = np.random.default_rng(seed)
    targets = np.empty((num_fibers, targets_per_fiber), dtype=np.int32)
    for i in range(num_fibers):
        targets[i, :] = rng.choice(N_E, size=targets_per_fiber, replace=False)
    return targets
