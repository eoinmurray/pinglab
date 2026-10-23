"""Encode the common external Poisson population without loading a dataset.

Each X neuron has an independent stream. Both X projections consume this same
array, preserving shared events across targets and across the E/I populations.
The returned time-by-batch-by-source tensor is deterministic for a given recipe.
"""

import numpy as np
import torch


def encode_drive(cfg):
    steps = round(cfg["duration_ms"] / cfg["dt_ms"])
    rng = np.random.default_rng(cfg["seed"] + cfg["external_seed_offset"])
    probability = cfg["external_rate_hz"] * cfg["dt_ms"] / 1000.0
    values = np.empty((steps, 1, cfg["n_external"]), dtype=np.float32)
    # Preserve the RNG stream while bounding the temporary allocation for long runs.
    for start in range(0, steps, cfg["recording_block_steps"]):
        block = values[start : start + cfg["recording_block_steps"]]
        block[:] = rng.random(block.shape, dtype=np.float32) < probability
    return torch.from_numpy(values)
