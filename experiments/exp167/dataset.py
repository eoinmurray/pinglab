"""Generate constant-rate spikes from the one shared external input pool.

Independent Poisson-source streams feed fixed Input→E and Input→I projections.
Blockwise generation bounds temporary storage and retains reproducible draws.
"""

import numpy as np
import torch


def encode_drive(cfg):
    steps = round(cfg["duration_ms"] / cfg["dt_ms"])
    rng = np.random.default_rng(cfg["seed"] + cfg["external_seed_offset"])
    values = np.empty((steps, 1, cfg["n_external"]), dtype=np.float32)
    probability = cfg["external_rate_hz"] * cfg["dt_ms"] / 1000.0
    for start in range(0, steps, cfg["recording_block_steps"]):
        block = values[start : start + cfg["recording_block_steps"]]
        block[:] = rng.random(block.shape, dtype=np.float32) < probability
    return torch.from_numpy(values)
