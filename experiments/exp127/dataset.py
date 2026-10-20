"""Build seeded synthetic input splits for the six-weight classifier.

The caller supplies a resolved recipe, sample count and split-specific seed.
Return binary spike occupancy, labels, rates and original Poisson counts; no
files or runs are read or written on import.
"""

import numpy as np


def dataset(samples, seed, cfg):
    rng = np.random.default_rng(seed)
    labels = np.arange(samples, dtype=np.int64) % 2
    rng.shuffle(labels)
    rates = np.asarray(cfg["rate_patterns_hz"])[labels] * rng.uniform(
        *cfg["rate_jitter"], size=(samples, 3)
    )
    steps = round(cfg["duration_ms"] / cfg["dt_ms"])
    counts = rng.poisson(rates * cfg["dt_ms"] / 1000, size=(steps, samples, 3))
    spikes = (counts > 0).astype(np.float32)
    return spikes, labels, rates, counts
