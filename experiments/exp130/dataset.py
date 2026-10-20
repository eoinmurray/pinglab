"""Generate independent private afferent counts for the recorded protocol.

The compute and analysis stages provide resolved recipe settings. These helpers
construct the prescribed rate envelope and seeded Poisson counts without loading
runs, compiling networks or launching execution on import.
"""

import numpy as np

from .recipe import CONDITIONS


def source_rates(times_ms, cfg):
    """Per-afferent rate; both targets share only the deterministic envelope."""
    times_ms = np.asarray(times_ms)
    condition = cfg["condition"]
    if condition not in CONDITIONS:
        raise ValueError(condition)
    phase = 2 * np.pi * cfg["theta_hz"] * (times_ms - cfg["modulation_onset_ms"]) / 1000
    oscillation = (
        cfg["modulation_hz"] * np.sin(phase)
        if condition == "theta"
        else np.zeros_like(times_ms)
    )
    oscillation = np.where(times_ms >= cfg["modulation_onset_ms"], oscillation, 0)
    return tuple(cfg[f"baseline_{pop}_hz"] + oscillation for pop in ("e", "i"))


def afferent_counts(cfg):
    """Independent across cells and populations; counts greater than one survive."""
    times = np.arange(round(cfg["t_ms"] / cfg["dt_ms"])) * cfg["dt_ms"]
    rngs = [
        np.random.default_rng(s) for s in np.random.SeedSequence(cfg["seed"]).spawn(2)
    ]
    result = {}
    for pop, rates, rng in zip(("e", "i"), source_rates(times, cfg), rngs):
        lam = (
            rates[:, None] * cfg["external_afferents_per_neuron"] * cfg["dt_ms"] / 1000
        )
        counts = rng.poisson(lam, size=(len(times), cfg[f"n_{pop}"]))
        if counts.max() > 255:
            raise ValueError("afferent counts exceed recording capacity")
        result[f"private_{pop}"] = counts.astype(np.uint8)
    return result
