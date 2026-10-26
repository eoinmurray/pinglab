"""Generate independent private afferent counts for the recorded protocol.

The compute and analysis stages provide resolved recipe settings. These helpers
construct the prescribed rate envelope and seeded Poisson counts without loading
runs, compiling networks or launching execution on import.
"""

import numpy as np


def source_rates(times_ms, cfg):
    rates = []
    for pop in ("e", "i"):
        base = cfg[f"baseline_{pop}_hz"]
        peak = cfg.get(f"stimulus_{pop}_hz", base)
        knots = [0.0, cfg["onset_ms"], cfg["peak_ms"]]
        values = [base, base, peak]
        if cfg["offset_ms"] > cfg["plateau_end_ms"]:
            knots.extend([cfg["plateau_end_ms"], cfg["offset_ms"]])
            values.extend([peak, base])
        rates.append(np.interp(times_ms, knots, values))
    return tuple(rates)


def afferent_counts(cfg):
    """Independent across cells and populations; counts greater than one survive."""
    times = np.arange(round(cfg["t_ms"] / cfg["dt_ms"])) * cfg["dt_ms"]
    rngs = [
        np.random.default_rng(s) for s in np.random.SeedSequence(cfg["seed"]).spawn(2)
    ]
    result = {}
    for pop, rates, rng in zip(("e", "i"), source_rates(times, cfg), rngs, strict=True):
        lam = (
            rates[:, None] * cfg["external_afferents_per_neuron"] * cfg["dt_ms"] / 1000
        )
        counts = rng.poisson(lam, size=(len(times), cfg[f"n_{pop}"]))
        if counts.max() > 255:
            raise ValueError("afferent counts exceed recording capacity")
        result[f"private_{pop}"] = counts.astype(np.uint8)
    return result
