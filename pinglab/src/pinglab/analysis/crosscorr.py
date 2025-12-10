"""E→I cross-correlation analysis for spike timing."""

import numpy as np
from pinglab.types import Spikes


def crosscorr(
    spikes: Spikes,
    N_E: int,
    bin_ms: float = 1.0,
    max_lag_ms: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute E→I cross-correlation histogram.

    Measures temporal lag between E spikes and subsequent I spikes
    within a specified time window.

    Note:
        Returns raw spike counts per bin, not normalized by the number of
        E spikes or bin width. To get a rate, divide by len(tE) * bin_ms.

    Parameters:
        spikes: Spike data with times and neuron IDs
        N_E: Number of excitatory neurons (IDs 0 to N_E-1 are E)
        bin_ms: Histogram bin width in milliseconds
        max_lag_ms: Maximum lag to consider (default 20ms for gamma)

    Returns:
        (centers, hist): Bin centers (ms) and raw spike counts per bin
    """
    tE = spikes.times[spikes.ids < N_E]
    tI = spikes.times[spikes.ids >= N_E]

    lags = []
    for t in tE:
        # only I spikes AFTER the E spike, within one cycle
        mask = (tI >= t) & (tI <= t + max_lag_ms)
        lags.extend(tI[mask] - t)

    bins = np.arange(0.0, max_lag_ms + bin_ms, bin_ms)
    hist, edges = np.histogram(lags, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, hist
