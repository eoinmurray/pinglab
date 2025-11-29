
import numpy as np
from pinglab.types import Spikes


def ei_crosscorr(
    spikes: Spikes,
    N_E: int,
    bin_ms: float = 1.0,
    max_lag_ms: float = 20.0,  # <= one gamma cycle
):
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
