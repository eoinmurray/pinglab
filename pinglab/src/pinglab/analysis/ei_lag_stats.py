
import numpy as np
from pinglab.types import Spikes


def ei_lag_stats(
    spikes: Spikes,
    N_E: int,
    max_lag_ms: float = 10.0,
) -> tuple[float, float]:
    """
    Compute stats of E→I spike time lags.

    Returns
    -------
    lag_mean_ms : float
        Mean time (ms) from an E spike to the next I spike within max_lag_ms.
    lag_std_ms : float
        Standard deviation of those lags.
    """
    times = spikes.times
    ids = spikes.ids

    # Split populations: 0..N_E-1 = E, N_E.. = I
    mask_E = ids < N_E
    mask_I = ids >= N_E

    tE = times[mask_E]
    tI = times[mask_I]

    if tE.size == 0 or tI.size == 0:
        return 0.0, 0.0

    # Ensure sorted (should already be, but be paranoid)
    tE = np.sort(tE)
    tI = np.sort(tI)

    # For each E spike, find first I spike at or after tE
    idx = np.searchsorted(tI, tE, side="left")  # indices into tI

    valid = idx < tI.size
    if not np.any(valid):
        return 0.0, 0.0

    # Candidate lags (ms)
    lags = np.full(tE.shape, np.nan, dtype=float)
    lags[valid] = tI[idx[valid]] - tE[valid]

    # Keep only lags within [0, max_lag_ms]
    valid = valid & (lags >= 0.0) & (lags <= max_lag_ms)
    lags = lags[valid]

    if lags.size == 0:
        return 0.0, 0.0

    lag_mean = float(np.mean(lags))
    lag_std = float(np.std(lags))

    return lag_mean, lag_std
