"""Rhythmicity metric from autocorrelation peak."""

from __future__ import annotations

import numpy as np


def autocorr_rhythmicity(
    rate_hz: np.ndarray,
    dt_ms: float,
    tau_min_ms: float,
    tau_max_ms: float,
) -> float:
    """
    Compute rhythmicity from the autocorrelation peak of a rate signal.

    Returns the maximum autocorrelation value within [tau_min_ms, tau_max_ms].
    """
    if rate_hz.size == 0:
        return 0.0

    mean = float(np.mean(rate_hz))
    std = float(np.std(rate_hz))
    if std == 0.0:
        return 0.0

    x = (rate_hz - mean) / std
    n = x.size
    corr = np.correlate(x, x, mode="full")[n - 1 :]
    norm = np.arange(n, 0, -1, dtype=float)
    C = corr / norm

    lag_min = max(1, int(np.ceil(tau_min_ms / dt_ms)))
    lag_max = min(n - 1, int(np.floor(tau_max_ms / dt_ms)))
    if lag_max < lag_min:
        return 0.0

    window = C[lag_min : lag_max + 1]
    return float(np.max(window))
