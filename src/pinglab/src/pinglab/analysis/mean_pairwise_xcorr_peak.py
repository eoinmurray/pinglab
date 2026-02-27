"""Mean pairwise E-E spike cross-correlation peak."""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from pinglab.backends.types import Spikes


def _first_significant_peak(
    lags_ms: np.ndarray,
    corr: np.ndarray,
    min_lag_ms: float,
    max_lag_ms: float,
    peak_min: float,
    peak_prominence: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    if lags_ms.size == 0 or corr.size == 0:
        return 0.0
    mask = (lags_ms >= min_lag_ms) & (lags_ms <= max_lag_ms)
    if not np.any(mask):
        return 0.0
    idxs = np.where(mask)[0]
    window = corr[idxs]
    peaks, _ = find_peaks(window, height=peak_min, prominence=peak_prominence)
    if peaks.size > 0:
        peak_idx = idxs[int(peaks[0])]
        return float(corr[peak_idx])
    max_idx = idxs[int(np.argmax(window))]
    return float(corr[max_idx])


def _mean_pairwise_xcorr(
    spikes: Spikes,
    T_ms: float,
    dt_ms: float,
    N_E: int,
    max_lag_ms: float,
) -> tuple[np.ndarray, np.ndarray]:
    if N_E <= 1:
        return np.array([]), np.array([])

    n_bins = int(np.ceil(T_ms / dt_ms))
    edges = np.linspace(0.0, T_ms, n_bins + 1)
    max_lag_bins = int(np.floor(max_lag_ms / dt_ms))

    sum_series = np.zeros(n_bins, dtype=float)
    sum_auto = np.zeros(n_bins, dtype=float)
    stds: list[float] = []

    for nid in range(N_E):
        neuron_times = spikes.times[spikes.ids == nid]
        if neuron_times.size == 0:
            continue
        counts, _ = np.histogram(neuron_times, bins=edges)
        series = counts.astype(float)
        series -= np.mean(series)
        if np.allclose(series, 0.0):
            continue
        sum_series += series
        stds.append(float(np.std(series)))
        auto = np.correlate(series, series, mode="full")[n_bins - 1 :]
        sum_auto += auto

    total_corr_full = np.correlate(sum_series, sum_series, mode="full")
    auto_full = np.concatenate((sum_auto[1:][::-1], sum_auto))
    pair_corr_full = (total_corr_full - auto_full) / (N_E * (N_E - 1))
    lags_full = np.arange(-n_bins + 1, n_bins)
    norm = n_bins - np.abs(lags_full)
    pair_corr_full = pair_corr_full / norm

    if stds:
        stds_arr = np.array(stds)
        denom = (np.sum(stds_arr) ** 2 - np.sum(stds_arr**2)) / (N_E * (N_E - 1))
        if denom > 0:
            pair_corr_full = pair_corr_full / denom

    center = n_bins - 1
    max_lag_bins = min(max_lag_bins, center)
    lags = np.arange(-max_lag_bins, max_lag_bins + 1) * dt_ms
    start = center - max_lag_bins
    end = center + max_lag_bins + 1
    return lags, pair_corr_full[start:end]


def mean_pairwise_xcorr_peak(
    spikes: Spikes,
    T_ms: float,
    *,
    N_E: int,
    dt_ms: float = 5.0,
    xcorr_max_lag_ms: float = 400.0,
    corr_min_lag_ms: float = 20.0,
    corr_max_lag_ms: float = 150.0,
    corr_peak_min: float = 0.1,
    corr_peak_prominence: float = 0.02,
) -> float:
    """
    Mean pairwise E-E spike cross-correlation peak.

    Returns:
        peak_value, lags_ms, corr
    """
    if T_ms <= 0 or dt_ms <= 0:
        raise ValueError("T_ms and dt_ms must be positive")
    lags_ms, pair_corr = _mean_pairwise_xcorr(
        spikes,
        T_ms=T_ms,
        dt_ms=dt_ms,
        N_E=N_E,
        max_lag_ms=xcorr_max_lag_ms,
    )
    peak = _first_significant_peak(
        lags_ms,
        pair_corr,
        corr_min_lag_ms,
        corr_max_lag_ms,
        corr_peak_min,
        corr_peak_prominence,
    )
    return peak, lags_ms, pair_corr
