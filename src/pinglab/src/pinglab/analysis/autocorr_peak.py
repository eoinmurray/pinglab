"""Autocorrelation peak of the population rate."""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from pinglab.backends.types import Spikes
from .population_rate import population_rate


def _autocorr_full(signal: np.ndarray) -> tuple[np.ndarray, int]:
    if signal.size == 0:
        return np.array([]), 0
    x = signal - float(np.mean(signal))
    var = float(np.var(x))
    if np.allclose(x, 0.0) or var == 0.0:
        return np.zeros(2 * signal.size - 1), signal.size - 1
    corr = np.correlate(x, x, mode="full")
    n = signal.size
    center = n - 1
    norm = n - np.abs(np.arange(-center, center + 1))
    corr = corr / norm
    corr = corr / var
    return corr, center


def _first_significant_peak(
    lags_ms: np.ndarray,
    corr: np.ndarray,
    min_lag_ms: float,
    max_lag_ms: float,
    peak_min: float,
    peak_prominence: float,
) -> float:
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


def first_significant_peak_lag_ms(
    lags_ms: np.ndarray,
    corr: np.ndarray,
    *,
    corr_min_lag_ms: float = 20.0,
    corr_max_lag_ms: float = 150.0,
    corr_peak_min: float = 0.1,
    corr_peak_prominence: float = 0.02,
) -> float:
    if lags_ms.size == 0 or corr.size == 0:
        return 0.0
    mask = (lags_ms >= corr_min_lag_ms) & (lags_ms <= corr_max_lag_ms)
    if not np.any(mask):
        return 0.0
    idxs = np.where(mask)[0]
    window = corr[idxs]
    peaks, _ = find_peaks(window, height=corr_peak_min, prominence=corr_peak_prominence)
    if peaks.size > 0:
        peak_idx = idxs[int(peaks[0])]
        return float(lags_ms[peak_idx])
    max_idx = idxs[int(np.argmax(window))]
    return float(lags_ms[max_idx])


def autocorr_peak(
    spikes: Spikes,
    T_ms: float,
    dt_ms: float = 5.0,
    *,
    pop: str = "E",
    N_E: int | None = None,
    N_I: int | None = None,
    smooth_sigma_ms: float | None = 10.0,
    smooth_bin_ms: float | None = 5.0,
    autocorr_max_lag_ms: float = 400.0,
    corr_min_lag_ms: float = 20.0,
    corr_max_lag_ms: float = 150.0,
    corr_peak_min: float = 0.1,
    corr_peak_prominence: float = 0.02,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Autocorrelation peak of the (smoothed) population rate.

    Returns:
        peak_value, t_ms, rate_hz, lags_ms, corr
    """
    if T_ms <= 0 or dt_ms <= 0:
        raise ValueError("T_ms and dt_ms must be positive")
    t_ms, rate_hz = population_rate(
        spikes,
        T_ms=T_ms,
        dt_ms=dt_ms,
        pop=pop,
        N_E=N_E,
        N_I=N_I,
        smooth_sigma_ms=smooth_sigma_ms,
        smooth_bin_ms=smooth_bin_ms,
    )
    auto_full, auto_center = _autocorr_full(rate_hz)
    max_lag_bins = int(np.floor(autocorr_max_lag_ms / dt_ms))
    max_lag_bins = min(max_lag_bins, auto_center)
    lags_ms = np.arange(-max_lag_bins, max_lag_bins + 1) * dt_ms
    start = auto_center - max_lag_bins
    end = auto_center + max_lag_bins + 1

    peak = _first_significant_peak(
        lags_ms,
        auto_full[start:end],
        corr_min_lag_ms,
        corr_max_lag_ms,
        corr_peak_min,
        corr_peak_prominence,
    )
    return peak, t_ms, rate_hz, lags_ms, auto_full[start:end]
