"""Oscillation coherence via mean pairwise cross-correlation of smoothed rates."""

from __future__ import annotations

import numpy as np

from pinglab.backends.types import Spikes


def _gaussian_kernel(dt_ms: float, sigma_ms: float) -> np.ndarray:
    if sigma_ms <= 0:
        raise ValueError(f"sigma_ms must be positive, got {sigma_ms}")
    if dt_ms <= 0:
        raise ValueError(f"dt_ms must be positive, got {dt_ms}")
    kernel_radius_ms = max(1.0, 4.0 * sigma_ms)
    kernel_half = int(np.ceil(kernel_radius_ms / dt_ms))
    kernel_times = np.arange(-kernel_half, kernel_half + 1) * dt_ms
    kernel = np.exp(-0.5 * (kernel_times / sigma_ms) ** 2)
    kernel = kernel / np.sum(kernel)
    return kernel


def coherence(
    spikes: Spikes,
    T_ms: float,
    *,
    N_E: int,
    dt_ms: float = 5.0,
    sigma_ms: float = 10.0,
    max_lag_ms: float = 400.0,
    normalize: bool = True,
    abs_value: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Oscillation coherence as mean pairwise cross-correlation of smoothed rates.

    Parameters:
        spikes: Spike data (times in ms, ids)
        T_ms: Total simulation time in ms
        N_E: Number of excitatory neurons
        dt_ms: Bin width in ms
        sigma_ms: Gaussian smoothing sigma in ms
        max_lag_ms: Maximum lag for cross-correlation window

    Returns:
        lags_ms, coherence_curve
    """
    if T_ms <= 0 or dt_ms <= 0:
        raise ValueError("T_ms and dt_ms must be positive")
    if N_E <= 1:
        return np.array([]), np.array([])

    n_bins = int(np.ceil(T_ms / dt_ms))
    edges = np.linspace(0.0, T_ms, n_bins + 1)
    max_lag_bins = int(np.floor(max_lag_ms / dt_ms))
    max_lag_bins = min(max_lag_bins, n_bins - 1)

    kernel = _gaussian_kernel(dt_ms, sigma_ms)

    sum_series = np.zeros(n_bins, dtype=float)
    sum_auto = np.zeros(n_bins, dtype=float)
    stds: list[float] = []

    for nid in range(N_E):
        neuron_times = spikes.times[spikes.ids == nid]
        if neuron_times.size == 0:
            continue
        counts, _ = np.histogram(neuron_times, bins=edges)
        series = counts.astype(float)
        smooth = np.convolve(series, kernel, mode="same")
        if smooth.size != n_bins:
            # For very short simulations, convolution with a wide kernel can
            # exceed n_bins even in "same" mode. Center-crop back to n_bins.
            extra = smooth.size - n_bins
            left = max(0, extra // 2)
            smooth = smooth[left : left + n_bins]
        smooth = smooth - float(np.mean(smooth))
        sum_series += smooth
        if normalize:
            stds.append(float(np.std(smooth)))
        auto = np.correlate(smooth, smooth, mode="full")[n_bins - 1 :]
        sum_auto += auto

    total_corr_full = np.correlate(sum_series, sum_series, mode="full")
    auto_full = np.concatenate((sum_auto[1:][::-1], sum_auto))
    pair_corr_full = (total_corr_full - auto_full) / (N_E * (N_E - 1))
    lags_full = np.arange(-n_bins + 1, n_bins)
    norm = n_bins - np.abs(lags_full)
    pair_corr_full = pair_corr_full / norm
    if normalize and stds:
        stds_arr = np.array(stds)
        denom = (np.sum(stds_arr) ** 2 - np.sum(stds_arr**2)) / (N_E * (N_E - 1))
        if denom > 0:
            pair_corr_full = pair_corr_full / denom

    center = n_bins - 1
    lags = np.arange(-max_lag_bins, max_lag_bins + 1) * dt_ms
    start = center - max_lag_bins
    end = center + max_lag_bins + 1
    corr = pair_corr_full[start:end]
    if abs_value:
        corr = np.clip(np.abs(corr), 0.0, 1.0)
    return lags, corr


def coherence_peak(
    spikes: Spikes,
    T_ms: float,
    *,
    N_E: int,
    dt_ms: float = 5.0,
    sigma_ms: float = 10.0,
    max_lag_ms: float = 400.0,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Scalar coherence as the peak of the coherence curve over lag.

    Returns:
        peak_value, lags_ms, coherence_curve
    """
    lags_ms, corr = coherence(
        spikes,
        T_ms=T_ms,
        N_E=N_E,
        dt_ms=dt_ms,
        sigma_ms=sigma_ms,
        max_lag_ms=max_lag_ms,
        normalize=True,
        abs_value=True,
    )
    if corr.size == 0:
        return 0.0, lags_ms, corr
    return float(np.max(corr)), lags_ms, corr
