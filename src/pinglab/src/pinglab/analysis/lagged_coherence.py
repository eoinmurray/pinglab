"""Lagged coherence (rhythmicity) for spike trains."""

from __future__ import annotations

import numpy as np

from pinglab.backends.types import Spikes
from .population_rate import population_rate


def _windowed_fourier_coeffs(
    times_ms: np.ndarray,
    rate_hz: np.ndarray,
    window_ms: float,
    lag_ms: float,
    freq_hz: float,
    remove_mean: bool,
    taper: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if times_ms.size == 0 or rate_hz.size == 0:
        return np.array([])
    if window_ms <= 0 or lag_ms <= 0 or freq_hz <= 0:
        return np.array([])
    start = float(times_ms[0])
    end = float(times_ms[-1])
    coeffs: list[complex] = []
    centers: list[float] = []
    windows: list[tuple[float, float]] = []
    t = start
    while t + window_ms <= end + 1e-9:
        mask = (times_ms >= t) & (times_ms < t + window_ms)
        if np.any(mask):
            window_t = (times_ms[mask] - t) * 1e-3
            window_x = rate_hz[mask].astype(float)
            if remove_mean:
                window_x = window_x - float(np.mean(window_x))
            if taper == "hann":
                window_x = window_x * np.hanning(window_x.size)
            expo = np.exp(-1j * 2.0 * np.pi * freq_hz * window_t)
            coeffs.append(np.mean(window_x * expo))
            centers.append(t + 0.5 * window_ms)
            windows.append((t, t + window_ms))
        t += lag_ms
    if not coeffs:
        return np.array([]), np.array([]), np.array([])
    return np.array(coeffs), np.array(centers), np.array(windows)


def _lagged_coherence_from_coeffs(coeffs: np.ndarray) -> float:
    if coeffs.size < 2:
        return 0.0
    products = coeffs[:-1] * np.conjugate(coeffs[1:])
    avg = np.mean(products)
    mags_a = np.abs(coeffs[:-1]) ** 2
    mags_b = np.abs(coeffs[1:]) ** 2
    denom = float(np.sqrt(np.mean(mags_a) * np.mean(mags_b)))
    if denom == 0.0:
        return 0.0
    value = float(np.abs(avg) / denom)
    return float(np.clip(value, 0.0, 1.0))


def lagged_coherence(
    spikes: Spikes,
    T_ms: float,
    dt_ms: float = 5.0,
    freq_hz: float = 5.0,
    *,
    window_cycles: float = 3.0,
    lag_cycles: float = 3.0,
    pop: str = "E",
    N_E: int | None = None,
    N_I: int | None = None,
    remove_mean: bool = True,
    taper: str = "hann",
) -> tuple[
    float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """
    Compute lagged coherence (rhythmicity) for spike data.

    Parameters:
        spikes: Spike data (times in ms, ids)
        T_ms: Total simulation time in ms
        dt_ms: Population rate bin size in ms (default 5.0)
        freq_hz: Target frequency in Hz (default 5.0)
        window_cycles: Window length in cycles of freq_hz
        lag_cycles: Lag between windows in cycles of freq_hz
        pop: Population to analyze ('E', 'I', or 'all')
        N_E: Number of excitatory neurons
        N_I: Number of inhibitory neurons
        remove_mean: Subtract per-window mean before coefficient
        taper: Window taper ('hann' or 'none')

    Returns:
        lam, t_ms, rate_hz, windows, coeffs, phase_mag
    """
    if freq_hz <= 0:
        raise ValueError(f"freq_hz must be positive, got {freq_hz}")
    if window_cycles <= 0:
        raise ValueError(f"window_cycles must be positive, got {window_cycles}")
    if lag_cycles <= 0:
        raise ValueError(f"lag_cycles must be positive, got {lag_cycles}")
    if taper not in ("hann", "none"):
        raise ValueError("taper must be 'hann' or 'none'")

    t_ms, rate_hz = population_rate(
        spikes,
        T_ms=T_ms,
        dt_ms=dt_ms,
        pop=pop,
        N_E=N_E,
        N_I=N_I,
    )
    window_ms = float(window_cycles / freq_hz * 1000.0)
    lag_ms = float(lag_cycles / freq_hz * 1000.0)
    coeffs, centers_ms, windows = _windowed_fourier_coeffs(
        t_ms,
        rate_hz,
        window_ms,
        lag_ms,
        freq_hz,
        remove_mean,
        taper,
    )
    if coeffs.size < 2:
        return 0.0, t_ms, rate_hz, windows, coeffs, np.array([])

    lam = _lagged_coherence_from_coeffs(coeffs)
    phase_mag = np.column_stack((np.angle(coeffs), np.abs(coeffs), centers_ms))
    return lam, t_ms, rate_hz, windows, coeffs, phase_mag

