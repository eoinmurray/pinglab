"""First-order low-pass filtering utilities for decoded rate analysis."""

import numpy as np
from scipy.signal import lfilter


def lowpass_first_order(values: np.ndarray, dt_ms: float, cutoff_hz: float) -> np.ndarray:
    """
    Apply a causal first-order low-pass filter to a 1D signal.

    Args:
        values: Input series.
        dt_ms: Sample interval in milliseconds.
        cutoff_hz: Low-pass cutoff frequency in Hz.

    Returns:
        Filtered series, same shape as input.
    """
    x = np.asarray(values, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"values must be 1D, got shape {x.shape}")
    if x.size == 0:
        return np.array([], dtype=float)
    if dt_ms <= 0:
        raise ValueError(f"dt_ms must be positive, got {dt_ms}")
    if cutoff_hz <= 0:
        raise ValueError(f"cutoff_hz must be positive, got {cutoff_hz}")

    dt_sec = float(dt_ms) / 1000.0
    tau_c = 1.0 / (2.0 * np.pi * float(cutoff_hz))
    alpha = dt_sec / (tau_c + dt_sec)

    # y[n] = alpha * x[n] + (1 - alpha) * y[n - 1]
    b = np.array([alpha], dtype=float)
    a = np.array([1.0, -(1.0 - alpha)], dtype=float)
    zi = np.array([(1.0 - alpha) * x[0]], dtype=float)
    y, _ = lfilter(b, a, x, zi=zi)
    return np.asarray(y, dtype=float)
