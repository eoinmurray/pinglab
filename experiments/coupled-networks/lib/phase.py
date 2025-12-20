from __future__ import annotations

import numpy as np


def smooth_signal(signal: np.ndarray, *, dt_ms: float, window_ms: float) -> np.ndarray:
    if window_ms <= 0.0:
        return signal
    window = max(1, int(round(window_ms / dt_ms)))
    if window <= 1:
        return signal
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(signal, kernel, mode="same")


def analytic_signal(signal: np.ndarray) -> np.ndarray:
    """
    Return the analytic signal using an FFT-based Hilbert transform.
    """
    x = np.asarray(signal, dtype=np.float32)
    n = x.size
    if n == 0:
        return x.astype(np.complex64)

    X = np.fft.fft(x)
    h = np.zeros(n, dtype=np.float32)
    if n % 2 == 0:
        h[0] = 1.0
        h[n // 2] = 1.0
        h[1 : n // 2] = 2.0
    else:
        h[0] = 1.0
        h[1 : (n + 1) // 2] = 2.0
    return np.fft.ifft(X * h)


def estimate_phase(
    signal: np.ndarray,
    *,
    dt_ms: float,
    smoothing_ms: float,
) -> np.ndarray:
    """
    Estimate instantaneous phase from a real-valued signal.
    """
    if signal.size == 0:
        return np.array([], dtype=np.float32)
    x = signal.astype(np.float32)
    x = x - float(np.mean(x))
    x = smooth_signal(x, dt_ms=dt_ms, window_ms=smoothing_ms)
    analytic = analytic_signal(x)
    return np.angle(analytic).astype(np.float32)
