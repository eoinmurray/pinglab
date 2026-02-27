"""Decode-path analysis helpers for envelope recovery diagnostics."""

import numpy as np


def minmax_normalize(values: np.ndarray) -> np.ndarray:
    """Normalize values to [0, 1] using min-max scaling."""
    x = np.asarray(values, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"values must be 1D, got shape {x.shape}")
    if x.size == 0:
        return np.array([], dtype=float)
    lo = float(np.min(x))
    hi = float(np.max(x))
    span = hi - lo
    if not np.isfinite(span) or span <= 1e-12:
        return np.zeros_like(x, dtype=float)
    return (x - lo) / span


def pearson_corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Pearson correlation with shape and finite-value guards."""
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError(f"inputs must be 1D, got {x.shape} and {y.shape}")
    n = min(x.size, y.size)
    if n < 3:
        return 0.0
    xx = x[:n]
    yy = y[:n]
    mx = float(np.mean(xx))
    my = float(np.mean(yy))
    dx = xx - mx
    dy = yy - my
    denom = float(np.sqrt(np.sum(dx * dx) * np.sum(dy * dy)))
    if not np.isfinite(denom) or denom <= 1e-12:
        return 0.0
    return float(np.sum(dx * dy) / denom)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    """Compute root-mean-square error over the common prefix length."""
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError(f"inputs must be 1D, got {x.shape} and {y.shape}")
    n = min(x.size, y.size)
    if n == 0:
        return 0.0
    d = x[:n] - y[:n]
    return float(np.sqrt(np.mean(d * d)))


def envelope_rate_hz(
    t_ms: np.ndarray,
    *,
    lambda0_hz: float,
    mod_depth: float,
    envelope_freq_hz: float,
    phase_rad: float,
) -> np.ndarray:
    """Compute clamped inhomogeneous-Poisson envelope rate over t_ms."""
    t = np.asarray(t_ms, dtype=float)
    if t.ndim != 1:
        raise ValueError(f"t_ms must be 1D, got shape {t.shape}")
    rate = lambda0_hz * (
        1.0 + mod_depth * np.sin((2.0 * np.pi * envelope_freq_hz * t / 1000.0) + phase_rad)
    )
    return np.maximum(0.0, rate)


def decode_fit_metrics(
    decoded: np.ndarray,
    envelope: np.ndarray,
    *,
    normalize: bool = True,
) -> tuple[float, float]:
    """Return (corr, rmse) between decoded rate and target envelope."""
    x = np.asarray(decoded, dtype=float)
    y = np.asarray(envelope, dtype=float)
    if normalize:
        x = minmax_normalize(x)
        y = minmax_normalize(y)
    return pearson_corrcoef(x, y), rmse(x, y)
