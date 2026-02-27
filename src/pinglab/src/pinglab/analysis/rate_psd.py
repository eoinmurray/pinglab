"""Power spectral density analysis of population firing rates."""

import numpy as np
from scipy.signal import welch


def rate_psd(
    rate_hz: np.ndarray,
    dt_ms: float,
    nperseg: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute power spectral density of population firing rate.

    Parameters:
        rate_hz: Population firing rate time series (Hz)
        dt_ms: Bin width used to compute rate (ms)
        nperseg: Segment length for Welch's method (default: min(1024, len(rate)))

    Returns:
        (freqs, psd): Frequencies (Hz) and power spectral density (Hz²/Hz)
    """
    fs = 1000.0 / dt_ms

    # optional: remove DC component so PSD peak is clearer
    rate_centered = rate_hz - np.mean(rate_hz)

    # Preserve low-frequency resolution by default.
    # Example: dt=0.1 ms -> fs=10 kHz; nperseg=1024 gives ~9.77 Hz bins and hides <10 Hz content.
    if nperseg is None:
        default_nperseg = min(len(rate_centered), max(256, int(fs)))
    else:
        default_nperseg = int(nperseg)
    f, Pxx = welch(
        rate_centered,
        fs=fs,
        nperseg=default_nperseg,
    )

    return f, Pxx
