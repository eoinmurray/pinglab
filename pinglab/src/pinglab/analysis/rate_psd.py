
import numpy as np
from scipy.signal import welch


def rate_psd(
    rate_hz: np.ndarray,
    dt_ms: float,
    nperseg: int | None = None,
):
    """
    rate_hz: population firing rate time series (Hz)
    dt_ms: bin width used to make rate
    returns freqs (Hz), psd (Hz^2/Hz)
    """
    fs = 1000.0 / dt_ms

    # optional: remove DC component so PSD peak is clearer
    rate_centered = rate_hz - np.mean(rate_hz)

    f, Pxx = welch(
        rate_centered,
        fs=fs,
        nperseg=nperseg or min(1024, len(rate_centered)),
    )

    return f, Pxx