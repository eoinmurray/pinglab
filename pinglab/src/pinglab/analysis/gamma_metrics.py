
from pathlib import Path
import numpy as np
from pinglab.plots.styles import save_both, figsize
from pinglab.types import Spikes
import matplotlib.pyplot as plt


def gamma_metrics(
    spikes: Spikes,
    T: float,
    fs: float,
    fmin: float = 20.0,
    fmax: float = 100.0,
    data_path: Path | None = None,
) -> tuple[float | None, float | None, float | None]:
    """
    Compute gamma-band metrics from population firing rate PSD.

    Parameters
    ----------
    spikes : Spikes
        .times in ms, .ids arbitrary.
    T : float
        Total simulation time in ms.
    fs : float
        Sampling frequency for rate signal in Hz.
    fmin, fmax : float
        Gamma band limits in Hz.
    plot : bool
        If True, plot the PSD (0–fs/2) and highlight gamma band / peak.
    ax : matplotlib.axes.Axes | None
        Optional axis to plot into. If None and plot=True, a new figure is created.

    Returns
    -------
    gamma_peak_freq : float | None
        Frequency of the largest peak in [fmin, fmax], or None if no peak.
    gamma_peak_power : float | None
        Power at that peak, or None.
    gamma_Q : float | None
        Q-factor = f_peak / bandwidth_(-3dB), or None if cannot estimate.
    """
    times = spikes.times

    # Bin spikes into population rate
    bin_ms = 1000.0 / fs
    n_bins = int(np.round(T / bin_ms))
    if n_bins < 4:
        return None, None, None

    counts, edges = np.histogram(times, bins=n_bins, range=(0.0, T))
    rate = counts * (1000.0 / bin_ms)  # spikes/s across population

    # Remove DC component
    x = rate - np.mean(rate)

    # FFT-based PSD
    n = x.size
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)   # Hz
    fft_vals = np.fft.rfft(x)
    psd = (np.abs(fft_vals) ** 2) / (n * fs)  # scaling not crucial

    # Restrict to gamma band
    band_mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(band_mask) and data_path is not None:
        print("Warning: No frequencies in the specified gamma band. Not plotting.")

    freqs_band = freqs[band_mask]
    psd_band = psd[band_mask]

    # Peak in gamma band
    peak_idx = int(np.argmax(psd_band))
    gamma_peak_freq = float(freqs_band[peak_idx])
    gamma_peak_power = float(psd_band[peak_idx])

    # -3 dB bandwidth (~half power)
    gamma_Q: float | None
    if gamma_peak_power <= 0.0:
        gamma_Q = None
    else:
        half_power = gamma_peak_power / 2.0

        # search left
        left_idx = peak_idx
        while left_idx > 0 and psd_band[left_idx] > half_power:
            left_idx -= 1
        # search right
        right_idx = peak_idx
        while right_idx < len(psd_band) - 1 and psd_band[right_idx] > half_power:
            right_idx += 1

        if left_idx == peak_idx or right_idx == peak_idx:
            gamma_Q = None
        else:
            bandwidth = float(freqs_band[right_idx] - freqs_band[left_idx])
            gamma_Q = gamma_peak_freq / bandwidth if bandwidth > 0 else None

    # Optional plotting
    if data_path is not None:
        def plot_fn():
            _, ax = plt.subplots(figsize=figsize)
            ax.plot(freqs, psd, lw=1.0)
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("PSD (a.u.)")
            ax.set_title(f"Population rate PSD (gamma peak ≈ {gamma_peak_freq:.1f} Hz)")
            ax.grid(True)
            ax.set_xlim((0, 200))
            plt.tight_layout()

        save_both(data_path, plot_fn)
        

    return gamma_peak_freq, gamma_peak_power, gamma_Q
