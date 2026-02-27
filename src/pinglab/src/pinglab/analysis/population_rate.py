"""Population firing rate computation with time binning."""

import numpy as np
from pinglab.backends.types import Spikes


def population_rate(
    spikes: Spikes,
    T_ms: float,
    dt_ms: float,
    pop: str = "E",
    N_E: int | None = None,
    N_I: int | None = None,
    smooth_sigma_ms: float | None = None,
    smooth_bin_ms: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute time-binned population firing rate.

    Parameters:
        spikes: Spike data with times and neuron IDs
        T_ms: Total simulation time in milliseconds
        dt_ms: Bin width in milliseconds
        pop: Population to analyze ('E', 'I', or 'all')
        N_E: Number of excitatory neurons
        N_I: Number of inhibitory neurons

    Returns:
        (t_ms, rate_hz): Bin centers (ms) and firing rates (Hz). If smoothing is
        enabled, the returned rate is smoothed with a Gaussian kernel.
    """
    # Validate parameters
    if T_ms <= 0:
        raise ValueError(f"T_ms must be positive, got {T_ms}")
    if dt_ms <= 0:
        raise ValueError(f"dt_ms must be positive, got {dt_ms}")

    # Choose neuron indices
    if pop == "E":
        if N_E is None:
            raise ValueError("N_E is required when pop='E'")
        mask = spikes.ids < N_E
        N_pop = N_E
    elif pop == "I":
        if N_E is None or N_I is None:
            raise ValueError("N_E and N_I are required when pop='I'")
        mask = spikes.ids >= N_E
        N_pop = N_I
    elif pop == "all":
        if N_E is None or N_I is None:
            raise ValueError("N_E and N_I are required when pop='all'")
        mask = np.ones_like(spikes.ids, dtype=bool)
        N_pop = N_E + N_I
    else:
        raise ValueError("pop must be 'E', 'I', or 'all'")

    spike_times = spikes.times[mask]  # in ms

    # bin edges
    n_bins = int(np.ceil(T_ms / dt_ms))
    edges = np.linspace(0.0, T_ms, n_bins + 1)

    counts, _ = np.histogram(spike_times, bins=edges)

    # bin centers for plotting
    t_ms = 0.5 * (edges[:-1] + edges[1:])

    if N_pop <= 0:
        rate_hz = np.zeros_like(t_ms, dtype=float)
    else:
        # convert counts → firing rate (Hz)
        dt_s = dt_ms / 1000.0
        rate_hz = counts / (N_pop * dt_s)

    if smooth_sigma_ms is None:
        return t_ms, rate_hz

    if smooth_sigma_ms <= 0:
        raise ValueError(f"smooth_sigma_ms must be positive, got {smooth_sigma_ms}")

    kernel_dt_ms = smooth_bin_ms if smooth_bin_ms is not None else dt_ms
    if kernel_dt_ms <= 0:
        raise ValueError(f"smooth_bin_ms must be positive, got {kernel_dt_ms}")

    kernel_radius_ms = max(1.0, 4.0 * smooth_sigma_ms)
    kernel_half = int(np.ceil(kernel_radius_ms / kernel_dt_ms))
    kernel_times = np.arange(-kernel_half, kernel_half + 1) * kernel_dt_ms
    kernel = np.exp(-0.5 * (kernel_times / smooth_sigma_ms) ** 2)
    kernel = kernel / np.sum(kernel)

    if not np.isclose(kernel_dt_ms, dt_ms):
        resample_t = np.arange(0.0, T_ms + kernel_dt_ms, kernel_dt_ms)
        resample_rate = np.interp(resample_t, t_ms, rate_hz, left=0.0, right=0.0)
        smoothed = np.convolve(resample_rate, kernel, mode="same")
        smoothed_rate = np.interp(t_ms, resample_t, smoothed, left=0.0, right=0.0)
    else:
        smoothed_rate = np.convolve(rate_hz, kernel, mode="same")

    return t_ms, smoothed_rate
