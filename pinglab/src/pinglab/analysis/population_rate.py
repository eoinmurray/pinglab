"""Population firing rate computation with time binning."""

import numpy as np
from pinglab.types import Spikes


def population_rate(
    spikes: Spikes,
    T_ms: float,
    dt_ms: float,
    pop: str = "E",
    N_E: int | None = None,
    N_I: int | None = None,
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
        (t_ms, rate_hz): Bin centers (ms) and firing rates (Hz)
    """
    # choose neuron indices
    if pop == "E":
        assert N_E is not None
        mask = spikes.ids < N_E
        N_pop = N_E
    elif pop == "I":
        assert N_E is not None and N_I is not None
        mask = spikes.ids >= N_E
        N_pop = N_I
    elif pop == "all":
        mask = np.ones_like(spikes.ids, dtype=bool)
        assert N_E is not None and N_I is not None
        N_pop = N_E + N_I
    else:
        raise ValueError("pop must be 'E', 'I', or 'all'")

    spike_times = spikes.times[mask]  # in ms

    # bin edges
    n_bins = int(np.ceil(T_ms / dt_ms))
    edges = np.linspace(0.0, T_ms, n_bins + 1)

    counts, _ = np.histogram(spike_times, bins=edges)

    # convert counts → firing rate (Hz)
    dt_s = dt_ms / 1000.0
    rate_hz = counts / (N_pop * dt_s)

    # bin centers for plotting
    t_ms = 0.5 * (edges[:-1] + edges[1:])

    return t_ms, rate_hz
