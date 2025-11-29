
import numpy as np
from pinglab.types import Spikes


def calculate_population_rate(
    spikes: Spikes,
    T_ms: float,
    dt_ms: float,
    pop: str = "E",
    N_E: int | None = None,
    N_I: int | None = None,
):
    """
    pop: 'E', 'I', or 'all'
    returns t, rate in Hz
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
