"""Mean firing rate computation for E/I populations from ISIs."""

import numpy as np

from pinglab.backends.types import Spikes


def _mean_rate_from_isi(
    times: np.ndarray,
    ids: np.ndarray,
    neuron_ids: np.ndarray,
    population_size: int,
    min_spikes: int,
) -> float:
    if population_size <= 0:
        return 0.0

    rate_sum = 0.0
    for nid in neuron_ids:
        t = times[ids == nid]
        if t.size < min_spikes:
            continue
        t = np.sort(t)
        isi = np.diff(t)
        if isi.size == 0:
            continue
        mean_isi = float(np.mean(isi))
        if mean_isi <= 0.0:
            continue
        rate_sum += 1000.0 / mean_isi

    return rate_sum / population_size


def mean_firing_rates(
    spikes: Spikes,
    N_E: int,
    N_I: int,
    min_spikes: int = 2,
) -> tuple[float, float]:
    """
    Compute mean firing rates for E and I populations from ISIs.

    Each neuron's rate is computed as 1 / mean(ISI), and the population
    mean includes silent neurons as 0 Hz.

    Parameters:
        spikes: Spike data with times and neuron IDs
        N_E: Number of excitatory neurons
        N_I: Number of inhibitory neurons
        min_spikes: Minimum spikes per neuron to include ISIs

    Returns:
        (E_rate, I_rate): Mean firing rates in Hz for each population
    """
    assert spikes.times.shape == spikes.ids.shape

    if spikes.times.size == 0:
        return 0.0, 0.0

    ids = spikes.ids
    times = spikes.times

    e_ids = np.arange(N_E)
    i_ids = np.arange(N_E, N_E + N_I)

    e_rate = _mean_rate_from_isi(times, ids, e_ids, N_E, min_spikes)
    i_rate = _mean_rate_from_isi(times, ids, i_ids, N_I, min_spikes)

    return e_rate, i_rate
