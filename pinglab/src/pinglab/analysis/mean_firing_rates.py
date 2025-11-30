"""Mean firing rate computation for E/I populations."""

from pinglab.types import Spikes


def mean_firing_rates(spikes: Spikes, N_E: int, N_I: int) -> tuple[float, float]:
    """
    Compute mean firing rates for E and I populations.

    Parameters:
        spikes: Spike data with times and neuron IDs
        N_E: Number of excitatory neurons
        N_I: Number of inhibitory neurons

    Returns:
        (E_rate, I_rate): Mean firing rates in Hz for each population
    """
    assert spikes.times.shape == spikes.ids.shape

    # Handle empty spike arrays
    if len(spikes.times) == 0:
        return 0.0, 0.0

    mask_E = spikes.ids < N_E
    nE = mask_E.sum()
    nI = (~mask_E).sum()

    T_s = (spikes.times.max() - spikes.times.min()) / 1000

    # Guard against division by zero (single spike or all spikes at same time)
    if T_s <= 0:
        return 0.0, 0.0

    E_rate = nE / (N_E * T_s) if N_E > 0 else 0.0
    I_rate = nI / (N_I * T_s) if N_I > 0 else 0.0

    return E_rate, I_rate
