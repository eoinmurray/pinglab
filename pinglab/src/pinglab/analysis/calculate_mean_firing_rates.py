from pinglab.types import Spikes

def calculate_mean_firing_rates(spikes: Spikes, N_E: int, N_I: int):
    assert spikes.times.shape == spikes.ids.shape

    mask_E = spikes.ids < N_E
    nE = mask_E.sum() / N_E
    nI = (~mask_E).sum() / N_I

    T_s = (spikes.times.max() - spikes.times.min()) / 1000

    E_rate = nE / (N_E * T_s)
    I_rate = nI / (N_I * T_s)

    return E_rate, I_rate
