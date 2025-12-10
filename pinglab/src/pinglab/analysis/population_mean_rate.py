
import numpy as np
from pinglab.types import Spikes


def population_mean_rate(spikes: Spikes, T: float, N_E: int, N_I: int) -> tuple[float, float]:
    """
    Compute mean firing rate for E and I populations.
    
    Parameters
    ----------
    spikes : Spikes
        Object with .times and .ids arrays.
    T : float
        Total simulation time in ms.
    N_E : int
        Number of excitatory neurons.
    N_I : int
        Number of inhibitory neurons.
    
    Returns
    -------
    mean_rate_E : float
        Mean excitatory firing rate (Hz).
    mean_rate_I : float
        Mean inhibitory firing rate (Hz).
    """
    ids = spikes.ids

    # Excitatory neurons are assumed to be 0..N_E-1
    # Inhibitory neurons are N_E..N_E+N_I-1
    spikes_E = np.sum(ids < N_E)
    spikes_I = np.sum(ids >= N_E)

    # Convert from spikes per ms to spikes per second
    sec = T / 1000.0

    mean_rate_E = spikes_E / (N_E * sec)
    mean_rate_I = spikes_I / (N_I * sec)

    return mean_rate_E, mean_rate_I
