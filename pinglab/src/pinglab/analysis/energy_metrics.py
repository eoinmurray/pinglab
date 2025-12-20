
import numpy as np
from pinglab.types import Spikes


def energy_metrics(
    g_e: np.ndarray,
    g_i: np.ndarray,
    spikes: Spikes,
    dt: float,
    N_E: int,
    N_I: int,
) -> tuple[float, float, float, float]:
    """
    Compute energy proxy metrics from conductances and spikes.

    Parameters
    ----------
    g_e : np.ndarray
        Excitatory conductance array, shape (T, N).
    g_i : np.ndarray
        Inhibitory conductance array, shape (T, N).
    spikes : Spikes
        Spike times and neuron IDs.
    dt : float
        Timestep in ms.
    N_E : int
        Number of excitatory neurons.
    N_I : int
        Number of inhibitory neurons.

    Returns
    -------
    energy_conductance : float
        Time-integrated total conductance: ∫(g_e + g_i) dt, summed over neurons.
    energy_spikes : float
        Total spike count (E + I).
    energy_total : float
        Combined energy proxy: conductance integral + spike count.
    energy_per_spike : float
        Efficiency: total spikes / energy_total (spikes per unit energy).
    """
    g_e = np.asarray(g_e, dtype=float)
    g_i = np.asarray(g_i, dtype=float)

    # Sum conductances over neurons, then integrate over time
    # Handle both 2D (T, N) and 1D (T,) arrays (population means)
    if g_e.ndim == 2:
        g_total_per_step = np.sum(g_e, axis=1) + np.sum(g_i, axis=1)
    else:
        # 1D population mean traces - scale by population size for energy estimate
        g_total_per_step = g_e + g_i
    energy_conductance = float(np.sum(g_total_per_step) * dt)

    # Count spikes
    n_spikes_E = int(np.sum(spikes.ids < N_E))
    n_spikes_I = int(np.sum(spikes.ids >= N_E))
    energy_spikes = float(n_spikes_E + n_spikes_I)

    # Combined proxy (conductance integral + spike count)
    energy_total = energy_conductance + energy_spikes

    # Efficiency
    energy_per_spike = energy_spikes / energy_total if energy_total > 0 else 0.0

    return energy_conductance, energy_spikes, energy_total, energy_per_spike
