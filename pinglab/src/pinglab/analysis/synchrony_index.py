
import numpy as np
from pinglab.types import Spikes


def synchrony_index(
    spikes: Spikes,
    T: float,
    bin_ms: float,
    N_E: int,
    N_I: int,
) -> float:
    """
    Compute a scalar synchrony index from population spike counts.

    Definition:
      SI = Var(population_rate) / Mean(population_rate)

    Population_rate is the spike count per bin pooled across all neurons.

    Parameters
    ----------
    spikes : Spikes
        .times in ms, .ids in [0 .. N_E+N_I-1]
    T : float
        Total simulation time in ms.
    bin_ms : float
        Width of time bins in ms.

    Returns
    -------
    synchrony_index : float
    """
    times = spikes.times
    N = N_E + N_I

    n_bins = int(np.ceil(T / bin_ms))
    if n_bins < 2:
        return 0.0

    # Bin edges
    edges = np.linspace(0.0, T, n_bins + 1)

    # Histogram spike counts (all neurons pooled)
    counts, _ = np.histogram(times, bins=edges)

    # Convert to rate per neuron (Hz) — scale doesn't change SI structure
    bin_sec = bin_ms / 1000.0
    rate = counts / (N * bin_sec)

    mean_r = float(np.mean(rate))
    var_r  = float(np.var(rate))

    return var_r / mean_r if mean_r > 0 else 0.0
