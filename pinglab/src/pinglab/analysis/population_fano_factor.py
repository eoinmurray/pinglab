
import numpy as np
from pinglab.types import Spikes


def population_fano_factor(
    spikes: Spikes,
    T: float,
    N_E: int,
    N_I: int,
    window_ms: float,
) -> tuple[float, float]:
    """
    Compute Fano factor (variance/mean of spike counts) for E and I populations.

    Parameters
    ----------
    spikes : Spikes
        .times in ms, .ids in [0 .. N_E+N_I-1]
    T : float
        Total simulation time in ms.
    N_E, N_I : int
        Population sizes.
    window_ms : float
        Bin width for spike counting.

    Returns
    -------
    fano_E : float
    fano_I : float
    """
    times = spikes.times
    ids = spikes.ids

    n_bins = int(np.ceil(T / window_ms))
    if n_bins < 2:
        return 0.0, 0.0

    # Prepare bins
    edges = np.linspace(0.0, T, n_bins + 1)

    # E and I masks
    mask_E = ids < N_E
    mask_I = ids >= N_E

    # Spike times
    tE = times[mask_E]
    tI = times[mask_I]

    # Histogram spike counts
    counts_E, _ = np.histogram(tE, bins=edges)
    counts_I, _ = np.histogram(tI, bins=edges)

    # Compute means and variances
    mean_E = float(np.mean(counts_E))
    var_E  = float(np.var(counts_E))
    mean_I = float(np.mean(counts_I))
    var_I  = float(np.var(counts_I))

    # Fano factors (avoid 0-division)
    fano_E = var_E / mean_E if mean_E > 0 else 0.0
    fano_I = var_I / mean_I if mean_I > 0 else 0.0

    return fano_E, fano_I
