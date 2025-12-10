
from pinglab.types import Spikes


import numpy as np
from pinglab.types import Spikes


def pairwise_spike_count_corr(
    spikes: Spikes,
    T: float,
    bin_ms: float,
    N_E: int,
    N_I: int,
) -> tuple[float, float, float]:
    """
    Compute mean pairwise spike-count correlations for EE, II, and EI pairs.

    Parameters
    ----------
    spikes : Spikes
        Object with .times (ms) and .ids (0..N_E+N_I-1).
    T : float
        Total simulation time in ms.
    bin_ms : float
        Bin width in ms for spike-count correlation.
    N_E : int
        Number of excitatory neurons.
    N_I : int
        Number of inhibitory neurons.

    Returns
    -------
    corr_EE_mean : float
        Mean pairwise spike-count correlation between excitatory neurons.
    corr_II_mean : float
        Mean pairwise spike-count correlation between inhibitory neurons.
    corr_EI_mean : float
        Mean pairwise spike-count correlation between excitatory and inhibitory neurons.
    """
    times = spikes.times
    ids = spikes.ids

    N = N_E + N_I
    n_bins = int(np.ceil(T / bin_ms))

    # Spike count matrix: shape (N_neurons, n_bins)
    counts = np.zeros((N, n_bins), dtype=np.float32)

    # Only consider spikes within [0, T)
    mask = (times >= 0.0) & (times < T)
    times = times[mask]
    ids = ids[mask]

    # Bin indices for each spike
    bin_idx = (times / bin_ms).astype(int)
    # Guard against any bin_idx == n_bins due to numerical edge
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    # Accumulate counts: counts[neuron_id, bin_idx] += 1
    np.add.at(counts, (ids, bin_idx), 1.0)

    # Correlation matrix across neurons
    with np.errstate(invalid="ignore", divide="ignore"):
        C = np.corrcoef(counts)

    def mean_block_corr(block: np.ndarray, upper_triangular: bool = True) -> float:
        if block.size == 0:
            return 0.0
        if upper_triangular and block.shape[0] > 1:
            mask = np.triu(np.ones(block.shape, dtype=bool), k=1)
            vals = block[mask]
        else:
            vals = block.ravel()
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return 0.0
        return float(vals.mean())

    # EE block: neurons 0..N_E-1
    EE_block = C[0:N_E, 0:N_E]
    corr_EE_mean = mean_block_corr(EE_block, upper_triangular=True)

    # II block: neurons N_E..N_E+N_I-1
    II_block = C[N_E:N_E + N_I, N_E:N_E + N_I]
    corr_II_mean = mean_block_corr(II_block, upper_triangular=True)

    # EI block: E rows, I columns
    EI_block = C[0:N_E, N_E:N_E + N_I]
    corr_EI_mean = mean_block_corr(EI_block, upper_triangular=False)

    return corr_EE_mean, corr_II_mean, corr_EI_mean
