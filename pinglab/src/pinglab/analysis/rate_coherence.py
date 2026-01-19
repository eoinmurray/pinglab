import numpy as np
from pinglab.types import Spikes


def rate_coherence(
    spikes: Spikes,
    T: float,
    bin_ms: float,
    N_E: int,
    N_I: int,
    pop: str = "all",
    max_pairs: int | None = None,
    seed: int | None = None,
) -> float:
    """
    Compute average pairwise Pearson correlation of binned firing rates.

    Parameters
    ----------
    spikes : Spikes
        .times in ms, .ids in [0 .. N_E+N_I-1]
    T : float
        Total simulation time in ms.
    bin_ms : float
        Width of time bins in ms.
    N_E : int
        Number of excitatory neurons.
    N_I : int
        Number of inhibitory neurons.
    pop : str
        One of {"all", "E", "I"} selecting which population to include.
    max_pairs : int | None
        If set, randomly subsample at most this many neuron pairs.
    seed : int | None
        RNG seed for pair subsampling.

    Returns
    -------
    float
        Mean pairwise rate correlation. Returns 0.0 if insufficient variance
        or fewer than two neurons.
    """
    if bin_ms <= 0:
        raise ValueError("bin_ms must be positive.")

    if pop not in {"all", "E", "I"}:
        raise ValueError("pop must be one of {'all', 'E', 'I'}.")

    N = N_E + N_I
    if N < 2:
        return 0.0

    if pop == "E":
        neuron_ids = np.arange(0, N_E, dtype=int)
    elif pop == "I":
        neuron_ids = np.arange(N_E, N, dtype=int)
    else:
        neuron_ids = np.arange(0, N, dtype=int)

    if neuron_ids.size < 2:
        return 0.0

    n_bins = int(np.ceil(T / bin_ms))
    if n_bins < 2:
        return 0.0

    edges = np.linspace(0.0, T, n_bins + 1)
    bin_sec = bin_ms / 1000.0

    rates = np.zeros((neuron_ids.size, n_bins), dtype=float)
    for idx, neuron_id in enumerate(neuron_ids):
        times = spikes.times[spikes.ids == neuron_id]
        counts, _ = np.histogram(times, bins=edges)
        rates[idx] = counts / bin_sec

    n_neurons = rates.shape[0]
    if n_neurons < 2:
        return 0.0

    pairs = []
    if max_pairs is None or max_pairs >= (n_neurons * (n_neurons - 1)) // 2:
        for i in range(n_neurons):
            for j in range(i + 1, n_neurons):
                pairs.append((i, j))
    else:
        rng = np.random.RandomState(seed)
        all_pairs = [
            (i, j) for i in range(n_neurons) for j in range(i + 1, n_neurons)
        ]
        rng.shuffle(all_pairs)
        pairs = all_pairs[:max_pairs]

    corr_vals = []
    for i, j in pairs:
        x = rates[i]
        y = rates[j]
        std_x = np.std(x)
        std_y = np.std(y)
        if std_x == 0.0 or std_y == 0.0:
            continue
        corr = float(np.corrcoef(x, y)[0, 1])
        corr_vals.append(corr)

    if not corr_vals:
        return 0.0

    return float(np.mean(corr_vals))
