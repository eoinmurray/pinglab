
import numpy as np
from pinglab.types import Spikes


def population_isi_cv(
    spikes: Spikes,
    neuron_ids: np.ndarray | None = None,
    min_spikes: int = 2,
) -> tuple[float, np.ndarray]:
    """
    Compute population ISI CV and per-neuron CVs.

    Returns
    -------
    cv_population : float
        CV of all ISIs pooled across the selected neurons.
    cv_per_neuron : np.ndarray
        1D array of CVs for each neuron with >= min_spikes.
    """
    times = spikes.times
    ids = spikes.ids

    # choose neurons
    if neuron_ids is None:
        neuron_ids = np.unique(ids)

    per_neuron = []

    all_isis = []   # pooled for population CV

    for nid in neuron_ids:
        t = times[ids == nid]
        if t.size < min_spikes:
            continue

        t = np.sort(t)
        isi = np.diff(t)
        if isi.size == 0:
            continue

        mu = np.mean(isi)
        sigma = np.std(isi)

        per_neuron.append(sigma / mu if mu > 0 else np.nan)
        all_isis.append(isi)

    # If no valid neurons, return zeros
    if not per_neuron:
        return 0.0, np.array([])

    cv_per_neuron = np.array(per_neuron)

    # population CV = CV of pooled ISIs
    pooled = np.concatenate(all_isis)
    pop_mu = np.mean(pooled)
    pop_sigma = np.std(pooled)
    cv_population = float(pop_sigma / pop_mu if pop_mu > 0 else 0.0)

    return cv_population, cv_per_neuron
