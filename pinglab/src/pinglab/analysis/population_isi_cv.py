import numpy as np
from typing import Tuple
from pinglab.types import Spikes  # or wherever your Spikes lives


def isi_cv_per_neuron(
    spikes: Spikes,
    neuron_ids: np.ndarray | None = None,
    min_spikes: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute ISI CV per neuron.

    Returns
    -------
    neuron_ids_out : array of neuron indices (only those with >= min_spikes)
    cv : array of CV values, same length as neuron_ids_out
    """
    times = spikes.times
    ids = spikes.ids

    if neuron_ids is None:
        neuron_ids = np.unique(ids)

    cvs = []
    kept_ids = []

    for nid in neuron_ids:
        mask = ids == nid
        t = times[mask]
        if t.size < min_spikes:
            continue

        # ensure sorted by time (should already be, but be paranoid)
        t = np.sort(t)
        isi = np.diff(t)
        if isi.size == 0:
            continue

        mean_isi = np.mean(isi)
        std_isi = np.std(isi)

        if mean_isi <= 0:
            continue

        cvs.append(std_isi / mean_isi)
        kept_ids.append(nid)

    if len(cvs) == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    return np.array(kept_ids, dtype=int), np.array(cvs, dtype=float)


def population_isi_cv(
    spikes: Spikes,
    N_E: int,
    N_I: int,
    min_spikes: int = 2,
) -> Tuple[float | None, float | None]:
    """
    Compute population ISI CV for E and I as the **median** across neurons.
    Silent/near-silent neurons (spikes < min_spikes) are ignored.

    Returns
    -------
    cv_E, cv_I : floats or None if no valid neurons in that population
    """
    all_E_ids = np.arange(0, N_E)
    all_I_ids = np.arange(N_E, N_E + N_I)

    _, cv_E_neurons = isi_cv_per_neuron(spikes, all_E_ids, min_spikes=min_spikes)
    _, cv_I_neurons = isi_cv_per_neuron(spikes, all_I_ids, min_spikes=min_spikes)

    cv_E = float(np.median(cv_E_neurons)) if cv_E_neurons.size > 0 else None
    cv_I = float(np.median(cv_I_neurons)) if cv_I_neurons.size > 0 else None

    return cv_E, cv_I
