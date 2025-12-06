"""Inter-spike interval coefficient of variation (CV) analysis."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from pinglab.types import Spikes


def isi_cv_per_neuron(
    spikes: Spikes,
    neuron_ids: np.ndarray | None = None,
    min_spikes: int = 2,
    ddof: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute ISI CV per neuron.

    Parameters
    ----------
    spikes : Spikes
        Spikes object with `times` (float array) and `ids` (int array).
    neuron_ids : array-like or None, optional
        Neuron IDs to include. If None, all unique IDs in `spikes.ids` are used.
    min_spikes : int, optional
        Minimum number of spikes required for a neuron to be included.
        Neurons with fewer spikes are ignored.
    ddof : int, optional
        Delta degrees of freedom for the ISI standard deviation.
        Use 0 for population std, 1 for sample std.

    Returns
    -------
    neuron_ids_out : np.ndarray (int)
        IDs of neurons with a valid CV estimate (size >= min_spikes).
    cv : np.ndarray (float)
        CV values for each neuron in `neuron_ids_out`, same length.
    """
    times = np.asarray(spikes.times, dtype=float)
    ids = np.asarray(spikes.ids, dtype=int)

    if neuron_ids is None:
        neuron_ids = np.unique(ids)
    else:
        neuron_ids = np.asarray(neuron_ids, dtype=int)

    cvs: list[float] = []
    kept_ids: list[int] = []

    for nid in neuron_ids:
        mask = ids == nid
        t = times[mask]

        # Not enough spikes to define ISIs
        if t.size < min_spikes:
            continue

        # Ensure sorted by time (paranoia in case upstream ever changes)
        t = np.sort(t)
        isi = np.diff(t)

        # No ISIs => skip
        if isi.size == 0:
            continue

        mean_isi = float(np.mean(isi))
        if mean_isi <= 0.0:
            # Degenerate / invalid timing
            continue

        std_isi = float(np.std(isi, ddof=ddof))
        cvs.append(std_isi / mean_isi)
        kept_ids.append(int(nid))

    if not cvs:
        return np.array([], dtype=int), np.array([], dtype=float)

    return np.array(kept_ids, dtype=int), np.array(cvs, dtype=float)


def population_isi_cv(
    spikes: Spikes,
    N_E: int,
    N_I: int,
    min_spikes: int = 2,
    ddof: int = 0,
) -> tuple[float | None, float | None]:
    """
    Compute population ISI CV for E and I as the median across neurons.

    Neurons with fewer than `min_spikes` spikes (or degenerate ISIs)
    are ignored. If no valid neurons exist in a population, returns None.

    Parameters
    ----------
    spikes : Spikes
        Spikes object with `times` and `ids`.
    N_E : int
        Number of excitatory neurons. Assumes E IDs are [0 .. N_E-1].
    N_I : int
        Number of inhibitory neurons. Assumes I IDs are [N_E .. N_E+N_I-1].
    min_spikes : int, optional
        Minimum number of spikes for a neuron to be considered.
    ddof : int, optional
        Delta degrees of freedom for ISI std (passed to `isi_cv_per_neuron`).

    Returns
    -------
    cv_E : float or None
        Median ISI CV across excitatory neurons, or None if no valid E neurons.
    cv_I : float or None
        Median ISI CV across inhibitory neurons, or None if no valid I neurons.
    """
    # Assumes contiguous indexing: E = [0..N_E-1], I = [N_E..N_E+N_I-1]
    all_E_ids = np.arange(0, N_E, dtype=int)
    all_I_ids = np.arange(N_E, N_E + N_I, dtype=int)

    _, cv_E_neurons = isi_cv_per_neuron(
        spikes,
        neuron_ids=all_E_ids,
        min_spikes=min_spikes,
        ddof=ddof,
    )
    _, cv_I_neurons = isi_cv_per_neuron(
        spikes,
        neuron_ids=all_I_ids,
        min_spikes=min_spikes,
        ddof=ddof,
    )

    cv_E = float(np.median(cv_E_neurons)) if cv_E_neurons.size > 0 else None
    cv_I = float(np.median(cv_I_neurons)) if cv_I_neurons.size > 0 else None

    return cv_E, cv_I
