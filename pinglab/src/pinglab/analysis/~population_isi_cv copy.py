"""Inter-spike interval coefficient of variation (CV) analysis."""

from __future__ import annotations
import numpy as np

from .isi_cv_per_neuron import isi_cv_per_neuron
from pinglab.types import Spikes

def population_isi_cv(
    spikes: Spikes,
    N_E: int,
    N_I: int,
    min_spikes: int = 20,
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
