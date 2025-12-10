"""Inter-spike interval coefficient of variation (CV) analysis."""

from __future__ import annotations

import numpy as np
from pinglab.types import Spikes


def isi_cv_per_neuron(
    spikes: Spikes,
    neuron_ids: np.ndarray | None = None,
    min_spikes: int = 20,  # default: require at least 2 ISIs
    ddof: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute ISI CV per neuron.

    Returns only neurons with:
    - at least `min_spikes` spikes
    - at least `ddof + 1` ISIs (for std to be defined)
    - finite, positive mean ISI
    - finite CV

    Returns
    -------
    kept_ids : np.ndarray
        Neuron IDs for which CV was computed.
    cvs : np.ndarray
        ISI CV values for the kept neurons.
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

        # Not enough spikes to define a sensible ISI distribution
        if t.size < min_spikes:
            continue

        # Debug assertion – remove once you're confident
        if t.size > 1 and not np.all(np.diff(t) >= 0):
            raise RuntimeError(f"Non-monotonic spike times for neuron {nid}")

        t = np.sort(t)
        isi = np.diff(t)

        # Need enough ISIs for the chosen ddof
        if isi.size == 0 or isi.size <= ddof:
            continue

        mean_isi = float(np.mean(isi))
        if not np.isfinite(mean_isi) or mean_isi <= 0.0:
            continue

        std_isi = float(np.std(isi, ddof=ddof))
        if not np.isfinite(std_isi):
            continue

        cv = std_isi / mean_isi

        if not np.isfinite(cv):
            continue

        cvs.append(cv)
        kept_ids.append(int(nid))

    if not cvs:
        return np.array([], dtype=int), np.array([], dtype=float)

    return np.array(kept_ids, dtype=int), np.array(cvs, dtype=float)