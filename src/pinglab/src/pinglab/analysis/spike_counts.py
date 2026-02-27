from __future__ import annotations

import numpy as np

from pinglab.backends.types import Spikes


def spike_count_for_range(spike_ids: np.ndarray, start: int, stop: int) -> int:
    if stop <= start:
        return 0
    ids = np.asarray(spike_ids)
    mask = (ids >= start) & (ids < stop)
    return int(mask.sum())


def total_e_spikes(spikes: Spikes, n_e: int) -> int:
    types = getattr(spikes, "types", None)
    if types is not None and np.size(types) == np.size(spikes.ids):
        return int(np.sum(np.asarray(types) == 0))
    return int(np.sum(np.asarray(spikes.ids) < int(n_e)))
