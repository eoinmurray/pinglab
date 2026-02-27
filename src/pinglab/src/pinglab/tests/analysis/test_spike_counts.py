import numpy as np

from pinglab.analysis import spike_count_for_range, total_e_spikes
from pinglab.backends.types import Spikes


def test_spike_count_for_range_counts_ids_in_half_open_interval() -> None:
    ids = np.array([0, 1, 1, 3, 4, 5], dtype=int)
    assert spike_count_for_range(ids, 1, 4) == 3


def test_total_e_spikes_uses_types_when_present() -> None:
    spikes = Spikes(
        times=np.array([1.0, 2.0, 3.0, 4.0]),
        ids=np.array([0, 5, 2, 6], dtype=int),
        types=np.array([0, 1, 0, 1], dtype=int),
    )
    assert total_e_spikes(spikes, n_e=3) == 2


def test_total_e_spikes_falls_back_to_id_threshold() -> None:
    spikes = Spikes(
        times=np.array([1.0, 2.0, 3.0]),
        ids=np.array([0, 2, 5], dtype=int),
        types=None,
    )
    assert total_e_spikes(spikes, n_e=3) == 2
