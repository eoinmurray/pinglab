import numpy as np
import pytest

from pinglab.io import slice_spikes
from pinglab.backends.types import Spikes


def test_slice_spikes_basic():
    spikes = Spikes(times=np.array([0.0, 5.0, 10.0]), ids=np.array([0, 1, 2]))
    sliced = slice_spikes(spikes, 1.0, 10.0)
    np.testing.assert_allclose(sliced.times, [5.0])
    np.testing.assert_allclose(sliced.ids, [1])


def test_slice_spikes_invalid():
    spikes = Spikes(times=np.array([0.0]), ids=np.array([0]))
    with pytest.raises(ValueError):
        slice_spikes(spikes, 5.0, 5.0)
