import numpy as np
import pytest

from pinglab.plots.raster import save_raster
from pinglab.backends.types import Spikes


def test_save_raster_creates_files(tmp_path):
    spikes = Spikes(times=np.array([0.0, 10.0]), ids=np.array([0, 1]))
    out = tmp_path / "raster.png"
    save_raster(spikes, out)
    assert (tmp_path / "raster_light.png").exists()
    assert (tmp_path / "raster_dark.png").exists()


def test_save_raster_requires_dt_with_input(tmp_path):
    spikes = Spikes(times=np.array([0.0]), ids=np.array([0]))
    out = tmp_path / "raster.png"
    with pytest.raises(ValueError):
        save_raster(spikes, out, external_input=np.zeros((5, 1)))
