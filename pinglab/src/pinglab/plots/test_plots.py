import numpy as np
import pytest

from pinglab.plots.raster import save_raster
from pinglab.plots.instrument import save_instrument_traces
from pinglab.types import Spikes, InstrumentsResults


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


def test_save_instrument_traces_creates_files(tmp_path):
    instruments = InstrumentsResults(
        times=np.arange(5, dtype=float),
        neuron_ids=np.array([0, 1]),
        types=np.array([0, 1]),
        V=np.random.RandomState(0).rand(5, 2),
        g_e=np.random.RandomState(1).rand(5, 2),
        g_i=np.random.RandomState(2).rand(5, 2),
        V_mean_E=np.random.RandomState(3).rand(5),
        V_mean_I=np.random.RandomState(4).rand(5),
        g_e_mean_E=np.random.RandomState(5).rand(5),
        g_i_mean_E=np.random.RandomState(6).rand(5),
    )
    save_instrument_traces(instruments, tmp_path)

    expected = [
        "trace_voltage_light.png",
        "trace_g_e_light.png",
        "trace_g_i_light.png",
        "trace_voltage_mean_light.png",
        "trace_g_e_mean_light.png",
        "trace_g_i_mean_light.png",
    ]
    for name in expected:
        assert (tmp_path / name).exists()
