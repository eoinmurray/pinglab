import numpy as np
import pytest
from pathlib import Path

from pinglab.utils import expand_parameter_spec, load_config, slice_spikes
from pinglab.types import Spikes


def test_expand_parameter_spec_grid():
    spec = {
        "a": {"type": "linspace", "start": 0.0, "stop": 1.0, "num": 3},
        "b": {"type": "values", "values": [10, 20]},
    }
    expanded = expand_parameter_spec(spec, mode="grid")
    assert len(expanded) == 6
    assert {d["a"] for d in expanded} == {0.0, 0.5, 1.0}
    assert {d["b"] for d in expanded} == {10, 20}


def test_expand_parameter_spec_invalid():
    with pytest.raises(ValueError):
        expand_parameter_spec({"a": {"type": "bad"}}, mode="grid")
    with pytest.raises(ValueError):
        expand_parameter_spec({"a": {"type": "values", "values": [1]}}, mode="unknown")
    with pytest.raises(NotImplementedError):
        expand_parameter_spec({"a": {"type": "values", "values": [1]}}, mode="random")


def test_slice_spikes_basic():
    spikes = Spikes(times=np.array([0.0, 5.0, 10.0]), ids=np.array([0, 1, 2]))
    sliced = slice_spikes(spikes, 1.0, 10.0)
    np.testing.assert_allclose(sliced.times, [5.0])
    np.testing.assert_allclose(sliced.ids, [1])


def test_slice_spikes_invalid():
    spikes = Spikes(times=np.array([0.0]), ids=np.array([0]))
    with pytest.raises(ValueError):
        slice_spikes(spikes, 5.0, 5.0)


def test_load_config_valid(tmp_path: Path):
    config = tmp_path / "config.yaml"
    config.write_text(
        """
base:
  dt: 0.1
  T: 10.0
  N_E: 1
  N_I: 1
  g_ei: 0.5
  g_ie: 1.0
  g_ee: 0.0
  g_ii: 0.0
  p_ee: 1.0
  p_ei: 1.0
  p_ie: 1.0
  p_ii: 1.0
  delay_ei: 1.0
  delay_ie: 1.0
  delay_ee: 1.0
  delay_ii: 1.0
  V_init: -65.0
  E_L: -65.0
  E_e: 0.0
  E_i: -80.0
  C_m_E: 1.0
  g_L_E: 0.1
  C_m_I: 1.0
  g_L_I: 0.1
  V_th: -50.0
  V_reset: -65.0
  tau_ampa: 5.0
  tau_gaba: 10.0
  t_ref_E: 2.0
  t_ref_I: 1.0
  connectivity_scaling: one_over_N_src

default_inputs:
  I_E: 1.0
  I_I: 1.0
  noise: 0.0
"""
    )
    loaded = load_config(config)
    assert loaded.base.N_E == 1
    assert loaded.default_inputs.I_E == 1.0


def test_load_config_invalid_yaml(tmp_path: Path):
    config = tmp_path / "bad.yaml"
    config.write_text("base: [")
    with pytest.raises(Exception):
        load_config(config)


def test_load_config_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml")
