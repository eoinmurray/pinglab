import numpy as np
import pytest

from pinglab.backends.pytorch import simulate_network as run_network_pytorch
from pinglab.io import compile_graph_to_runtime


def _tiny_graph() -> dict:
    return {
        "schema_version": "pinglab-graph.v1",
        "sim": {"dt_ms": 0.1, "T_ms": 200.0, "seed": 7, "neuron_model": "lif"},
        "execution": {},
        "constraints": {"nonnegative_weights": True, "nonnegative_input": True},
        "biophysics": {
            "V_init": -65.0,
            "E_L": -65.0,
            "E_e": 0.0,
            "E_i": -80.0,
            "C_m_E": 1.0,
            "g_L_E": 0.05,
            "C_m_I": 1.0,
            "g_L_I": 0.1,
            "V_th": -50.0,
            "V_reset": -65.0,
            "t_ref_E": 3.0,
            "t_ref_I": 1.5,
            "tau_ampa": 2.0,
            "tau_gaba": 6.5,
            "g_L_heterogeneity_sd": 0.0,
            "C_m_heterogeneity_sd": 0.0,
            "V_th_heterogeneity_sd": 0.0,
            "t_ref_heterogeneity_sd": 0.0,
        },
        "nodes": [
            {"id": "input_1", "kind": "input", "type": "tonic", "size": 0},
            {"id": "E", "kind": "population", "type": "E", "size": 20},
            {"id": "I", "kind": "population", "type": "I", "size": 5},
        ],
        "edges": [
            {"id": "in_to_e", "kind": "input", "from": "input_1", "to": "E", "w": {"mean": 1.0, "std": 0.0}},
            {"id": "e_to_i", "kind": "EI", "from": "E", "to": "I", "w": {"mean": 0.02, "std": 0.0}, "delay_ms": 0.5},
            {"id": "i_to_e", "kind": "IE", "from": "I", "to": "E", "w": {"mean": 0.03, "std": 0.0}, "delay_ms": 1.2},
            {"id": "e_to_e", "kind": "EE", "from": "E", "to": "E", "w": {"mean": 0.01, "std": 0.0}, "delay_ms": 0.8},
            {"id": "i_to_i", "kind": "II", "from": "I", "to": "I", "w": {"mean": 0.005, "std": 0.0}, "delay_ms": 0.8},
        ],
        "inputs": {"input_1": {"mode": "tonic", "mean": 1.6, "std": 0.0, "seed": 7}},
        "meta": {},
    }


def test_pytorch_runtime_compiles_and_produces_spikes() -> None:
    torch = pytest.importorskip("torch")
    _ = torch

    spec = _tiny_graph()
    runtime_pt = compile_graph_to_runtime(spec, backend="pytorch")
    out_pt = run_network_pytorch(runtime_pt)
    spikes_pt = out_pt.spikes
    assert spikes_pt.times.ndim == 1
    assert spikes_pt.ids.ndim == 1
    assert spikes_pt.types is not None
    assert np.all(spikes_pt.ids >= 0)


def test_pytorch_simulation_deterministic_given_same_runtime_seed() -> None:
    torch = pytest.importorskip("torch")
    _ = torch

    spec = _tiny_graph()
    runtime_pt_a = compile_graph_to_runtime(spec, backend="pytorch")
    runtime_pt_b = compile_graph_to_runtime(spec, backend="pytorch")

    out_a = run_network_pytorch(runtime_pt_a)
    out_b = run_network_pytorch(runtime_pt_b)

    np.testing.assert_array_equal(out_a.spikes.times, out_b.spikes.times)
    np.testing.assert_array_equal(out_a.spikes.ids, out_b.spikes.ids)
    np.testing.assert_array_equal(out_a.spikes.types, out_b.spikes.types)
