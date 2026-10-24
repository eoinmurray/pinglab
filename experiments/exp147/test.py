"""Check native graded timing, protocol boundaries and single-HCO continuation.

The independent reference verifies the proposal's equations, not Toolbox code.
All native runs use one HCO and float64 CPU execution; no batching benchmark
or training is performed by these tests.
"""

import copy

import numpy as np
import pytest
import torch
from experiments.exp147 import recipe
from experiments.exp147.analyse import validate_and_measure
from experiments.exp147.custom.execution import execute, executor
from scipy.special import expit
from snnlab.sim import NetworkSnapshot, plan_graph


def arrays(result, cfg):
    saved = {
        name: value[:, 0].copy()
        for name, value in result.numpy(batch=0).outputs.items()
    }
    steps = len(saved["n1_voltage_mv"])
    saved["time_ms"] = (np.arange(steps) + 1) * cfg["dt_ms"]
    saved["input_time_ms"] = np.arange(steps) * cfg["dt_ms"]
    return saved


def test_one_step_uses_old_source_voltage_and_new_conductance():
    cfg = recipe.configuration(duration_ms=0.001)
    bundle = recipe.author_network(cfg)
    graph = copy.deepcopy(bundle.graph)
    graph["populations"][0]["neuron"]["config"]["initial_voltage_mv"] = -40.0
    graph["populations"][1]["neuron"]["config"]["initial_voltage_mv"] = -65.0
    from snnlab.sim import GraphExecutor

    model = GraphExecutor(
        plan_graph(graph), compile_mode=None, extension_validation="strict"
    ).double()
    with torch.no_grad():
        result = model({"clock": recipe.clock(cfg)}, diagnostics=False)
    gate = -70 + 0.001 / 10 * (-40 + 70)
    conductance = expit(gate + 55)
    assert result.outputs["n1_to_n2_gate_mv"].item() == pytest.approx(gate, abs=1e-13)
    assert result.outputs["n1_to_n2_conductance_us"].item() == pytest.approx(
        conductance, abs=1e-15
    )
    voltage = -65.0
    g_na = expit(voltage + 60)
    g_k = 10 * expit(voltage + 60)
    expected = voltage + 0.001 / 0.01 * (
        -0.01 * (voltage + 70)
        - g_na * (voltage - 50)
        - g_k * (voltage + 85)
        - conductance * (voltage + 85)
    )
    assert result.outputs["n2_voltage_mv"].item() == pytest.approx(expected, abs=1e-12)
    plan = plan_graph(bundle.graph)
    assert all(edge.continuous and edge.delay_steps == 0 for edge in plan.projections)
    assert all(parameter["unit"] == "1" for parameter in bundle.graph["parameters"])


def test_current_boundaries_independent_euler_and_snapshot(tmp_path):
    cfg = recipe.configuration(duration_ms=10.002)
    torch.set_num_threads(1)
    bundle = recipe.author_network(cfg)
    result, _ = execute(
        bundle, recipe.clock(cfg), cfg, snapshot_path=tmp_path / "start"
    )
    data = arrays(result, cfg)
    summary = validate_and_measure(data, cfg)
    assert summary["independent_euler_passed"]
    assert not summary["toolbox_reference_validated"]
    for neuron in summary["neurons"].values():
        assert neuron["max_voltage_difference_mv"] < 1e-8
    assert data["n1_current_na"][4999] == 0
    assert data["n1_current_na"][5000] == 1
    assert data["n1_current_na"][9999] == 1
    assert data["n1_current_na"][10000] == 0.8
    assert data["n2_current_na"][9999] == 0
    assert data["n2_current_na"][10000] == 0.8
    snapshot = NetworkSnapshot.read(tmp_path / "start" / "snapshot.json")
    assert all(
        value.dtype == torch.float64 for value in snapshot.load_parameters().values()
    )
    # Split at the kick onset: integer schedule state and synaptic gates persist.
    model = executor(bundle, cfg)
    clock = recipe.clock(cfg)
    with torch.no_grad():
        first = model({"clock": clock[:5000]}, diagnostics=False)
        tail = model(
            {"clock": clock[5000:]},
            diagnostics=False,
            runtime_state=first.runtime_state,
        )
    for name, output in result.outputs.items():
        torch.testing.assert_close(
            torch.cat((first.outputs[name], tail.outputs[name])), output, rtol=0, atol=0
        )


def test_configuration_and_single_hco_shape():
    cfg = recipe.configuration()
    cfg["synapse"]["g_max_us"] = 99
    assert recipe.configuration()["synapse"]["g_max_us"] == 1
    assert recipe.clock(recipe.configuration(duration_ms=0.002)).shape == (2, 1, 1)


@pytest.mark.parametrize("duration", [0, -1, 201, float("nan"), 0.0015])
def test_invalid_durations(duration):
    with pytest.raises(ValueError):
        recipe.configuration(duration_ms=duration)
