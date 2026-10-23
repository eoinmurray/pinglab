"""Validate numerical ordering and the native custom-neuron integration.

These tests compare against independent scalar equations, exercise the current
onset through native graph execution and check split continuation. They do not
claim agreement with the absent Toolbox reference files.
"""

from dataclasses import replace

import numpy as np
import pytest
import torch
from experiments.exp146 import recipe
from experiments.exp146.custom import dynamics
from experiments.exp146.custom.execution import execute
from experiments.exp146.custom.validation import compare_reference, validate_traces
from snnlab.extensions import NeuronContext, StateContext
from snnlab.sim import GraphExecutor, NetworkSnapshot, plan_graph


def test_gate_first_step_and_no_reset():
    cfg = recipe.configuration()
    state = dynamics.initial_state(
        StateContext((1, 1), "cpu", torch.float64, cfg["dt_ms"], cfg)
    )
    # Unequal gates distinguish updated-gate from old-gate membrane ordering.
    state["voltage"].fill_(-20.01)
    state["z_na"].fill_(-58)
    state["z_k"].fill_(-65)
    current = torch.tensor([[0.8]], dtype=torch.float64)
    ctx = NeuronContext(
        state, current, torch.zeros_like(current), cfg["dt_ms"], cfg, lambda x: x
    )
    updated, spikes = dynamics.step(ctx)
    z_na = -58 + 0.001 * (-20.01 + 58)
    z_k = -65 + 0.001 / 10 * (-20.01 + 65)
    g_na = 1 / (1 + np.exp(-(z_na + 60)))
    g_k = 10 / (1 + np.exp(-(z_k + 60)))
    expected = -20.01 + 0.001 / 0.01 * (
        0.8 - 0.01 * (-20.01 + 70) - g_na * (-20.01 - 50) - g_k * (-20.01 + 85)
    )
    assert updated["voltage"].item() == pytest.approx(expected, abs=1e-12)
    assert updated["z_na"].item() == pytest.approx(z_na)
    assert updated["z_k"].item() == pytest.approx(z_k)
    assert spikes.item() == 1
    assert updated["voltage"].item() > -20
    # A second step above threshold must not emit another crossing event.
    _, repeated = dynamics.step(replace(ctx, state=updated))
    assert repeated.item() == 0


def test_native_current_onset_euler_and_snapshot(tmp_path):
    cfg = recipe.configuration(duration_ms=10.1)
    bundle = recipe.author_network(cfg)
    drive = recipe.current_waveform(cfg)
    torch.set_num_threads(1)
    result, _ = execute(bundle, drive, cfg, snapshot_path=tmp_path / "start")
    data = {key: value[:, 0] for key, value in result.numpy(batch=0).outputs.items()}
    data["input_time_ms"] = np.arange(len(drive)) * cfg["dt_ms"]
    data["time_ms"] = (np.arange(len(drive)) + 1) * cfg["dt_ms"]
    validate_traces(data, cfg)
    summary = compare_reference(data, cfg)
    assert summary["max_voltage_difference_mv"] < 1e-8
    assert summary["max_gate_difference_mv"] < 1e-8
    assert not summary["toolbox_reference_validated"]
    assert data["current_na"][9999] == 0
    assert data["current_na"][10000] == 0.8
    snapshot = NetworkSnapshot.read(tmp_path / "start" / "snapshot.json")
    weights = snapshot.load_parameters()
    assert weights["injection.weight"].dtype == torch.float64


def test_split_continuation_matches_and_configurations_are_independent():
    cfg = recipe.configuration(duration_ms=0.02)
    bundle = recipe.author_network(cfg)
    model = GraphExecutor(plan_graph(bundle.graph), compile_mode=None).to(
        dtype=torch.float64
    )
    drive = torch.full((20, 2, 1), 0.8, dtype=torch.float64)
    with torch.no_grad():
        full = model({"drive": drive}, diagnostics=False)
        first = model({"drive": drive[:7]}, diagnostics=False)
        tail = model(
            {"drive": drive[7:]}, diagnostics=False, runtime_state=first.runtime_state
        )
    for name, values in full.outputs.items():
        torch.testing.assert_close(
            torch.cat((first.outputs[name], tail.outputs[name])), values, rtol=0, atol=0
        )
    cfg["validation"]["voltage_atol_mv"] = 100
    assert recipe.configuration()["validation"]["voltage_atol_mv"] == 0.1


@pytest.mark.parametrize("duration", [0, -1, 50.1, float("nan"), 0.0015])
def test_invalid_durations(duration):
    with pytest.raises(ValueError):
        recipe.configuration(duration_ms=duration)
