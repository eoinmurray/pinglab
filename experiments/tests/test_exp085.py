from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
from execution import ExecutionSpec, plan_graph, runtime_state_signature, simulate
from experiments import exp085


def test_protocol_scans_eleven_couplings_and_predeclares_rasters():
    assert exp085.COUPLINGS == tuple(round(value, 2) for value in np.linspace(0, 0.1, 11))
    assert exp085.REPRESENTATIVE_K == (0.0, 0.03, 0.06, 0.1)
    assert exp085.TAU_A_MS == 4.0
    assert exp085.TAU_B_MS == 5.0


def test_private_inputs_are_deterministic_and_continuous():
    first = exp085.make_inputs()
    repeated = exp085.make_inputs()
    assert set(first) == {"drive_a", "drive_b"}
    np.testing.assert_array_equal(first["drive_a"], repeated["drive_a"])
    assert not np.array_equal(first["drive_a"], first["drive_b"])
    expected_steps = round((exp085.EQUILIBRATION_MS + exp085.CONTINUATION_MS) / exp085.DT_MS)
    assert first["drive_a"].shape == (expected_steps, 5, 128)


def test_only_cross_coupling_initializers_change_across_graphs():
    zero = exp085.author_network(0.0).graph
    strong = exp085.author_network(0.1).graph
    changed = []
    zero_parameters = {row["id"]: row for row in zero["parameters"]}
    for row in strong["parameters"]:
        if row != zero_parameters[row["id"]]:
            changed.append(row["id"])
    assert set(changed) == {
        "a_E_to_b_E.weight",
        "a_E_to_b_I.weight",
        "b_E_to_a_E.weight",
        "b_E_to_a_I.weight",
    }
    assert runtime_state_signature(plan_graph(zero)) == runtime_state_signature(plan_graph(strong))


def test_zero_coupling_state_branch_matches_uninterrupted_continuation():
    bundle = exp085.author_network(0.0)
    graph = copy.deepcopy(bundle.graph)
    steps = 40
    inputs = {
        "drive_a": torch.zeros(steps, 1, exp085.N_INPUT),
        "drive_b": torch.zeros(steps, 1, exp085.N_INPUT),
    }
    inputs["drive_a"][::5] = 1
    inputs["drive_b"][2::7] = 1
    whole = simulate(
        ExecutionSpec(kind="simulate", executor="graph", graph=graph, inputs=inputs, seed=7)
    )
    first = simulate(
        ExecutionSpec(
            kind="simulate",
            executor="graph",
            graph=graph,
            inputs={name: value[:17] for name, value in inputs.items()},
            seed=7,
        )
    )
    assert first.runtime_state is not None
    second = simulate(
        ExecutionSpec(
            kind="simulate",
            executor="graph",
            graph=graph,
            inputs={name: value[17:] for name, value in inputs.items()},
            seed=7,
            runtime_state=first.runtime_state,
        )
    )
    for name, expected in whole.recordings.items():
        actual = torch.cat((first.recordings[name], second.recordings[name]))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_relative_phase_reports_linear_drift():
    steps = round((exp085.EQUILIBRATION_MS + exp085.CONTINUATION_MS) / exp085.DT_MS)
    time_s = np.arange(steps) * exp085.DT_MS / 1_000.0
    a = np.sin(2 * np.pi * 38 * time_s)[:, None, None]
    b = np.sin(2 * np.pi * 33 * time_s)[:, None, None]
    phase = exp085.relative_phase(a, b, onset_step=round(exp085.EQUILIBRATION_MS / exp085.DT_MS))
    slope = np.polyfit(np.arange(phase.shape[1]) * exp085.DT_MS / 1_000.0, phase[0], 1)[0]
    assert slope == pytest.approx(2 * np.pi * 5, rel=0.05)


def test_terminal_phase_error_reports_approach_to_terminal_offset():
    steps = round(1_000.0 / exp085.TRACE_BIN_MS)
    time_s = np.arange(steps) * exp085.TRACE_BIN_MS / 1_000.0
    approach = np.exp(-time_s / 0.15)

    _, error = exp085.trial_terminal_phase_error(approach)

    assert error[0] > 0.7
    assert error[-1] < 0.01


def test_clean_convergence_trials_rejects_late_phase_slip():
    steps = round(1_000.0 / exp085.DT_MS)
    stable = np.zeros(steps)
    slipping = stable.copy()
    slipping[round(600.0 / exp085.DT_MS) :] = np.pi

    assert exp085.clean_convergence_trials(np.stack((stable, slipping))) == [0]
