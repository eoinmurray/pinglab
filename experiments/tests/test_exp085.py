from __future__ import annotations

import numpy as np
import pytest
from experiments.exp085 import (
    COUPLING_DELAY_MS,
    CROSS_FAN_IN,
    CROSS_ZERO_FRACTION,
    DT_MS,
    E_REFRACTORY_MS,
    E_TO_I_TAU_MS,
    E_TO_I_WEIGHT,
    I_REFRACTORY_MS,
    INPUT_RATE_A_HZ,
    INPUT_RATE_B_HZ,
    K_EE,
    K_EI,
    N_E,
    N_I,
    N_INPUT,
    T_MS,
    author_network,
    interpolated_phase,
    inhibitory_cycle_summary,
    make_uncoupled_inputs,
    rhythm_summary,
)


@pytest.fixture(scope="module")
def graph() -> dict:
    return author_network().graph


def test_network_contains_two_matched_ping_circuits(graph: dict) -> None:
    populations = {row["id"]: row for row in graph["populations"]}
    assert populations["PING_A_E"]["size"] == N_E
    assert populations["PING_B_E"]["size"] == N_E
    assert populations["PING_A_I"]["size"] == N_I
    assert populations["PING_B_I"]["size"] == N_I

    assert (
        populations["PING_A_E"]["neuron"]
        == populations["PING_B_E"]["neuron"]
    )
    assert (
        populations["PING_A_I"]["neuron"]
        == populations["PING_B_I"]["neuron"]
    )
    assert populations["PING_A_E"]["neuron"]["refractory_steps"] == round(
        E_REFRACTORY_MS / DT_MS
    )
    assert populations["PING_A_I"]["neuron"]["refractory_steps"] == round(
        I_REFRACTORY_MS / DT_MS
    )


def test_network_has_only_local_e_to_i_to_e_ping_loops(graph: dict) -> None:
    projections = {row["id"]: row for row in graph["projections"]}
    parameters = {row["id"]: row for row in graph["parameters"]}
    for name in ("PING_A", "PING_B"):
        assert projections[f"{name}_E_to_I"]["target"] == f"{name}_I.excitatory"
        assert projections[f"{name}_I_to_E"]["target"] == f"{name}_E.inhibitory"
        assert f"{name}_E_to_E" not in projections
        assert f"{name}_I_to_I" not in projections
        e_to_i = projections[f"{name}_E_to_I"]
        assert e_to_i["synapse"]["tau"]["value"] == E_TO_I_TAU_MS
        initializer = parameters[f"{name}_E_to_I.weight"]["initializer"]
        assert initializer["mean"] == E_TO_I_WEIGHT


def test_cross_network_paths_are_reciprocal_and_separately_weighted(
    graph: dict,
) -> None:
    projections = {row["id"]: row for row in graph["projections"]}
    parameters = {row["id"]: row for row in graph["parameters"]}
    expected = {
        "PING_A_E_to_PING_B_E_K_EE": K_EE,
        "PING_A_E_to_PING_B_I_K_EI": K_EI,
        "PING_B_E_to_PING_A_E_K_EE": K_EE,
        "PING_B_E_to_PING_A_I_K_EI": K_EI,
    }

    for projection_id, strength in expected.items():
        projection = projections[projection_id]
        assert projection["connection"] == "feedback"
        assert projection["delay"]["value"] == COUPLING_DELAY_MS
        initializer = parameters[f"{projection_id}.weight"]["initializer"]
        assert initializer["mean"] == strength
        assert initializer["initial_zero_fraction"] == pytest.approx(
            CROSS_ZERO_FRACTION
        )
        assert initializer["zeroing"] == "exact_k"

    realised_fan_in = round((1.0 - CROSS_ZERO_FRACTION) * N_E)
    assert realised_fan_in == CROSS_FAN_IN


def test_uncoupled_inputs_use_the_two_design_rates() -> None:
    inputs = make_uncoupled_inputs()
    duration_s = T_MS / 1_000.0
    realised_a = (
        float(inputs[f"drive_A_{INPUT_RATE_A_HZ:g}_Hz"].sum())
        / N_INPUT
        / duration_s
    )
    realised_b = (
        float(inputs[f"drive_B_{INPUT_RATE_B_HZ:g}_Hz"].sum())
        / N_INPUT
        / duration_s
    )
    assert realised_a == pytest.approx(INPUT_RATE_A_HZ, rel=0.03)
    assert realised_b == pytest.approx(INPUT_RATE_B_HZ, rel=0.03)


def test_phase_and_frequency_follow_detected_volley_intervals() -> None:
    interval_steps = round(25.0 / DT_MS)
    peaks = np.arange(0, 5 * interval_steps, interval_steps)
    summary = rhythm_summary(peaks)
    phase = interpolated_phase(peaks, steps=6 * interval_steps)

    assert summary["frequency_hz"] == 40.0
    assert summary["iei_cv"] == 0.0
    np.testing.assert_allclose(
        phase[:interval_steps],
        2.0 * np.pi * np.arange(interval_steps) / interval_steps,
    )


def test_inhibitory_cycle_summary_counts_each_neuron_between_volleys() -> None:
    spikes = np.zeros((8, 1, 2), dtype=np.uint8)
    spikes[1, 0] = 1
    spikes[5, 0] = 1

    summary = inhibitory_cycle_summary(spikes, np.array([0, 4, 8]))

    assert summary == {
        "cycles": 2,
        "mean_spikes_per_neuron": 1.0,
        "minimum": 1,
        "maximum": 1,
    }
