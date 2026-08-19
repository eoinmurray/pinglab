from __future__ import annotations

import pytest
from experiments.exp085 import (
    COUPLING_DELAY_MS,
    CROSS_FAN_IN,
    CROSS_ZERO_FRACTION,
    DT_MS,
    E_REFRACTORY_MS,
    I_REFRACTORY_MS,
    K_EE,
    K_EI,
    N_E,
    N_I,
    author_network,
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
    for name in ("PING_A", "PING_B"):
        assert projections[f"{name}_E_to_I"]["target"] == f"{name}_I.excitatory"
        assert projections[f"{name}_I_to_E"]["target"] == f"{name}_E.inhibitory"
        assert f"{name}_E_to_E" not in projections
        assert f"{name}_I_to_I" not in projections


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
