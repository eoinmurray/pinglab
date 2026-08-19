from __future__ import annotations

import pytest
from experiments.exp087 import (
    BACKGROUND_CHANNELS,
    BACKGROUND_FAN_IN,
    FEEDFORWARD_DELAY_MS,
    FEEDFORWARD_FAN_IN,
    LAYERS,
    NEURONS_PER_LAYER,
    PACKET_CHANNELS,
    author_network,
)


@pytest.fixture(scope="module")
def graph() -> dict:
    return author_network().graph


def test_network_is_a_six_pool_feedforward_chain(graph: dict) -> None:
    populations = {row["id"]: row for row in graph["populations"]}
    assert list(populations) == [f"pool_{layer}" for layer in range(1, LAYERS + 1)]
    assert all(row["size"] == NEURONS_PER_LAYER for row in populations.values())

    projections = {row["id"]: row for row in graph["projections"]}
    expected_chain = ["packet_to_pool_1"] + [
        f"pool_{layer}_to_pool_{layer + 1}" for layer in range(1, LAYERS)
    ]
    for projection_id in expected_chain:
        projection = projections[projection_id]
        assert projection["connection"] == "feedforward"
        assert projection["delay"]["value"] == FEEDFORWARD_DELAY_MS


def test_packet_and_background_connections_use_exact_fan_in(graph: dict) -> None:
    parameters = {row["id"]: row for row in graph["parameters"]}

    packet = parameters["packet_to_pool_1.weight"]["initializer"]
    assert packet["zeroing"] == "exact_k"
    assert round((1.0 - packet["initial_zero_fraction"]) * PACKET_CHANNELS) == (
        FEEDFORWARD_FAN_IN
    )

    for layer in range(1, LAYERS + 1):
        background = parameters[f"background_to_pool_{layer}.weight"]["initializer"]
        assert background["zeroing"] == "exact_k"
        assert round(
            (1.0 - background["initial_zero_fraction"]) * BACKGROUND_CHANNELS
        ) == BACKGROUND_FAN_IN

    for layer in range(1, LAYERS):
        feedforward = parameters[f"pool_{layer}_to_pool_{layer + 1}.weight"]
        initializer = feedforward["initializer"]
        assert initializer["initial_zero_fraction"] == pytest.approx(
            1.0 - FEEDFORWARD_FAN_IN / NEURONS_PER_LAYER
        )
