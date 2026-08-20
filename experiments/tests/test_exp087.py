from __future__ import annotations

import pytest
from experiments.exp087 import (
    BACKGROUND_CHANNELS,
    BACKGROUND_FAN_IN,
    DT_MS,
    FEEDFORWARD_DELAY_MS,
    FEEDFORWARD_FAN_IN,
    LAYERS,
    NEURONS_PER_LAYER,
    PACKET_CHANNELS,
    REPRESENTATIVE_PACKETS,
    T_MS,
    TARGET_OUTPUT_RATE_HZ,
    VOLLEY_MIN_PEAK_SPIKES_BY_POOL,
    author_network,
    make_background,
    make_packet,
    packet_width_ms,
    run_background_only,
    run_packet,
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

    for pathway in ("excitation", "inhibition"):
        for layer in range(1, LAYERS + 1):
            background = parameters[f"background_{pathway}_to_pool_{layer}.weight"][
                "initializer"
            ]
            assert background["zeroing"] == "exact_k"
            assert (
                round((1.0 - background["initial_zero_fraction"]) * BACKGROUND_CHANNELS)
                == BACKGROUND_FAN_IN
            )

    for layer in range(1, LAYERS):
        feedforward = parameters[f"pool_{layer}_to_pool_{layer + 1}.weight"]
        initializer = feedforward["initializer"]
        assert initializer["initial_zero_fraction"] == pytest.approx(
            1.0 - FEEDFORWARD_FAN_IN / NEURONS_PER_LAYER
        )


def test_packet_generator_preserves_size_and_width() -> None:
    packet = make_packet(alpha=50, sigma_ms=2.0).numpy()

    assert packet.shape[0] == round(T_MS / DT_MS)
    assert int(packet.sum()) == 50
    assert packet_width_ms(packet) == pytest.approx(2.0, abs=0.5)


def test_selected_point_separates_extinction_from_convergence(graph: dict) -> None:
    background = make_background()
    background_metrics = run_background_only(graph, background)
    assert background_metrics["mean_settled_rate_hz"] == pytest.approx(
        TARGET_OUTPUT_RATE_HZ,
        abs=0.5,
    )
    assert all(
        8.0 <= rate <= 12.0 for rate in background_metrics["settled_rate_hz_by_pool"]
    )
    assert all(
        peak < threshold
        for peak, threshold in zip(
            background_metrics["max_1ms_spikes_by_pool"],
            VOLLEY_MIN_PEAK_SPIKES_BY_POOL,
            strict=True,
        )
    )

    packets = {
        packet_id: (label, alpha, sigma_ms)
        for packet_id, label, alpha, sigma_ms in REPRESENTATIVE_PACKETS
    }
    weak_label, weak_alpha, weak_sigma = packets["weak"]
    weak = run_packet(
        graph,
        packet_id="weak",
        label=weak_label,
        alpha=weak_alpha,
        sigma_ms=weak_sigma,
        background=background,
    )
    broad_label, broad_alpha, broad_sigma = packets["broad"]
    broad = run_packet(
        graph,
        packet_id="broad",
        label=broad_label,
        alpha=broad_alpha,
        sigma_ms=broad_sigma,
        background=background,
    )
    oversized_label, oversized_alpha, oversized_sigma = packets["oversized"]
    oversized = run_packet(
        graph,
        packet_id="oversized",
        label=oversized_label,
        alpha=oversized_alpha,
        sigma_ms=oversized_sigma,
        background=background,
    )

    assert not weak.survives
    assert broad.alphas[-1] >= 0.9 * NEURONS_PER_LAYER
    assert oversized.alphas[-1] >= 0.9 * NEURONS_PER_LAYER
    assert broad.sigmas_ms[-1] == pytest.approx(oversized.sigmas_ms[-1], abs=0.1)
