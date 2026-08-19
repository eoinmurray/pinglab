from __future__ import annotations

import numpy as np
from experiments.exp085 import DT_MS, N_INPUT, author_network
from experiments.exp086 import (
    K_VALUES,
    PHASE_BINS,
    analyse_trajectory,
    circular_distance,
    instantaneous_frequency,
    make_inputs,
)


def test_coupling_sweep_has_locked_endpoint_intermediate_values_and_zero() -> None:
    assert K_VALUES[0] == 0.08
    assert K_VALUES[-1] == 0.0
    assert np.all(np.diff(K_VALUES) < 0)
    assert len(K_VALUES) > 2


def test_each_branch_uses_equal_e_to_e_and_e_to_i_strength() -> None:
    for k in (K_VALUES[0], K_VALUES[len(K_VALUES) // 2], K_VALUES[-1]):
        graph = author_network(k_ee=float(k), k_ei=float(k)).graph
        parameters = {row["id"]: row for row in graph["parameters"]}
        for source, target in (("PING_A", "PING_B"), ("PING_B", "PING_A")):
            ee = parameters[
                f"{source}_E_to_{target}_E_K_EE.weight"
            ]["initializer"]["mean"]
            ei = parameters[
                f"{source}_E_to_{target}_I_K_EI.weight"
            ]["initializer"]["mean"]
            assert ee == ei == float(k)


def test_fixed_input_generation_is_reproducible() -> None:
    first = make_inputs()
    second = make_inputs()
    assert first.keys() == second.keys()
    for name in first:
        assert first[name].shape[-1] == N_INPUT
        assert np.array_equal(first[name].numpy(), second[name].numpy())


def test_instantaneous_frequency_follows_intervolley_intervals() -> None:
    interval = round(25.0 / DT_MS)
    peaks = np.arange(0, 4 * interval, interval)
    frequency = instantaneous_frequency(peaks, steps=5 * interval)
    assert np.all(frequency[: 3 * interval] == 40.0)
    assert np.isnan(frequency[3 * interval :]).all()


def test_circular_distance_wraps_at_pi() -> None:
    np.testing.assert_allclose(
        circular_distance(-np.pi + 0.1, np.pi - 0.1),
        0.2,
        atol=1e-12,
    )


def test_phase_analysis_reports_expected_number_of_bins() -> None:
    steps = 10_000
    recordings = {}
    for index, period in enumerate((200, 200, 220, 220)):
        population = 80 if index % 2 == 0 else 20
        spikes = np.zeros((steps, 1, population), dtype=np.uint8)
        spikes[np.arange(50, steps, period), 0, :] = 1
        recordings[f"population_{index}"] = spikes
    analysis = analyse_trajectory(recordings, k=0.01)
    assert len(analysis["phase_bin_centres"]) == PHASE_BINS
    assert len(analysis["phase_density"]) == PHASE_BINS
