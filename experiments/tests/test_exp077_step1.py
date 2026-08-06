"""Focused unit tests for exp077 Step 1."""

from __future__ import annotations

import math

import numpy as np
import pytest
from experiments import exp077


def test_encoder_is_bounded_and_deterministic() -> None:
    pixels = np.asarray([0.0, 0.5, 1.0])
    first = exp077.encode_poisson(pixels, 25.0, np.random.default_rng(42))
    second = exp077.encode_poisson(pixels, 25.0, np.random.default_rng(42))
    assert np.array_equal(first, second)
    assert not first[:, 0].any()
    with pytest.raises(ValueError):
        exp077.encode_poisson(np.asarray([1.01]), 25.0, np.random.default_rng(42))
    with pytest.raises(ValueError):
        exp077.encode_poisson(np.asarray([1.0]), 20_000.0, np.random.default_rng(42))


def test_decay_then_add_and_exponential_euler_first_step() -> None:
    result = exp077.probe_spikes(exp077.single_spike_train(0.0))
    conductance = np.asarray(result["conductance_uS"])
    voltage = np.asarray(result["voltage_mV"])
    assert conductance[0] == exp077.PROBE_US
    assert conductance[1] == pytest.approx(
        exp077.PROBE_US * math.exp(-exp077.DT_MS / exp077.PARAMETERS["tau_ampa_ms"]),
        abs=1e-14,
    )
    g_total = exp077.PARAMETERS["g_L_uS"] + exp077.PROBE_US
    v_inf = (
        exp077.PARAMETERS["g_L_uS"] * exp077.PARAMETERS["E_L_mV"]
        + exp077.PROBE_US * exp077.PARAMETERS["E_e_mV"]
    ) / g_total
    expected = v_inf + (exp077.PARAMETERS["E_L_mV"] - v_inf) * math.exp(
        -exp077.DT_MS * g_total / exp077.PARAMETERS["C_m_nF"]
    )
    assert voltage[0] == pytest.approx(expected, abs=1e-13)


def test_all_registered_step_1_validations_pass() -> None:
    validations = exp077.validate_probe()
    assert validations
    assert all(record["ok"] for record in validations.values())


def test_equal_count_timing_changes_feature() -> None:
    early = float(exp077.probe_spikes(exp077.single_spike_train(20.0))["z_mV"])
    late = float(exp077.probe_spikes(exp077.single_spike_train(180.0))["z_mV"])
    assert early > late > 0.0


def test_step_2_grid_is_deterministic_and_zero_intensity_rests() -> None:
    kwargs = {
        "intensities": (0, 128, 255),
        "rates_hz": (0.25, 25.0),
        "probes_uS": (0.6, 2.4),
        "draws": 4,
    }
    first = exp077.simulate_condition_grid(**kwargs, rng=exp077._step2_rng(42, 99))
    second = exp077.simulate_condition_grid(**kwargs, rng=exp077._step2_rng(42, 99))
    assert first.shape == (2, 2, 3, 4)
    assert first.dtype == np.float32
    assert np.array_equal(first, second)
    assert np.all(first[:, :, 0, :] == 0.0)
    assert np.all(np.isfinite(first))
    assert np.all((first >= 0.0) & (first <= 65.0))


def test_locked_step_2_outcome_stops_without_a_library() -> None:
    outcome = exp077.FIGURES / "step2_pilot_outcome.json"
    record = __import__("json").loads(outcome.read_text())
    assert record["candidate_K"] == [64, 128, 256, 512]
    assert record["hard_maximum_K"] == 512
    assert record["selected_K"] is None
    assert not record["passed"]
    assert all(not row["passed"] for row in record["trajectory"])


def test_authorized_step_2_extension_selects_2048() -> None:
    outcome = exp077.FIGURES / "step2_pilot_extension_outcome.json"
    record = __import__("json").loads(outcome.read_text())
    assert record["candidate_K"] == [1024, 2048]
    assert record["selected_K"] == 2048
    assert record["passed"]
    assert not record["trajectory"][0]["passed"]
    assert record["trajectory"][1]["passed"]


def test_full_library_contract_and_chunk_streams_are_locked() -> None:
    assert exp077.LIBRARY_K == 2048
    assert exp077.LIBRARY_SHAPE == (3, 3, 12, 256, 2048)
    assert exp077.LIBRARY_AXIS_ORDER == (
        "seed",
        "probe_uS",
        "rate_hz",
        "intensity",
        "draw",
    )
    assert exp077._library_chunks()[0] == (0, 0, 8)
    assert exp077._library_chunks()[-1] == (31, 248, 256)
    first = exp077._library_chunk_rng(42, 0).random(16)
    replay = exp077._library_chunk_rng(42, 0).random(16)
    other_seed = exp077._library_chunk_rng(43, 0).random(16)
    other_chunk = exp077._library_chunk_rng(42, 1).random(16)
    assert np.array_equal(first, replay)
    assert not np.array_equal(first, other_seed)
    assert not np.array_equal(first, other_chunk)


def test_linear_filter_zero_drive_and_grid_convergence() -> None:
    lambdas = np.asarray([0.0, 0.25, 3.0, 25.0])
    probes = np.full_like(lambdas, 1.2)
    coarse = exp077.predicted_linear_variance(
        lambdas, probes, grid_points=2049
    )
    fine = exp077.predicted_linear_variance(
        lambdas, probes, grid_points=4097
    )
    assert coarse[0] == fine[0] == 0.0
    assert np.all(np.isfinite(fine))
    assert np.all(fine[1:] > 0.0)
    assert np.max(np.abs(coarse[1:] - fine[1:]) / fine[1:]) < 1e-4


def test_numerical_gain_matches_registered_transfer() -> None:
    rate_hz = 3.0
    probe_uS = 1.2
    frequency_hz = 10.0
    numerical = exp077.numerical_sinusoidal_gain(rate_hz, probe_uS, frequency_hz)
    mean_g, mean_v = exp077.linear_operating_point(rate_hz, probe_uS)
    omega = 2.0 * np.pi * frequency_hz / 1000.0
    analytical = abs(
        probe_uS / (1j * omega + 1 / exp077.PARAMETERS["tau_ampa_ms"])
        * (exp077.PARAMETERS["E_e_mV"] - mean_v)
        / (
            1j * omega * exp077.PARAMETERS["C_m_nF"]
            + exp077.PARAMETERS["g_L_uS"]
            + mean_g
        )
    ) / 1000.0
    assert numerical == pytest.approx(analytical, rel=exp077.GAIN_REL_TOL)


def test_step_4_streams_replay_and_are_independent() -> None:
    first = exp077._step4_rng(1, 0, 0, 0, 0).integers(0, 2048, size=784)
    replay = exp077._step4_rng(1, 0, 0, 0, 0).integers(0, 2048, size=784)
    other_pixel_stream = exp077._step4_rng(1, 0, 0, 0, 1).integers(
        0, 2048, size=784
    )
    direct_stream = exp077._step4_rng(2, 0, 0, 0, 0).integers(
        0, 2048, size=784
    )
    assert np.array_equal(first, replay)
    assert not np.array_equal(first, other_pixel_stream)
    assert not np.array_equal(first, direct_stream)


def test_step_4_comparison_passes_identical_features() -> None:
    rng = np.random.default_rng(42)
    values = rng.uniform(0.0, 10.0, size=(16, 8, 784))
    record = exp077.compare_feature_condition(values, values.copy(), "high")
    assert record["passed"]
    assert all(record["checks"].values())


def test_direct_black_image_stays_at_rest() -> None:
    features = exp077.direct_feature_replicates(
        np.zeros((28, 28), dtype=np.uint8), 25.0, 1.2, 1, 11, 0
    )
    assert features.shape == (exp077.STEP4_REPLICATES, 784)
    assert np.all(features == 0.0)


def test_torch_direct_features_replay_and_fresh_draw() -> None:
    import torch

    images = torch.full((2, 28, 28), 255, dtype=torch.uint8)
    rates = torch.tensor([3.0, 25.0])
    first_generator = torch.Generator().manual_seed(123)
    replay_generator = torch.Generator().manual_seed(123)
    fresh_generator = torch.Generator().manual_seed(124)
    first = exp077.direct_feature_batch_torch(images, rates, 1.2, first_generator)
    replay = exp077.direct_feature_batch_torch(images, rates, 1.2, replay_generator)
    fresh = exp077.direct_feature_batch_torch(images, rates, 1.2, fresh_generator)
    assert torch.equal(first, replay)
    assert not torch.equal(first, fresh)
    assert torch.isfinite(first).all()
    assert torch.all((first >= 0.0) & (first <= 65.0))


def test_decoder_partitions_are_disjoint() -> None:
    train = set(range(*exp077.TRAIN_INDICES))
    validation = set(range(*exp077.VALIDATION_INDICES))
    assert train.isdisjoint(validation)
    assert len(train) == 55_000
    assert len(validation) == 5_000


def test_registered_rate_sampler_can_reach_every_rate() -> None:
    assert exp077.DECODER_RATES_HZ == (
        0.01,
        0.05,
        0.1,
        0.25,
        0.5,
        0.75,
        1.0,
        1.5,
        2.0,
        2.5,
        3.0,
        4.0,
        5.0,
        10.0,
        25.0,
    )
    assert exp077.TRAINING_RATES_HZ == exp077.DECODER_RATES_HZ[3:]
    rng = np.random.default_rng(exp077._decoder_seed(42, 1, 2))
    positions = rng.integers(0, len(exp077.DECODER_RATES_HZ), 10_000)
    assert set(positions) == set(range(len(exp077.DECODER_RATES_HZ)))


def test_expanded_rate_protocol_matches_decoder_grid() -> None:
    protocol = exp077.verify_expanded_rate_training_protocol()
    assert protocol["decoder_rate_grid_hz"] == list(exp077.DECODER_RATES_HZ)
    assert protocol["training_rate_distribution"]["probability_per_rate"] == pytest.approx(
        1.0 / len(exp077.DECODER_RATES_HZ)
    )


def test_expanded_rate_protocol_remote_payload(monkeypatch, tmp_path) -> None:
    protocol = (exp077.FIGURES / "expanded_rate_training_protocol.json").read_text()
    monkeypatch.setattr(exp077, "FIGURES", tmp_path)
    monkeypatch.setenv("EXP077_FROZEN_TRAINING_PROTOCOL_JSON", protocol)
    assert exp077.verify_expanded_rate_training_protocol()["decoder_rate_grid_hz"] == list(
        exp077.DECODER_RATES_HZ
    )


def test_held_out_loader_fails_closed_without_protocol(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="frozen evaluation protocol"):
        exp077.load_held_out_mnist_test(tmp_path / "absent.json")


def test_hierarchical_interval_is_deterministic(monkeypatch) -> None:
    monkeypatch.setattr(exp077, "BOOTSTRAP_REPETITIONS_HELDOUT", 20)
    values = np.zeros((3, 3, 40), dtype=np.bool_)
    values[:, :, :30] = True
    first = exp077._hierarchical_lower_bound(values, 123)
    replay = exp077._hierarchical_lower_bound(values, 123)
    assert first == replay
    assert first[0] == 0.75
    assert first[1] <= first[0] <= first[2]
