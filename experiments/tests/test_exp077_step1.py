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


def test_step_4_remains_a_hard_stop() -> None:
    with pytest.raises(NotImplementedError, match="Step 4"):
        exp077.step_4()
