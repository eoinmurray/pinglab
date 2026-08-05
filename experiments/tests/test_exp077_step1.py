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
    first = exp077.simulate_condition_grid(
        **kwargs, rng=exp077._step2_rng(42, 99)
    )
    second = exp077.simulate_condition_grid(
        **kwargs, rng=exp077._step2_rng(42, 99)
    )
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


def test_step_3_remains_a_hard_stop() -> None:
    with pytest.raises(NotImplementedError, match="Step 3"):
        exp077.step_3()
