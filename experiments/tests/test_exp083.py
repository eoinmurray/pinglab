from __future__ import annotations

import numpy as np
from experiments import exp083


def test_protocol_is_frozen_to_default_ping_gamma_policy():
    assert exp083.GAMMA_CONFIG.name == "default-ping-gamma-v1"
    assert exp083.GAMMA_CONFIG.band_hz == (30.0, 80.0)
    assert exp083.GAMMA_CONFIG.burn_ms == 200.0
    assert exp083.GAMMA_CONFIG.min_prominence_ratio == 3.0
    assert exp083.GAMMA_CONFIG.reject_band_edges


def test_input_trials_are_deterministic_and_paired_across_rates():
    low = exp083.make_inputs(25.0)
    repeated = exp083.make_inputs(25.0)
    high = exp083.make_inputs(150.0)
    np.testing.assert_array_equal(low, repeated)
    assert low.shape == (10_000, 5, 128)
    assert high.sum() > low.sum()


def test_authored_graph_is_the_untuned_default_ping_component():
    bundle = exp083.author_network()
    assert bundle.graph["name"] == "default_ping_drive_response"
    assert {row["id"] for row in bundle.graph["inputs"]} == {"drive"}
    assert {row["id"] for row in bundle.graph["populations"]} == {"ping_E", "ping_I"}
    assert len(bundle.graph["projections"]) == 3


def test_silent_condition_remains_unresolved():
    raster = np.zeros((10_000, 5, exp083.N_E), dtype=np.uint8)
    estimate = exp083.estimate_gamma_from_raster(
        raster,
        dt_ms=exp083.DT_MS,
        config=exp083.GAMMA_CONFIG,
    )
    assert not estimate.resolved
    assert estimate.frequency_hz is None
    assert {trial.reason for trial in estimate.trials} == {"insufficient_activity"}


def test_silent_condition_has_zero_rhythmicity_score():
    rows = [
        {
            "e_rate_hz": 0.0,
            "i_rate_hz": 0.0,
            "e_i_peak_lag_ms": None,
            "rhythmicity_contrast": None,
            "gamma": {
                "resolved": False,
                "frequency_hz": None,
                "prominence_ratio": None,
            },
        }
        for _ in exp083.TRIAL_SEEDS
    ]
    summary = exp083.summarize_condition(0.0, rows)
    assert summary["rhythmicity_score_median"] == 0.0
    assert summary["rhythmicity_score_iqr"] == 0.0


def test_periodic_volleys_have_strong_standard_rhythmicity_contrast():
    raster = np.zeros((10_000, exp083.N_E), dtype=np.uint8)
    raster[2_000::250] = 1
    score = exp083._rhythmicity_contrast(raster)
    assert score is not None
    assert score > 0.8
