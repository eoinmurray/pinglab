from __future__ import annotations

import numpy as np
from experiments import exp083, exp084


def test_protocol_is_one_dimensional_and_contains_default():
    assert exp084.INPUT_RATE_HZ == 100.0
    assert exp084.TAU_GABA_MS == (2.0, 4.0, 6.0, 9.0, 12.0, 16.0)
    assert exp084.REPRESENTATIVE_TAU_MS == (2.0, 9.0, 16.0)
    assert exp084.SCALE["network_seed"] == exp083.NETWORK_SEED


def test_authored_graph_records_requested_inhibitory_decay():
    bundle = exp084.author_network(4.0)
    projection = next(
        row for row in bundle.graph["projections"] if row["id"] == "ping_I_to_E"
    )
    assert projection["synapse"]["kind"] == "gaba"
    assert projection["synapse"]["tau"] == {"value": 4.0, "unit": "ms"}


def test_tau_changes_only_the_inhibitory_synapse_policy():
    fast = exp084.author_network(2.0).graph
    slow = exp084.author_network(16.0).graph
    fast_projection = next(row for row in fast["projections"] if row["id"] == "ping_I_to_E")
    slow_projection = next(row for row in slow["projections"] if row["id"] == "ping_I_to_E")
    fast_projection["synapse"]["tau"] = slow_projection["synapse"]["tau"]
    assert fast == slow


def test_summary_preserves_tau_and_standard_metrics():
    steps = round(exp083.T_MS / exp083.DT_MS)
    e = np.zeros((steps, len(exp083.TRIAL_SEEDS), exp083.N_E), dtype=np.uint8)
    i = np.zeros((steps, len(exp083.TRIAL_SEEDS), exp083.N_I), dtype=np.uint8)
    estimate = exp083.estimate_gamma_from_raster(
        e, dt_ms=exp083.DT_MS, config=exp083.FREQUENCY_CONFIG
    )
    summary = exp084.summarize_condition(9.0, e, i, estimate)
    assert summary["tau_gaba_ms"] == 9.0
    assert summary["rhythmicity_score_median"] == 0.0
    assert summary["rhythm_frequency_median_hz"] is None
