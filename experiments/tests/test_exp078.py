"""Focused tests for exp078's registered pairing and locking contract."""

from __future__ import annotations

import numpy as np
from experiments import exp078


def _trial(frequency_a: float, frequency_b: float) -> dict:
    return {
        "valid": True,
        "frequency_a_hz": frequency_a,
        "frequency_b_hz": frequency_b,
    }


def test_input_streams_are_deterministic_private_and_paired_across_coupling():
    first, ledger = exp078.make_inputs(90.0, 110.0, detuning_index=2, trials=2)
    repeated, repeated_ledger = exp078.make_inputs(
        90.0, 110.0, detuning_index=2, trials=2
    )

    assert ledger == repeated_ledger
    np.testing.assert_array_equal(first["drive_a"], repeated["drive_a"])
    np.testing.assert_array_equal(first["drive_b"], repeated["drive_b"])
    assert not np.array_equal(first["drive_a"], first["drive_b"])
    assert ledger[0]["drive_a_seed"] != ledger[0]["drive_b_seed"]


def test_coupled_trials_receive_their_paired_k_zero_natural_detuning():
    rows = [
        {
            "detuning_index": index,
            "coupling": coupling,
            "trials": [_trial(40.0 + index, 41.0), _trial(42.0 + index, 40.0)],
        }
        for index in range(len(exp078.TARGET_DETUNINGS_HZ))
        for coupling in (0.0, 0.08)
    ]
    joined = exp078.attach_natural_detunings(rows)

    for row in joined:
        assert [trial["natural_detuning_hz"] for trial in row["trials"]] == [
            -1.0 + row["detuning_index"],
            2.0 + row["detuning_index"],
        ]
        assert row["measured_detuning_hz"] == 0.5 + row["detuning_index"]


def test_locking_requires_every_registered_estimator():
    tolerances = {
        "frequency_difference_hz": 0.8,
        "absolute_phase_slope_rad_s": 3.0,
        "phase_slips": 1,
    }
    trial = {
        "valid": True,
        "frequency_difference_hz": 0.8,
        "phase_slope_rad_s": -3.0,
        "phase_slips": 1,
    }
    assert exp078.classify_trial(trial, tolerances)
    for field, invalid_value in (
        ("frequency_difference_hz", 0.81),
        ("phase_slope_rad_s", 3.01),
        ("phase_slips", 2),
    ):
        changed = dict(trial, **{field: invalid_value})
        assert not exp078.classify_trial(changed, tolerances)
    assert not exp078.classify_trial(dict(trial, valid=False), tolerances)


def test_authored_graph_retains_all_cross_circuit_projections_at_zero_weight():
    bundle = exp078.author_network(coupling=0.0)
    ids = {row["id"] for row in bundle.graph["projections"]}
    assert {
        "a_E_to_b_E",
        "a_E_to_b_I",
        "b_E_to_a_E",
        "b_E_to_a_I",
    } <= ids
    parameters = {row["id"]: row for row in bundle.graph["parameters"]}
    for projection in ("a_E_to_b_E", "a_E_to_b_I", "b_E_to_a_E", "b_E_to_a_I"):
        assert parameters[f"{projection}.weight"]["initializer"]["value"] == 0.0


def test_finite_size_followup_is_the_registered_mirrored_boundary_panel():
    panel = {
        name: row for name, row in exp078.FOLLOWUP_JOBS.items()
        if name != "benchmark"
    }
    assert set(panel) == {
        "m1_k000", "m1_k016", "m1_k024",
        "p1_k000", "p1_k016", "p1_k024",
    }
    assert {row["target_detuning_hz"] for row in panel.values()} == {-1.0, 1.0}
    assert {row["coupling"] for row in panel.values()} == {0.0, 0.016, 0.024}
    assert exp078.FOLLOWUP_JOBS["benchmark"] == panel["m1_k016"]
