from __future__ import annotations

import numpy as np
from experiments import exp094


def test_gold_star_checkpoint_contract_is_final_epoch_and_hash_pinned() -> None:
    assert exp094.GOLD_STAR_PUBLICATION == "ggs-production-composite-20260821-6d9c38eb"
    assert exp094.UPSTREAM_CAMPAIGN == "ggs-fr-repair-20260820-ac6f4988"
    assert exp094.CHECKPOINT_ROLE == "final_epoch"
    assert set(exp094.EXPECTED_CHECKPOINTS) == {"coba", "ping"}
    assert exp094.TRAINING_ROOT == (
        exp094.REPO
        / "runs"
        / "restored"
        / "gold-2"
        / "state"
        / "checkpoints"
        / "current-repair-exp022"
        / "cells"
    )


def test_cumulative_and_window_counts_have_declared_memory() -> None:
    spikes = np.zeros((5, 2), dtype=float)
    spikes[0, 0] = 1
    spikes[2, 1] = 1
    spikes[4, 0] = 1

    cumulative = exp094.cumulative_count(spikes)
    window = exp094.window_count(spikes, 2)

    np.testing.assert_array_equal(cumulative[-1], [2, 1])
    np.testing.assert_array_equal(window[1], [1, 0])
    np.testing.assert_array_equal(window[2], [0, 1])
    np.testing.assert_array_equal(window[-1], [1, 0])


def test_leaky_count_adds_spikes_and_forgets_between_them() -> None:
    spikes = np.zeros((4, 1), dtype=float)
    spikes[0, 0] = 1
    result = exp094.leaky_count(spikes, 0.5)
    np.testing.assert_allclose(result[:, 0], [1.0, 0.5, 0.25, 0.125])


def test_cycle_votes_discard_within_bin_margin() -> None:
    spikes = np.zeros((6, 2), dtype=float)
    spikes[0:3, 0] = 1
    spikes[3:5, 1] = 1
    votes = exp094.cumulative_bin_votes(spikes, np.array([0, 3, 6]))
    np.testing.assert_array_equal(votes[2], [1, 0])
    np.testing.assert_array_equal(votes[-1], [1, 1])


def test_softmax_temperature_changes_sharpness_not_winner() -> None:
    counts = np.array([[0.0, 0.0], [3.0, 1.0]])
    ordinary = exp094.softmax(counts)
    softened = exp094.softmax(counts, temperature=4.0)
    assert ordinary[-1].argmax() == softened[-1].argmax() == 0
    assert softened[-1, 0] < ordinary[-1, 0]
    np.testing.assert_allclose(ordinary.sum(axis=1), 1.0)
    np.testing.assert_allclose(softened.sum(axis=1), 1.0)


def test_sigmoid_is_independent_and_can_raise_multiple_classes() -> None:
    values = exp094.sigmoid(np.array([[4.0, 3.0]]))
    assert np.all(values > 0.9)
    assert values.sum() > 1.0


def test_pre_reset_voltage_uses_recorded_output_reset() -> None:
    e = np.array([[1.0], [0.0]])
    out = np.array([[1.0], [0.0]])
    weights = np.array([[1.0]])
    result = exp094.replay_pre_reset_voltage(e, out, weights, 1.0, 1.0)
    beta = np.exp(-1.0)
    first = 1.0 - beta
    np.testing.assert_allclose(result[:, 0], [first, beta * (first - 1.0)])


def test_balanced_screen_uses_equal_outcome_blind_class_counts() -> None:
    labels = np.repeat(np.arange(10), 3)
    indices = exp094.balanced_test_indices(labels, per_class=2)
    np.testing.assert_array_equal(np.bincount(labels[indices]), np.full(10, 2))
    np.testing.assert_array_equal(indices[:2], [0, 1])


def test_screening_summary_tracks_native_decision_transitions() -> None:
    labels = np.arange(10)
    coba_native = np.array([0, 1, 9, 9, 4, 5, 6, 7, 8, 9])
    ping_native = np.array([0, 9, 2, 9, 4, 5, 6, 7, 8, 9])
    predictions = {
        "coba": {
            name: coba_native.copy() for name in exp094.DECODER_ORDER
        },
        "ping": {
            name: ping_native.copy() for name in exp094.DECODER_ORDER
        },
    }
    predictions["coba"]["cumulative"] = ping_native.copy()
    summary = exp094.screening_summary(labels, predictions)
    transitions = summary["models"]["coba"]["cumulative"]["transitions"]
    assert transitions == {
        "correct_to_correct": 7,
        "correct_to_wrong": 1,
        "wrong_to_correct": 1,
        "wrong_to_wrong": 1,
    }
