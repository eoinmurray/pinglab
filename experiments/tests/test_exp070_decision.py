"""Focused checks for exp070's preregistered short-run promotion gate."""

from __future__ import annotations

import copy

from experiments import exp070


def candidate_cells() -> dict[str, dict[str, dict[str, list[dict[str, float | int]]]]]:
    cells = {}
    for model in exp070.baseline.MODELS:
        old = exp070.epoch_five_baseline(model)
        row: dict[str, float | int] = {
            "loss": 2.0,
            "test_loss": old["cross_entropy"] - 0.1,
            "acc": old["accuracy_pct"] + exp070.PROMOTION_ACCURACY_GAIN_PP,
            "grad_norm": 1.0,
            "test_rate_e": 10.0,
            "test_rate_i": 20.0 if model == "ping" else 0.0,
            "skipped_steps": 0,
            "nan_forward_batches": 0,
        }
        cells[model] = {"training": {"epochs": [copy.deepcopy(row) for _ in range(5)]}}
    return cells


def test_both_cells_must_clear_accuracy_and_loss_gate() -> None:
    cells = candidate_cells()
    status, diagnostics = exp070.attempt_decision(cells)
    assert status == "promoted"
    assert all(row["clears_promotion_gate"] for row in diagnostics.values())

    cells["coba"]["training"]["epochs"][4]["acc"] -= 0.01
    status, diagnostics = exp070.attempt_decision(cells)
    assert status == "killed"
    assert diagnostics["coba"]["clears_promotion_gate"] is False


def test_nonfinite_or_saturated_activity_kills_candidate() -> None:
    cells = candidate_cells()
    cells["ping"]["training"]["epochs"][4]["test_rate_i"] = 1000.0
    status, diagnostics = exp070.attempt_decision(cells)
    assert status == "killed"
    assert diagnostics["ping"]["clears_promotion_gate"] is False
