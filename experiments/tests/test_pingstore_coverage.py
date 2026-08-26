from __future__ import annotations

from pathlib import Path

from pingstore.registry import coverage

REPO = Path(__file__).resolve().parents[2]


def test_every_runnable_experiment_has_membership_and_capture_route() -> None:
    result = coverage(REPO)
    assert result["missing_membership"] == []
    assert result["stale_membership"] == []
    assert result["missing_capture"] == []
    assert result["writing_mismatches"] == {}
    assert result["passed"] is True


def test_historical_experiment_dispositions_are_explicit() -> None:
    historical = coverage(REPO)["historical"]
    assert historical["exp087"]["disposition"] == "removed-and-pruned"
    assert historical["exp094"]["disposition"] == "removed-and-pruned"
    assert historical["exp096"]["disposition"] == "removed-and-pruned"
