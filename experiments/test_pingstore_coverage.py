from __future__ import annotations

from pathlib import Path

from pingstore.membership import coverage

REPO = Path(__file__).resolve().parents[1]


def test_every_runnable_experiment_has_membership_and_capture_route() -> None:
    result = coverage(REPO)
    assert result["missing_membership"] == []
    assert result["stale_membership"] == []
    assert result["missing_capture"] == []
    assert result["passed"] is True
    assert set(result["capture_routes"]) == set(result["registered"])
    assert result["capture_routes"]["exp024"] == "independent-stages"
