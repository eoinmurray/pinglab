from __future__ import annotations

import json
from pathlib import Path

import pytest
from pingstore.contracts import PingstoreError
from pingstore.membership import load_history, membership, memberships


def _writing(repo: Path, experiment: str, collection: str) -> None:
    path = repo / "writings" / f"{experiment}.typ"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'#let metadata = (collection: "{collection}",)\n')


def test_memberships_come_from_writing_metadata(tmp_path: Path) -> None:
    _writing(tmp_path, "exp001", "demo")
    _writing(tmp_path, "exp002", "gamma-gated-sparsity")
    assert memberships(tmp_path) == {
        "exp001": "demo",
        "exp002": "gamma-gated-sparsity",
    }
    assert membership(tmp_path, "exp002") == "gamma-gated-sparsity"


def test_missing_active_membership_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(PingstoreError, match="must declare collection metadata"):
        membership(tmp_path, "exp001")


def test_history_is_separate_and_validated(tmp_path: Path) -> None:
    history = tmp_path / "experiments/history.json"
    history.parent.mkdir(parents=True)
    history.write_text(
        json.dumps(
            {
                "schema": "pinglab.experiment-history/v1",
                "historical": {
                    "exp009": {
                        "collection": "demo",
                        "disposition": "removed-and-pruned",
                        "evidence": "fixture",
                    }
                },
            }
        )
    )
    assert load_history(tmp_path)["exp009"]["disposition"] == "removed-and-pruned"
