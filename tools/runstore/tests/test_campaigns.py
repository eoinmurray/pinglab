from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.runstore.archive import archive_run
from tools.runstore.campaigns import (
    activate_campaign,
    catalogue,
    current_view,
    resolve_local_campaign,
)
from tools.runstore.cli import main as runstore_main
from tools.runstore.contract import inventory_payload, write_json_atomic
from tools.runstore.lifecycle import initialize_run
from tools.runstore.storage import LocalStore


def _complete_campaign(root: Path, campaign_id: str = "smoke-001") -> Path:
    root.parent.mkdir(parents=True, exist_ok=True)
    run = initialize_run(
        root,
        run_id=campaign_id,
        kind="campaign",
        experiment=None,
        collection="gamma-gated-sparsity",
        command=["collection", "run"],
        repository=root.parent,
    )
    run["status"] = "complete"
    write_json_atomic(root / "run.json", run)
    for slug in ("exp022", "exp082"):
        derived = root / "derived/artifacts/data" / slug
        derived.mkdir(parents=True, exist_ok=True)
        (derived / "numbers.json").write_text('{"passed": true}\n')
        (derived / "figure.svg").write_text(f'<svg id="{slug}"></svg>\n')
    write_json_atomic(
        root / "collection-plan.json",
        {
            "profile": "smoke",
            "campaign_id": campaign_id,
            "stages": [
                {"index": 0, "experiments": [{"slug": "exp022"}]},
                {"index": 1, "experiments": [{"slug": "exp082"}]},
            ],
        },
    )
    inventory = inventory_payload(root, run_id=campaign_id)
    write_json_atomic(root / "inventory.json", inventory)
    return root


def test_catalogue_merges_local_and_archive_locations(tmp_path: Path) -> None:
    local_root = _complete_campaign(tmp_path / "campaigns" / "smoke-001")
    store = LocalStore(tmp_path / "store/gold-star")
    archive_run(local_root, "smoke-001", store)

    rows = catalogue([tmp_path / "campaigns"], store, active_campaign_id="smoke-001")

    assert len(rows) == 1
    assert rows[0]["campaign_id"] == "smoke-001"
    assert rows[0]["locations"] == ["local", "r2"]
    assert rows[0]["store_key"] == "smoke-001"
    assert rows[0]["profile"] == "smoke"
    assert rows[0]["active"] is True
    assert resolve_local_campaign("smoke-001", [tmp_path / "campaigns"]) == local_root


def test_activate_swaps_whole_view_and_preserves_unrelated_data(tmp_path: Path) -> None:
    campaign = _complete_campaign(tmp_path / "campaign")
    repaired_commit = "d" * 40
    numbers = campaign / "derived/artifacts/data/exp082/numbers.json"
    numbers.write_text(
        json.dumps(
            {
                "passed": True,
                "collection_provenance": {
                    "campaign_id": "smoke-001",
                    "source_git_commit": repaired_commit,
                },
            }
        )
        + "\n"
    )
    write_json_atomic(
        campaign / "inventory.json",
        inventory_payload(campaign, run_id="smoke-001"),
    )
    artifacts = tmp_path / "artifacts" / "data"
    unrelated = artifacts / "exp999"
    unrelated.mkdir(parents=True)
    (unrelated / "keep.txt").write_text("keep")
    old = artifacts / "exp022"
    old.mkdir()
    (old / "obsolete.txt").write_text("old")

    result = activate_campaign(
        campaign,
        artifacts_root=artifacts,
        activated_at_utc="2026-08-13T12:00:00Z",
    )

    assert result["campaign_id"] == "smoke-001"
    assert result["experiments"] == ["exp022", "exp082"]
    assert (unrelated / "keep.txt").read_text() == "keep"
    assert not (artifacts / "exp022" / "obsolete.txt").exists()
    assert (
        json.loads((artifacts / "exp022/_provenance.json").read_text())["campaign_id"]
        == "smoke-001"
    )
    assert json.loads((artifacts / "exp082/_provenance.json").read_text())[
        "generating_git_commit"
    ] == repaired_commit
    assert current_view(artifacts)["valid"] is True


def test_current_detects_mixed_or_modified_view(tmp_path: Path) -> None:
    campaign = _complete_campaign(tmp_path / "campaign")
    artifacts = tmp_path / "artifacts/data"
    activate_campaign(campaign, artifacts_root=artifacts)
    (artifacts / "exp082/figure.svg").write_text("changed")

    result = current_view(artifacts)

    assert result["valid"] is False
    assert any("exp082/figure.svg" in error for error in result["errors"])


def test_campaign_cli_list_activate_and_current(tmp_path: Path, capsys) -> None:
    campaigns = tmp_path / "campaigns"
    campaign = _complete_campaign(campaigns / "smoke-001")
    artifacts = tmp_path / "artifacts/data"

    with pytest.raises(SystemExit, match="0"):
        runstore_main(
            [
                "campaigns",
                "--local-only",
                "--local-root",
                str(campaigns),
                "--artifacts-root",
                str(artifacts),
            ]
        )
    assert "smoke-001" in capsys.readouterr().out

    with pytest.raises(SystemExit, match="0"):
        runstore_main(["activate", str(campaign), "--artifacts-root", str(artifacts)])
    capsys.readouterr()

    with pytest.raises(SystemExit, match="0"):
        runstore_main(["current", "--artifacts-root", str(artifacts)])
    assert "valid" in capsys.readouterr().out
