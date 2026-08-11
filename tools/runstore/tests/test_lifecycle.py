from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.runstore.contract import (
    ContractError,
    inventory_payload,
    write_json_atomic,
)
from tools.runstore.lifecycle import initialize_run
from tools.runstore.promotion import promote_experiment


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _complete_adhoc_run(root: Path, *, archived: bool = False) -> Path:
    run = initialize_run(
        root,
        run_id="exp123-test-20260811T000000Z",
        kind="adhoc",
        experiment="exp123",
        collection=None,
        command=["uv", "run", "python", "experiments/exp123.py"],
        repository=root.parent,
    )
    run["status"] = "complete"
    if archived:
        run["archive"] = {
            "archive_id": "test-archive",
            "uri": "r2://pinglab/campaigns/adhoc/test-archive",
        }
    write_json_atomic(root / "run.json", run)
    source = root / "derived" / "artifacts" / "data" / "exp123"
    (source / "numbers.json").write_text('{"accuracy": 0.9}\n')
    (source / "figure.svg").write_text("<svg></svg>\n")
    inventory = inventory_payload(root, run_id=run["run_id"])
    write_json_atomic(root / "inventory.json", inventory)
    return source


def test_init_creates_adhoc_layout_and_refuses_existing_root(tmp_path: Path) -> None:
    root = tmp_path / "run"
    manifest = initialize_run(
        root,
        run_id="exp123-test",
        kind="adhoc",
        experiment="exp123",
        collection=None,
        command=["experiment", "--out-dir", str(root)],
        repository=tmp_path,
    )

    assert manifest["kind"] == "adhoc"
    assert (root / "state").is_dir()
    assert (root / "derived/artifacts/data/exp123").is_dir()
    assert (root / "logs").is_dir()
    assert not (root / "inventory.json").exists()
    with pytest.raises(ContractError, match="already exists"):
        initialize_run(
            root,
            run_id="another-run",
            kind="adhoc",
            experiment="exp123",
            collection=None,
            command=["experiment"],
            repository=tmp_path,
        )


def test_init_creates_campaign_layout(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    initialize_run(
        root,
        run_id="campaign-test",
        kind="campaign",
        experiment=None,
        collection="gamma-gated-sparsity",
        command=["campaign", "run"],
        repository=tmp_path,
    )

    assert (root / "exp022").is_dir()
    assert (root / "downstream").is_dir()
    assert (root / "derived/artifacts").is_dir()


def test_promotion_writes_reverse_provenance_without_changing_source(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    source = _complete_adhoc_run(root, archived=True)
    before = {item.name: _sha256(item) for item in source.iterdir()}

    result = promote_experiment(
        root,
        "exp123",
        artifacts_root=tmp_path / "published",
        promoted_at_utc="2026-08-11T01:02:03Z",
    )

    destination = tmp_path / "published" / "exp123"
    provenance = json.loads((destination / "_provenance.json").read_text())
    assert result["file_count"] == 2
    assert provenance["run_id"] == "exp123-test-20260811T000000Z"
    assert provenance["campaign_id"] is None
    assert provenance["archive"]["archive_id"] == "test-archive"
    assert provenance["source_directory"] == "derived/artifacts/data/exp123"
    assert provenance["promoted_at_utc"] == "2026-08-11T01:02:03Z"
    assert {row["path"] for row in provenance["files"]} == {
        "figure.svg",
        "numbers.json",
    }
    for row in provenance["files"]:
        assert _sha256(destination / row["path"]) == row["sha256"]
    assert {item.name: _sha256(item) for item in source.iterdir()} == before


def test_promotion_replaces_existing_view(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _complete_adhoc_run(root)
    existing = tmp_path / "published" / "exp123"
    existing.mkdir(parents=True)
    (existing / "obsolete.txt").write_text("old")

    promote_experiment(root, "exp123", artifacts_root=tmp_path / "published")

    assert not (existing / "obsolete.txt").exists()
    assert (existing / "_provenance.json").is_file()
    assert not list((tmp_path / "published").glob(".exp123.*"))


def test_promotion_rejects_incomplete_or_unaccepted_source(tmp_path: Path) -> None:
    root = tmp_path / "run"
    source = _complete_adhoc_run(root)
    (source / "figure.svg").unlink()
    run = json.loads((root / "run.json").read_text())
    inventory = inventory_payload(root, run_id=run["run_id"])
    write_json_atomic(root / "inventory.json", inventory)

    with pytest.raises(ContractError, match="figure"):
        promote_experiment(root, "exp123", artifacts_root=tmp_path / "published")


def test_promotion_rejects_planned_run(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _complete_adhoc_run(root)
    run = json.loads((root / "run.json").read_text())
    run["status"] = "planned"
    write_json_atomic(root / "run.json", run)

    with pytest.raises(ContractError, match="complete or legacy"):
        promote_experiment(root, "exp123", artifacts_root=tmp_path / "published")
