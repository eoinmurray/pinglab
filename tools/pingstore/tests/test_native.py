from __future__ import annotations

import json
from pathlib import Path

import pytest
from pingstore.catalogue import Catalogue
from pingstore.contracts import PingstoreError
from pingstore.native import capture_failed_local_run, capture_local_run


def test_capture_local_run_is_immutable_and_becomes_official(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "writings").mkdir(parents=True)
    (repo / "writings/exp001.typ").write_text('collection: "demo",\n')
    staging = repo / ".artifacts/exp001.staging"
    staging.mkdir(parents=True)
    (staging / "numbers.json").write_text("{}\n")
    (staging / "_manifest.json").write_text(
        json.dumps(
            {
                "run_id": "r001",
                "run_at": "2026-08-24T00:00:00Z",
                "git_sha": "abc123",
                "dirty": True,
                "code_dirty": True,
                "patch": {"file": "_dirty.patch"},
                "host": "local",
            }
        )
    )
    root = tmp_path / "store"
    run = capture_local_run(repo, "exp001", staging, root=root)
    assert run["run_id"] == "exp001/r001"
    assert run["disposition"] == "candidate"
    assert Path(run["payload"]["location"], "numbers.json").is_file()
    dataset = Catalogue(root).load_dataset("demo")
    assert dataset["runs"] == {"exp001": ["exp001/r001"]}
    assert dataset["official_runs"] == {"exp001": "exp001/r001"}
    with pytest.raises(PingstoreError, match="already exists"):
        capture_local_run(repo, "exp001", staging, root=root)


def test_failed_run_is_retained_without_replacing_official(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "writings").mkdir(parents=True)
    (repo / "writings/exp001.typ").write_text('collection: "demo",\n')
    staging = repo / "failed"
    staging.mkdir()
    (staging / "output.log").write_text("boom\n")
    root = tmp_path / "store"
    catalogue = Catalogue(root)
    catalogue.create_dataset("demo", ["exp001"])
    failed = capture_failed_local_run(repo, "exp001", staging, root=root)
    assert failed["status"] == "failed"
    assert Path(failed["payload"]["location"], "output.log").is_file()
    dataset = catalogue.load_dataset("demo")
    assert dataset["runs"]["exp001"] == [failed["run_id"]]
    assert dataset["official_runs"] == {}
