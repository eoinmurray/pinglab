from __future__ import annotations

import json
from pathlib import Path

import pytest
from pingstore.contracts import (
    PingstoreError,
    load_json,
    validate_collections,
    validate_run,
)
from pingstore.materialize import materialize_run, materialize_view
from pingstore.native import (
    capture_campaign_metadata,
    capture_failed_local_run,
    capture_local_run,
    execution_origin,
    finalize_local_run,
)


def _repo(tmp_path: Path) -> Path:
    registry = tmp_path / "experiments/collections/registry.json"
    registry.parent.mkdir(parents=True)
    registry.write_text(
        json.dumps(
            {
                "schema": "pingstore.experiment-registry/v1",
                "experiments": {"exp001": "demo"},
                "historical": {},
            }
        )
    )
    return tmp_path


def _staging(tmp_path: Path, run_id: str = "r001") -> Path:
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "_manifest.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "run_at": "2026-08-26T12:00:00+00:00",
                "host": "local",
                "git_sha": "abc123",
                "scale": {"samples": 2},
            }
        )
    )
    (staging / "result.svg").write_text("<svg/>")
    (staging / "state.npz").write_bytes(b"raw")
    return staging


def test_local_capture_is_flat_complete_and_immutable(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    staging = _staging(tmp_path)
    state = tmp_path / "state"
    state.mkdir()
    (state / "weights.pt").write_bytes(b"weights")
    run = capture_local_run(repo, "exp001", staging, state=state)
    assert run["run_id"] == "exp001-r001-local"
    root = repo / ".pingstore/runs/exp001-r001-local"
    assert sorted(path.name for path in root.iterdir()) == ["files", "run.json"]
    assert (root / "files/result.svg").read_text() == "<svg/>"
    assert (root / "files/state/weights.pt").read_bytes() == b"weights"
    assert validate_run(load_json(root / "run.json")) == run
    with pytest.raises(PingstoreError, match="already exists"):
        capture_local_run(repo, "exp001", staging, state=state)


def test_failed_capture_remains_hidden(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    destination = capture_failed_local_run(repo, "exp001", _staging(tmp_path))
    assert destination.name.startswith(".exp001-r001-failed-")
    assert destination.name.endswith("-local.tmp")
    assert (destination / "files/result.svg").is_file()
    assert not (destination / "run.json").exists()


def test_direct_working_run_is_finalized_in_place(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    temporary = repo / ".pingstore/runs/.exp001-r001-local.tmp"
    files = temporary / "files"
    files.mkdir(parents=True)
    (files / "_manifest.json").write_text(
        json.dumps(
            {
                "run_id": "r001",
                "run_at": "2026-08-26T12:00:00+00:00",
                "host": "local",
                "git_sha": "abc123",
                "scale": {"samples": 2},
            }
        )
    )
    (files / "state").mkdir()
    (files / "state/checkpoint.pt").write_bytes(b"weights")
    (files / "result.svg").write_text("<svg/>")

    run = finalize_local_run(repo, "exp001", temporary)

    destination = repo / ".pingstore/runs/exp001-r001-local"
    assert not temporary.exists()
    assert destination.is_dir()
    assert (destination / "files/state/checkpoint.pt").read_bytes() == b"weights"
    assert (destination / "files/result.svg").read_text() == "<svg/>"
    assert load_json(destination / "run.json") == run


def test_slurm_origin_contains_cluster_and_job(monkeypatch) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "48291")
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "Wilkes 3")
    assert execution_origin() == "slurm-wilkes-3-48291"


def test_campaign_capture_uses_same_flat_run_layout(
    tmp_path: Path, monkeypatch
) -> None:
    derived = tmp_path / "derived/exp001"
    derived.mkdir(parents=True)
    (derived / "numbers.json").write_text("{}\n")
    monkeypatch.setenv("SLURM_JOB_ID", "48291")
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "wilkes")
    plan = {
        "campaign_id": "paper-1",
        "collection": "demo",
        "source": {"git_commit": "abc123"},
        "stages": [
            {
                "experiments": [
                    {
                        "slug": "exp001",
                        "paths": {"derived": str(derived)},
                        "command": ["python", "experiments/exp001.py"],
                    }
                ]
            }
        ],
    }
    result = capture_campaign_metadata(tmp_path, plan)
    run_id = "exp001-paper-1-slurm-wilkes-48291"
    assert result["runs"] == [run_id]
    run = tmp_path / ".pingstore/runs" / run_id
    assert (run / "run.json").is_file()
    assert (run / "files/numbers.json").is_file()


def test_manual_view_materializes_one_run_per_experiment(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    run = capture_local_run(repo, "exp001", _staging(tmp_path))
    collections = {"demo/latest": [run["run_id"]]}
    validate_collections(collections)
    (repo / ".pingstore/collections.json").write_text(json.dumps(collections))

    active = repo / "active"
    materialize_run(repo / ".pingstore", run["run_id"], active)
    assert (active / "exp001/result.svg").is_file()
    assert not (active / "exp001/state.npz").exists()

    view = repo / "view"
    materialize_view(repo / ".pingstore", "demo/latest", view)
    assert (view / "exp001/result.svg").is_file()
    assert not (view / "exp001/state.npz").exists()
