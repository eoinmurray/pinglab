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


def _presentation(repo: Path) -> dict:
    from pingstore.contracts import payload_digest, write_json_atomic

    identity = "exp001-r001-present"
    directory = repo / ".pingstore/runs" / identity
    (directory / "export").mkdir(parents=True)
    (directory / "README.md").write_text("# exp001 run\n")
    (directory / "export/result.svg").write_text("<svg/>")
    record = {
        "schema": "pingstore.run/v4", "run_id": identity, "experiment": "exp001",
        "collection": "demo", "origin": "local", "stage": "present", "inputs": {},
        "created_at": "2026-08-27T12:00:00Z", "execution": {}, "provenance": {},
        "payload_digest": payload_digest(directory),
    }
    write_json_atomic(directory / "run.json", {**record, "payload_digest": "sha256:" + "0" * 64})
    record["payload_digest"] = payload_digest(directory)
    write_json_atomic(directory / "run.json", record)
    return record


def test_local_capture_is_flat_complete_and_immutable(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    staging = _staging(tmp_path)
    state = tmp_path / "state"
    state.mkdir()
    (state / "weights.pt").write_bytes(b"weights")
    run = capture_local_run(repo, "exp001", staging, state=state)
    assert run["run_id"] == "exp001-r001-local"
    root = repo / ".pingstore/runs/exp001-r001-local"
    assert sorted(path.name for path in root.iterdir()) == ["README.md", "export", "presentation", "run.json"]
    assert (root / "presentation/result.svg").read_text() == "<svg/>"
    assert (root / "export/state/weights.pt").read_bytes() == b"weights"
    assert validate_run(load_json(root / "run.json")) == run
    with pytest.raises(PingstoreError, match="already exists"):
        capture_local_run(repo, "exp001", staging, state=state)


def test_failed_capture_remains_hidden(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    destination = capture_failed_local_run(repo, "exp001", _staging(tmp_path))
    assert destination.name.startswith(".exp001-r001-failed-")
    assert destination.name.endswith("-local.tmp")
    assert (destination / "presentation/result.svg").is_file()
    assert not (destination / "run.json").exists()


def test_direct_working_run_is_finalized_in_place(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    temporary = repo / ".pingstore/runs/.exp001-r001-local.tmp"
    files = temporary / "presentation"
    files.mkdir(parents=True)
    records = temporary / "export/provenance"
    records.mkdir(parents=True)
    (records / "_manifest.json").write_text(
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
    (temporary / "export/state").mkdir()
    (temporary / "export/state/checkpoint.pt").write_bytes(b"weights")
    (files / "result.svg").write_text("<svg/>")

    run = finalize_local_run(repo, "exp001", temporary)

    destination = repo / ".pingstore/runs/exp001-r001-local"
    assert not temporary.exists()
    assert destination.is_dir()
    assert (destination / "export/state/checkpoint.pt").read_bytes() == b"weights"
    assert (destination / "presentation/result.svg").read_text() == "<svg/>"
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
    assert (run / "presentation/numbers.json").is_file()


def test_manual_view_materializes_one_run_per_experiment(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    run = _presentation(repo)
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


def _v1_store(tmp_path: Path, *, relocated_readme: bool = False) -> Path:
    from pingstore.payload import inventory_payload

    store = tmp_path / "store"
    run = store / "runs/exp001-r001-local"
    files = run / "files"
    (files / "state").mkdir(parents=True)
    (files / "rasters").mkdir()
    (files / "state/weights.pth").write_bytes(b"weights")
    (files / "rasters/example.png").write_bytes(b"picture")
    (files / "numbers.json").write_text('{"measurement": 42}\n')
    if relocated_readme:
        (files / "README.md").write_text("# Original notes\n")
    digest = inventory_payload(files, run_id=run.name)["payload_digest"]
    if relocated_readme:
        (files / "README.md").rename(run / "README.md")
    (run / "run.json").write_text(json.dumps({
        "schema": "pingstore.run/v1", "run_id": run.name, "experiment": "exp001",
        "collection": "demo", "origin": "local", "created_at": "2026-08-27T00:00:00Z",
        "execution": {"command": ["original", "command"]},
        "provenance": {"original": True}, "files_digest": "sha256:" + digest,
    }))
    return store


@pytest.mark.parametrize("relocated_readme", [False, True])
def test_migration_preserves_all_bytes_and_keeps_rollback(tmp_path, relocated_readme):
    from pingstore.contracts import validate_run_directory
    from pingstore.migrate_v2 import activate_store, prepare_store, tree_inventory

    source = _v1_store(tmp_path, relocated_readme=relocated_readme)
    baseline = tree_inventory(source)
    work = tmp_path / "migration"
    report = prepare_store(source, work)
    assert tree_inventory(source) == baseline
    run = work / "prepared/runs/exp001-r001-local"
    manifest = validate_run_directory(run)
    assert manifest["execution"]["command"] == ["original", "command"]
    assert manifest["provenance"] == {"original": True}
    assert (run / "presentation/rasters__example.png").read_bytes() == b"picture"
    mapping = load_json(run / "export/provenance/format-v1/mapping.json")
    assert len(mapping["files"]) == report["runs"][0]["source_file_count"]
    for row in mapping["files"]:
        assert (run / row["destination"]).read_bytes() == (source / "runs/exp001-r001-local" / row["path"]).read_bytes()
    activate_store(source, work)
    assert tree_inventory(work / "rollback") == baseline
    validate_run_directory(source / "runs/exp001-r001-local")
    with pytest.raises(PingstoreError):
        activate_store(source, work)


def test_migration_rejects_unexplained_checksum_change(tmp_path):
    from pingstore.migrate_v2 import prepare_store

    source = _v1_store(tmp_path)
    (source / "runs/exp001-r001-local/files/state/weights.pth").write_bytes(b"changed")
    with pytest.raises(PingstoreError, match="checksum mismatch"):
        prepare_store(source, tmp_path / "migration")
    assert (source / "runs/exp001-r001-local/files").is_dir()


def test_legacy_flattening_refuses_collision(tmp_path):
    from pingstore.layout import legacy_mapping

    (tmp_path / "rasters").mkdir()
    (tmp_path / "rasters/example.png").write_bytes(b"nested")
    (tmp_path / "rasters__example.png").write_bytes(b"flat")
    with pytest.raises(PingstoreError, match="collision"):
        legacy_mapping(tmp_path)


def test_activation_rechecks_source_and_recovers_interruption(tmp_path, monkeypatch):
    import os

    from pingstore.migrate_v2 import (
        activate_store,
        prepare_store,
        recover_store,
        tree_inventory,
    )

    source = _v1_store(tmp_path)
    work = tmp_path / "migration"
    prepare_store(source, work)
    baseline = tree_inventory(source)
    original_rename = os.rename

    def fail_second_rename(old, new):
        if Path(old) == work / "prepared":
            raise OSError("interrupted")
        return original_rename(old, new)

    monkeypatch.setattr(os, "rename", fail_second_rename)
    with pytest.raises(OSError, match="interrupted"):
        activate_store(source, work)
    assert tree_inventory(source) == baseline
    # Simulate process death after the first rename (no exception cleanup).
    journal = load_json(work / "migration.json")
    journal["phase"] = "activating"
    (work / "migration.json").write_text(json.dumps(journal))
    original_rename(source, work / "rollback")
    recover_store(source, work)
    assert tree_inventory(source) == baseline
    monkeypatch.setattr(os, "rename", original_rename)
    (source / "runs/exp001-r001-local/files/numbers.json").write_text('{"changed": true}')
    with pytest.raises(PingstoreError, match="changed since verification"):
        activate_store(source, work)


@pytest.mark.parametrize("invalid", ["nested", "symlink", "extra-root", "corrupt", "old-data-layout"])
def test_v2_reader_rejects_invalid_payload(tmp_path, invalid):
    from pingstore.contracts import validate_run_directory

    repo = _repo(tmp_path)
    run = capture_local_run(repo, "exp001", _staging(tmp_path))
    directory = repo / ".pingstore/runs" / run["run_id"]
    if invalid == "nested":
        (directory / "presentation/nested").mkdir()
    elif invalid == "symlink":
        (directory / "presentation/link.svg").symlink_to(directory / "presentation/result.svg")
    elif invalid == "extra-root":
        (directory / "extra.txt").write_text("no")
    elif invalid == "old-data-layout":
        (directory / "export").rename(directory / "data")
    else:
        (directory / "export/derived/state.npz").write_bytes(b"corrupt")
    with pytest.raises(PingstoreError):
        validate_run_directory(directory)
    with pytest.raises(PingstoreError):
        materialize_run(repo / ".pingstore", run["run_id"], tmp_path / "view")
    assert not (tmp_path / "view").exists()


def test_materialization_copies_presentation_exactly_without_suffix_filter(tmp_path):
    from pingstore.contracts import payload_digest, write_json_atomic

    repo = _repo(tmp_path)
    run = _presentation(repo)
    directory = repo / ".pingstore/runs" / run["run_id"]
    # Fixture assembly only: include an explicitly designated presentation file
    # whose suffix the v1 materializer would have silently discarded.
    (directory / "export/download.npz").write_bytes(b"presentation download")
    run["payload_digest"] = payload_digest(directory)
    write_json_atomic(directory / "run.json", run)
    materialize_run(repo / ".pingstore", run["run_id"], tmp_path / "view")
    source = {p.name: p.read_bytes() for p in (directory / "export").iterdir()}
    target = {p.name: p.read_bytes() for p in (tmp_path / "view/exp001").iterdir()}
    assert target == source


@pytest.mark.parametrize("export_root", ["../outside", "/tmp", "export/../../outside", "export/missing", "data/cells", 42])
def test_run_rejects_invalid_explicit_export_root(tmp_path, export_root):
    from pingstore.contracts import validate_run_directory, write_json_atomic

    repo = _repo(tmp_path)
    run = capture_local_run(repo, "exp001", _staging(tmp_path))
    directory = repo / ".pingstore/runs" / run["run_id"]
    run["export_root"] = export_root
    write_json_atomic(directory / "run.json", run)
    with pytest.raises(PingstoreError, match="export_root"):
        validate_run_directory(directory)


def test_run_rejects_obsolete_data_root(tmp_path):
    from pingstore.contracts import validate_run_directory, write_json_atomic

    repo = _repo(tmp_path)
    run = capture_local_run(repo, "exp001", _staging(tmp_path))
    directory = repo / ".pingstore/runs" / run["run_id"]
    run["data_root"] = "data/cells"
    write_json_atomic(directory / "run.json", run)
    with pytest.raises(PingstoreError, match="data_root is obsolete"):
        validate_run_directory(directory)
