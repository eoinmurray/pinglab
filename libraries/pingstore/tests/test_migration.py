"""Non-destructive migration of simple dependencies and archived source views."""

import json

import pytest
from pingstore.core import Store, write_json_atomic
from pingstore.migration import migrate


def old_record(identity, stage, inputs=None):
    return dict(
        schema="pingstore.run/v4",
        run_id=identity,
        experiment="exp001",
        stage=stage,
        created_at="2026-11-11T10:00:00Z",
        origin="local",
        payload_digest="sha256:" + "a" * 64,
        inputs=inputs or {},
        runset_inputs=inputs or {},
        execution={
            "completed_at": "2026-11-12T10:00:00Z",
            "configuration": {"seed": 42},
        },
    )


def test_migration_preserves_payloads_and_original_metadata(tmp_path):
    source, target = tmp_path / "old", tmp_path / "new"
    compute = old_record("exp001-r001-compute", "compute")
    present = old_record(
        "exp001-r002-present",
        "present",
        {
            "compute": {
                "run_id": compute["run_id"],
                "payload_digest": compute["payload_digest"],
            }
        },
    )
    for record in [compute, present]:
        directory = source / "runs" / record["run_id"]
        write_json_atomic(directory / "run.json", record)
        (directory / "export").mkdir()
        (directory / "export/data.bin").write_bytes(b"exact original bytes")
    before = (source / "runs" / compute["run_id"] / "run.json").read_bytes()
    assert not migrate(source, target)["applied"]
    assert not target.exists()
    migrate(source, target, apply=True)
    store = Store(target)
    assert store.get(present["run_id"]).record["depends_on"] == [compute["run_id"]]
    assert store.get(compute["run_id"]).details["execution"]["configuration"] == {
        "seed": 42
    }
    assert store.get(compute["run_id"]).record["created_at"] == "2026-11-12T10:00:00Z"
    assert (source / "runs" / compute["run_id"] / "run.json").read_bytes() == before
    assert (
        json.loads(
            (target / "migration/originals" / (compute["run_id"] + ".json")).read_text()
        )
        == compute
    )
    assert migrate(source, target, apply=True)["applied"]


def test_source_view_becomes_an_ordinary_run(tmp_path):
    source, target = tmp_path / "old", tmp_path / "new"
    original = old_record("exp001-r001-compute", "compute")
    owner = old_record("exp001-r003-compute", "compute")
    owner["consolidation"] = {
        "resolve_sources": True,
        "sources": {
            original["run_id"]: {
                "record": original,
                "export_paths": {"native/start.json": "retained.json"},
            }
        },
    }
    directory = source / "runs" / owner["run_id"]
    write_json_atomic(directory / "run.json", owner)
    write_json_atomic(directory / "export/retained.json", {"network": "native"})
    migrate(source, target, apply=True)
    restored = Store(target).get(original["run_id"])
    assert json.loads(restored.file("native/start.json").read_text()) == {
        "network": "native"
    }


def test_migration_cannot_overwrite_original_store(tmp_path):
    with pytest.raises(ValueError, match="separate destination"):
        migrate(tmp_path, tmp_path)
