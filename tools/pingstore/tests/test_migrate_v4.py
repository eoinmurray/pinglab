from __future__ import annotations

from pingstore.contracts import (
    PREVIOUS_RUN_SCHEMA,
    RUN_SCHEMA,
    file_sha256,
    load_json,
    payload_digest,
    validate_operational_run_directory,
    write_json_atomic,
)
from pingstore.migrate_v4 import apply_migration, rollback


def make_v3(store, run_id, stage, *, inputs=None):
    directory = store / "runs" / run_id
    (directory / "export").mkdir(parents=True)
    (directory / "export/result.json").write_text("{}\n")
    (directory / "export/run.sh").write_text("python run.py\n")
    (directory / "provenance/simulations").mkdir(parents=True)
    (directory / "provenance/simulations/config.json").write_text("{}\n")
    (directory / "provenance/command.json").write_text("{}\n")
    (directory / "provenance/source.patch").write_text("patch\n")
    record = {
        "schema": PREVIOUS_RUN_SCHEMA,
        "run_id": run_id,
        "experiment": "exp001",
        "collection": "demo",
        "stage": stage,
        "origin": "local",
        "created_at": "2026-08-29T10:00:00Z",
        "inputs": inputs or {},
        "execution": {"command": ["python", "run.py"]},
        "provenance": {"git_commit": "abc", "patch": {"path": "provenance/source.patch"}},
        "payload_digest": "sha256:" + "0" * 64,
    }
    write_json_atomic(directory / "run.json", record)
    record["payload_digest"] = payload_digest(directory)
    write_json_atomic(directory / "run.json", record)
    return directory, record


def test_v4_migration_archives_history_and_rewrites_input_pins(tmp_path):
    store = tmp_path / ".pingstore"
    parent, parent_record = make_v3(store, "exp001-r001-compute", "compute")
    reference = {
        "run_id": parent.name,
        "payload_digest": parent_record["payload_digest"],
        "run_json_sha256": file_sha256(parent / "run.json"),
    }
    child, _ = make_v3(
        store, "exp001-r002-analyse", "analyse", inputs={"compute": reference}
    )

    archive = apply_migration(store)

    for directory in (parent, child):
        assert {path.name for path in directory.iterdir()} == {"run.json", "README.md", "export"}
        assert validate_operational_run_directory(directory)["schema"] == RUN_SCHEMA
        assert "V4 history" in (directory / "README.md").read_text()
    assert (parent / "export/evidence/simulations/config.json").is_file()
    assert not (parent / "export/evidence/command.json").exists()
    assert not (parent / "export/run.sh").exists()
    assert (archive / parent.name / "provenance/source.patch").is_file()
    assert (archive / parent.name / "export_replay_scripts/run.sh").is_file()
    pin = load_json(child / "run.json")["inputs"]["compute"]
    assert pin == {
        "run_id": parent.name,
        "payload_digest": load_json(parent / "run.json")["payload_digest"],
    }

    rollback(store, archive)
    assert load_json(parent / "run.json")["schema"] == PREVIOUS_RUN_SCHEMA
    assert (parent / "provenance/source.patch").is_file()
    assert (parent / "export/run.sh").is_file()
    assert not (parent / "export/evidence").exists()
