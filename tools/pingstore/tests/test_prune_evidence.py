from __future__ import annotations

from pingstore.contracts import RUN_SCHEMA, payload_digest, write_json_atomic
from pingstore.prune_evidence import apply, rollback, validate_graph


def make_run(repo, run_id, stage, *, inputs=None):
    directory = repo / ".pingstore/runs" / run_id
    (directory / "export/evidence").mkdir(parents=True)
    (directory / "export/result.json").write_text("{}\n")
    (directory / "export/evidence/obsolete.log").write_text("unused\n")
    (directory / "export/evidence/used.json").write_text("{}\n")
    (directory / "README.md").write_text(f"# {run_id}\n")
    record = {
        "schema": RUN_SCHEMA,
        "run_id": run_id,
        "experiment": "exp001",
        "collection": "test",
        "stage": stage,
        "origin": "local",
        "created_at": "2026-08-30T00:00:00+00:00",
        "inputs": inputs or {},
        "execution": {"command": ["test"]},
        "provenance": {},
        "payload_digest": "sha256:" + "0" * 64,
        "historical_import": {
            "original_records": "export/evidence/obsolete.log",
            "retained": "export/evidence/used.json",
        },
    }
    write_json_atomic(directory / "run.json", record)
    record["payload_digest"] = payload_digest(directory)
    write_json_atomic(directory / "run.json", record)
    return directory, record


def test_prune_is_recoverable_and_updates_input_pins(tmp_path):
    repo = tmp_path / "repo"
    parent, original_parent = make_run(repo, "exp001-r001-compute", "compute")
    child, original_child = make_run(
        repo,
        "exp001-r002-analyse",
        "analyse",
        inputs={
            "compute": {
                "run_id": parent.name,
                "payload_digest": original_parent["payload_digest"],
            }
        },
    )
    manifest = repo / "prune.tsv"
    target = parent / "export/evidence/obsolete.log"
    manifest.write_text(
        f".pingstore/runs/{parent.name}/export/evidence/obsolete.log\t{target.stat().st_size}\n"
    )

    archive = apply(repo, manifest)

    records = validate_graph(repo / ".pingstore/runs")
    assert not target.exists()
    assert (parent / "export/evidence/used.json").is_file()
    assert "original_records" not in records[parent.name]["historical_import"]
    assert records[parent.name]["historical_import"]["retained"] == "export/evidence/used.json"
    assert records[child.name]["inputs"]["compute"]["payload_digest"] == records[parent.name]["payload_digest"]
    assert records[parent.name]["payload_digest"] != original_parent["payload_digest"]
    assert (archive / f"deleted/{parent.name}/export/evidence/obsolete.log").is_file()

    rollback(repo, archive)

    restored = validate_graph(repo / ".pingstore/runs")
    assert target.read_text() == "unused\n"
    assert restored[parent.name] == original_parent
    assert restored[child.name] == original_child
