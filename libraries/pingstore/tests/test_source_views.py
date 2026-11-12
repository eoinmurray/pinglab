"""Authenticated archived bank views; fixtures never execute scientific work."""

import copy
import shutil

import pytest
from pingstore import core


def make_run(store, identity, content='{"value": 1}\n'):
    directory = store / "runs" / identity
    core.initialize_layout(directory, "exp001")
    (directory / "export" / "numbers.json").write_text(content)
    record = dict(
        schema=core.RUN_SCHEMA,
        run_id=identity,
        experiment="exp001",
        collection="demo",
        origin="local",
        created_at="2026-11-12T00:00:00Z",
        stage="compute",
        scope="full",
        inputs={},
        runset_inputs={},
        execution={"completed_at": "2026-11-12T00:00:01Z"},
        provenance={},
    )
    record["payload_digest"] = core.payload_digest(directory)
    core.write_json_atomic(directory / "run.json", record)
    return directory, record


def archive(store):
    original, record = make_run(store, "exp001-r001-compute")
    owner, owner_record = make_run(store, "exp001-r002-compute")
    owner_record["consolidation"] = dict(
        resolve_sources=True,
        sources={
            record["run_id"]: dict(
                record=copy.deepcopy(record),
                readme=(original / "README.md").read_text(),
                export_paths={"numbers.json": "numbers.json"},
            )
        },
    )
    core.write_json_atomic(owner / "run.json", owner_record)
    shutil.rmtree(original)
    return record, owner, owner_record


def test_exact_archived_view_and_future_analysis_pins(tmp_path, monkeypatch):
    store = tmp_path / ".pingstore"
    record, owner, owner_record = archive(store)
    ref = {key: record[key] for key in ["run_id", "payload_digest"]}
    source = core.source_run(store, record["run_id"], reference=ref, stage="compute")
    assert source.record == record
    assert source.reference == ref
    assert (
        source.file("numbers.json").read_bytes()
        == (owner / "export/numbers.json").read_bytes()
    )
    source.check_unchanged()
    monkeypatch.setattr(core, "memberships", lambda repo: {"exp001": "demo"})
    monkeypatch.setattr(
        core, "capture_code", lambda repo, directory: {"git_commit": "0" * 40}
    )
    with core.stage_run(
        tmp_path, "exp001", "analyse", inputs={"compute": source}
    ) as run:
        (run.export / "results.json").write_text("{}")
    assert run.record["inputs"]["compute"] == ref
    assert run.record["runset_inputs"]["compute"]["run_id"] == owner_record["run_id"]
    assert (
        core._source_archives({owner_record["run_id"]: owner_record})[record["run_id"]][
            "source_digest"
        ]
        == ref["payload_digest"]
    )
    with pytest.raises(core.PingstoreError, match="checksum changed"):
        core.source_run(
            store,
            record["run_id"],
            reference={**ref, "payload_digest": "sha256:" + "0" * 64},
        )


def test_owner_loss_and_ambiguous_archives_are_not_silent_fallbacks(tmp_path):
    record, owner, owner_record = archive(tmp_path)
    source = core.source_run(tmp_path, record["run_id"])
    other, other_record = make_run(tmp_path, "exp001-r003-compute")
    other_record["consolidation"] = copy.deepcopy(owner_record["consolidation"])
    core.write_json_atomic(other / "run.json", other_record)
    with pytest.raises(core.PingstoreError, match="ambiguous"):
        core.source_run(tmp_path, record["run_id"])
    shutil.rmtree(other)
    shutil.rmtree(owner)
    with pytest.raises((FileNotFoundError, core.PingstoreError)):
        source.check_unchanged()
    with pytest.raises(FileNotFoundError):
        core.source_run(tmp_path, record["run_id"])


def test_changed_owner_bytes_are_rejected(tmp_path):
    record, owner, _ = archive(tmp_path)
    (owner / "export/numbers.json").write_text('{"changed": true}')
    with pytest.raises(core.PingstoreError, match="checksum mismatch"):
        core.source_run(tmp_path, record["run_id"])


def test_archived_mapping_must_reproduce_original_payload(tmp_path):
    record, owner, owner_record = archive(tmp_path)
    (owner / "export/other.json").write_text('{"different": 2}')
    owner_record["consolidation"]["sources"][record["run_id"]]["export_paths"] = {
        "numbers.json": "other.json"
    }
    owner_record["payload_digest"] = core.payload_digest(owner)
    core.write_json_atomic(owner / "run.json", owner_record)
    with pytest.raises(core.PingstoreError, match="archived source payload differs"):
        core.source_run(tmp_path, record["run_id"])


def test_view_cache_is_released_with_its_owner(tmp_path):
    record, owner, owner_record = archive(tmp_path)
    source = core.source_run(tmp_path, record["run_id"])
    assert source.directory.exists()
    core._clear_source_views(
        tmp_path, owner_record["run_id"], owner_record["payload_digest"]
    )
    assert not source.directory.exists()
    assert owner.exists()


def test_external_archived_consumer_retains_physical_owner(tmp_path, monkeypatch):
    store = tmp_path / ".pingstore"
    record, owner, owner_record = archive(store)
    source = core.source_run(store, record["run_id"])
    monkeypatch.setattr(core, "memberships", lambda repo: {"exp002": "demo"})
    monkeypatch.setattr(
        core, "capture_code", lambda repo, directory: {"git_commit": "0" * 40}
    )
    with core.stage_run(tmp_path, "exp002", "analyse", inputs={"bank": source}) as run:
        (run.export / "results.json").write_text("{}")
    assert run.record["inputs"]["bank"] == source.reference
    assert run.record["runset_inputs"] == {}
    assert (
        core._dependency_inputs(run.record)["archived:bank"] == source.owner.reference
    )
    assert core._runset(
        {run.run_id: run.record, owner.name: owner_record}, run.run_id
    ) == sorted([run.run_id, owner.name])


def test_materializing_view_refreshes_authenticated_owner_fingerprint(tmp_path):
    record, owner, owner_record = archive(tmp_path)
    runs = tmp_path / "runs"
    _, _, cache = core._local_index_records(runs, {}, verify=True)
    import hashlib

    namespace = hashlib.sha256(str(runs).encode()).hexdigest()[:12]
    path = tmp_path / f".report-index-cache-{namespace}.json"
    core.write_json_atomic(path, cache)
    core.source_run(tmp_path, record["run_id"])
    records, errors, _ = core._local_index_records(
        runs, core.load_json(path), verify=False
    )
    assert not errors
    assert records[owner.name] == owner_record
