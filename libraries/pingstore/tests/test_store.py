"""Filesystem contracts and performance boundaries; no scientific execution."""

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor

import pytest
from pingstore import core
from pingstore.core import (
    PingstoreError,
    Store,
    source_run,
    stage_run,
    validate_run,
    write_json_atomic,
)


def record(identity="sample", **kwargs):
    return dict(
        id=identity,
        experiment="exp001",
        stage="present",
        created_at="2026-11-12T12:00:00Z",
        depends_on=[],
        metadata={},
        **kwargs,
    )


def insert(store, tmp_path, identity="sample", dependencies=()):
    source = tmp_path / ("source-" + identity)
    source.mkdir()
    (source / "nested").mkdir()
    (source / "nested/data.bin").write_bytes(b"payload")
    value = record(identity)
    value["depends_on"] = list(dependencies)
    return store.insert(source, record=value)


def test_insert_read_and_list_do_not_hash_or_spawn(tmp_path, monkeypatch):
    store = Store(tmp_path / "store")
    monkeypatch.setattr(core, "file_sha256", lambda *a: pytest.fail("payload hash"))
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("subprocess"))
    run = insert(store, tmp_path)
    assert store.get("sample").record == run.record
    assert store.list() == [run.record]
    assert (run.directory / "nested/data.bin").read_bytes() == b"payload"
    with pytest.raises(FileExistsError):
        store.insert(tmp_path / "source-sample", record=run.record)


def test_failure_is_hidden_and_reused_ids_are_rejected(tmp_path):
    store = Store(tmp_path / "store")
    with pytest.raises(RuntimeError), store.create("exp001", identity="failed") as run:
        (run.directory / "result").write_text("partial")
        raise RuntimeError("interrupted")
    assert store.list() == []
    with pytest.raises(FileExistsError), store.create("exp001", identity="failed"):
        pass


def test_dependencies_warn_but_explicit_override_deletes_only_target(tmp_path):
    store = Store(tmp_path / "store")
    insert(store, tmp_path, "parent")
    child = insert(store, tmp_path, "child", ["parent"])
    with pytest.raises(PingstoreError, match="dependent runs"):
        store.delete(["parent"])
    assert store.delete(["parent"], allow_dependants=True) == ["parent"]
    assert store.get("child").record == child.record
    assert store.get("child").record["depends_on"] == ["parent"]


@pytest.mark.parametrize(
    "identity", ["../escape", "/tmp/escape", ".", "..", "bad/name", ""]
)
def test_reject_unsafe_ids(identity):
    with pytest.raises(PingstoreError):
        validate_run(record(identity))


def test_symlink_and_old_schema_are_rejected(tmp_path):
    store = Store(tmp_path / "store")
    source = tmp_path / "source"
    source.mkdir()
    (source / "link").symlink_to("/etc/hosts")
    with pytest.raises(PingstoreError, match="unsupported"):
        store.insert(source, record=record())
    directory = store.runs / "old"
    directory.mkdir(parents=True)
    write_json_atomic(
        directory / "run.json", {"schema": "pingstore.run/v4", "run_id": "old"}
    )
    with pytest.raises(PingstoreError, match="migrate"):
        store.get("old")


def test_latest_is_deterministic_and_does_not_require_ancestry(tmp_path):
    store = Store(tmp_path / "store")
    insert(store, tmp_path, "a", ["absent"])
    insert(store, tmp_path, "b")
    assert store.latest("exp001")["id"] == "b"


def test_stage_completion_keeps_scientific_details_outside_record(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(core, "capture_code", lambda *a: {"git_commit": "fixture"})
    with stage_run(tmp_path, "exp001", "compute", configuration={"seed": 42}) as first:
        (first.export / "nested").mkdir()
        (first.export / "nested/data.json").write_text('{"value": 3}')
    source = source_run(tmp_path / ".pingstore", first.run_id)
    with stage_run(
        tmp_path, "exp002", "present", inputs={"simulation": source}
    ) as second:
        (second.export / "numbers.json").write_text('{"score": 1}')
    final = Store(tmp_path / ".pingstore").get(second.run_id)
    assert final.record["depends_on"] == [first.run_id]
    assert "execution" not in final.record and "payload_digest" not in final.record
    assert source.details["execution"]["configuration"] == {"seed": 42}
    assert source.file("nested", "data.json").read_text() == '{"value": 3}'
    assert not (final.directory / ".scratch").exists()
    assert json.loads((final.directory / "details.json").read_text())["inputs"] == {
        "simulation": {"run_id": first.run_id}
    }


def test_reservations_do_not_reuse_deleted_ids_or_need_counters(tmp_path):
    def allocate(_):
        return core.reserve_stage(tmp_path, "exp001", "compute")

    with ThreadPoolExecutor(max_workers=4) as pool:
        ids = list(pool.map(allocate, range(30)))
    assert len(set(ids)) == 30
    assert not (tmp_path / "allocations.json").exists()
