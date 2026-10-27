"""Catalog integrity and deletion boundaries; all remote operations are mocked."""

import copy
import json
from io import BytesIO

import pytest
from pingstore import core
from pingstore.library import PingstoreError, file_sha256, load_json
from pingstore.tests.test_discovery import make_run


def fixture_catalog(source):
    entries = {}
    for directory in source.iterdir():
        files = {
            path.relative_to(directory).as_posix(): {
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in directory.rglob("*")
            if path.is_file() and path.name != ".protected"
        }
        entries[directory.name] = {
            "record": load_json(directory / "run.json"),
            "files": files,
        }
    return {
        "schema": core.SCHEMA,
        "updated_at": "2026-10-23T00:00:00Z",
        "runs": entries,
    }


def test_catalog_authenticates_payload_inventory_and_complete_ancestry(tmp_path):
    run = make_run(tmp_path, execution={"completed_at": "2026-10-23T00:00:00Z"})
    value = fixture_catalog(tmp_path)
    core.validate_catalog(value)
    broken = copy.deepcopy(value)
    broken["runs"][run.name]["files"]["export/numbers.json"]["sha256"] = "0" * 64
    with pytest.raises(PingstoreError, match="inventory mismatch"):
        core.validate_catalog(broken)
    broken = copy.deepcopy(value)
    broken["runs"][run.name]["record"]["inputs"] = {
        "source": {
            "run_id": "exp001-r002-compute",
            "payload_digest": "sha256:" + "a" * 64,
        }
    }
    with pytest.raises(PingstoreError, match="ancestor"):
        core.validate_catalog(broken)


def test_existing_backups_are_download_verified_before_catalog_publication(
    tmp_path, monkeypatch
):
    from pingstore import core

    run = make_run(tmp_path, execution={"completed_at": "2026-10-23T00:00:00Z"})
    value = fixture_catalog(tmp_path)
    files = {
        f"{run.name}/{name}": item["bytes"]
        for name, item in value["runs"][run.name]["files"].items()
    }
    plan = {"runs": [{"run_id": run.name, "files": files}], "existing_runs": [run.name]}
    calls, uploaded = [], []
    monkeypatch.setattr(
        core,
        "read_catalog",
        lambda destination: {"schema": core.SCHEMA, "runs": {}},
    )
    monkeypatch.setattr(core, "inventory", lambda destination: files)

    def remote(*args, **kwargs):
        calls.append(args[0])
        if args[0] == "copyto":
            uploaded.append(args[1].read_bytes())
        if args[0] == "cat":
            return uploaded[-1]
        return None

    monkeypatch.setattr(core, "rclone", remote)
    core.publish_catalog(tmp_path, "r2:bucket/pingstore/runs", plan)
    assert calls.index("check") < calls.index("copyto")
    core.validate_catalog(json.loads(uploaded[-1]))


def eviction_fixture(tmp_path, monkeypatch):
    source = tmp_path / ".pingstore/runs"
    run = make_run(source, execution={"completed_at": "2026-10-23T00:00:00Z"})
    value = fixture_catalog(source)
    monkeypatch.setattr(core, "read_catalog", lambda destination: value)
    monkeypatch.setattr(
        core, "urlopen", lambda *args, **kwargs: BytesIO(json.dumps(value).encode())
    )
    return source, run, value


def test_eviction_rechecks_bytes_preserves_allocation_and_keeps_remote(
    tmp_path, monkeypatch
):
    source, run, value = eviction_fixture(tmp_path, monkeypatch)
    plan = core.eviction_plan(tmp_path, "r2:bucket/pingstore/runs")
    assert [row["run_id"] for row in plan["evict"]] == [run.name]
    calls = []
    monkeypatch.setattr(core, "rclone", lambda *args: calls.append(args))
    core.apply_eviction(tmp_path, "r2:bucket/pingstore/runs", plan["plan_hash"])
    assert not run.exists()
    assert run.name in value["runs"]
    assert calls[0][0] == "check" and "--download" in calls[0]
    assert load_json(source.parent / ".allocation/exp001.json")["high_watermark"] == 1


def test_failed_remote_verification_leaves_local_copies_intact(tmp_path, monkeypatch):
    source, run, value = eviction_fixture(tmp_path, monkeypatch)
    plan = core.eviction_plan(tmp_path, "r2:bucket/pingstore/runs")

    def fail(*args):
        raise PingstoreError("download verification failed")

    monkeypatch.setattr(core, "rclone", fail)
    with pytest.raises(PingstoreError, match="verification failed"):
        core.apply_eviction(tmp_path, "r2:bucket/pingstore/runs", plan["plan_hash"])
    assert run.exists()
    (run / ".protected").touch()
    assert not core.eviction_plan(tmp_path, "r2:bucket/pingstore/runs")["evict"]


def test_eviction_rejects_changed_local_bytes_and_catalog_mismatch(
    tmp_path, monkeypatch
):
    source, run, value = eviction_fixture(tmp_path, monkeypatch)
    plan = core.eviction_plan(tmp_path, "r2:bucket/pingstore/runs")
    (run / "README.md").write_text("Changed history")
    with pytest.raises(PingstoreError, match="differs"):
        core.apply_eviction(tmp_path, "r2:bucket/pingstore/runs", plan["plan_hash"])
    monkeypatch.setattr(core, "urlopen", lambda *args, **kwargs: BytesIO(b"{}"))
    with pytest.raises(PingstoreError, match="catalogs differ"):
        core.eviction_plan(tmp_path, "r2:bucket/pingstore/runs")


def test_backup_accepts_digest_matched_remote_ancestors_after_eviction(
    tmp_path, monkeypatch
):
    from pingstore import core

    source = tmp_path / "runs"
    parent = make_run(
        source,
        "exp001-r001-compute",
        stage="compute",
        execution={"completed_at": "2026-10-23T00:00:00Z"},
    )
    reference = {
        "run_id": parent.name,
        "payload_digest": load_json(parent / "run.json")["payload_digest"],
    }
    child = make_run(
        source,
        "exp001-r002-present",
        inputs={"compute": reference},
        execution={"completed_at": "2026-10-23T00:00:00Z"},
    )
    value = fixture_catalog(source)
    import shutil

    shutil.rmtree(parent)
    monkeypatch.setattr(core, "read_catalog", lambda destination: value)
    records = core.records_for_backup(source, "r2:bucket/pingstore/runs")
    assert set(records) == {child.name}
    value["runs"][reference["run_id"]]["record"]["payload_digest"] = (
        "sha256:" + "f" * 64
    )
    with pytest.raises(PingstoreError, match="changed cataloged input"):
        core.records_for_backup(source, "r2:bucket/pingstore/runs")


def test_catalog_preserves_previous_remote_only_history(tmp_path, monkeypatch):
    from pingstore import core

    run = make_run(tmp_path, execution={"completed_at": "2026-10-23T00:00:00Z"})
    previous = fixture_catalog(tmp_path)
    files = {
        f"{run.name}/{name}": item["bytes"]
        for name, item in previous["runs"][run.name]["files"].items()
    }
    old_bytes = json.dumps(previous).encode()
    monkeypatch.setattr(core, "read_catalog", lambda destination: previous)
    monkeypatch.setattr(core, "inventory", lambda destination: files)
    uploaded, conditions = [], []

    def remote(*args, **kwargs):
        if args[0] == "copyto":
            uploaded.append(args[1].read_bytes())
            conditions.append(args[-1])
        if args[0] == "cat":
            return uploaded[-1] if uploaded else old_bytes
        return None

    monkeypatch.setattr(core, "rclone", remote)
    core.publish_catalog(
        tmp_path, "r2:bucket/pingstore/runs", {"runs": [], "existing_runs": []}
    )
    assert json.loads(uploaded[-1])["runs"] == previous["runs"]
    assert conditions[0].startswith('If-Match: "')


def test_catalog_uses_path_component_order_for_prefixed_unit_names(tmp_path):
    from pingstore.library import payload_digest

    run = make_run(
        tmp_path,
        "exp001-r001-compute",
        stage="compute",
        execution={"completed_at": "2026-10-23T00:00:00Z"},
    )
    (run / "export/unit").mkdir()
    (run / "export/unit/data.json").write_text("{}")
    (run / "export/unit/metrics.json").write_text("{}")
    (run / "export/unit-extra.json").write_text("{}")
    record = load_json(run / "run.json")
    record["payload_digest"] = payload_digest(run)
    (run / "run.json").write_text(json.dumps(record))
    core.validate_catalog(fixture_catalog(tmp_path))


def remote_prune_fixture(tmp_path, monkeypatch):
    source = tmp_path / ".pingstore/runs"
    old = make_run(
        source,
        "exp001-r001-present",
        execution={"completed_at": "2026-10-23T00:00:00Z"},
    )
    new = make_run(
        source,
        "exp001-r002-present",
        execution={"completed_at": "2026-10-24T00:00:00Z"},
    )
    other = make_run(
        source,
        "exp002-r001-present",
        execution={"completed_at": "2026-10-23T00:00:00Z"},
    )
    state = {"catalog": fixture_catalog(source)}
    state["objects"] = {
        f"{identity}/{name}": item["bytes"]
        for identity, entry in state["catalog"]["runs"].items()
        for name, item in entry["files"].items()
    }
    monkeypatch.setattr(
        core, "read_catalog", lambda destination: copy.deepcopy(state["catalog"])
    )
    monkeypatch.setattr(core, "inventory", lambda destination: dict(state["objects"]))
    monkeypatch.setattr(
        core,
        "report_usage",
        lambda *args: (
            {"exp001": {"runs": {"exp001": new.name}}},
            {new.name: {"exp001"}},
        ),
    )
    return old, new, other, state


def test_remote_prune_refuses_unbacked_current_evidence(tmp_path, monkeypatch):
    _, new, _, state = remote_prune_fixture(tmp_path, monkeypatch)
    del state["catalog"]["runs"][new.name]
    with pytest.raises(PingstoreError, match="back up the complete current"):
        core.remote_prune_plan(tmp_path, "r2:bucket/pingstore/runs", ["exp001"])


def test_remote_prune_can_retire_all_unreferenced_runs_in_one_scope(
    tmp_path, monkeypatch
):
    old, new, other, state = remote_prune_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(core, "report_usage", lambda *args: ({}, {}))
    plan = core.remote_prune_plan(tmp_path, "r2:bucket/pingstore/runs", ["exp001"])
    assert plan["prune"] == [old.name, new.name]
    assert plan["keep"] == []
    other_objects = {
        path: size
        for path, size in state["objects"].items()
        if path.startswith(other.name + "/")
    }

    def publish(destination, previous, revised):
        core.validate_catalog(revised)
        assert set(revised["runs"]) == {other.name}
        state["catalog"] = revised

    def delete(*args):
        assert set(state["catalog"]["runs"]) == {other.name}
        for path in args[-1].read_text().splitlines():
            assert path.split("/", 1)[0] in {old.name, new.name}
            del state["objects"][path]

    monkeypatch.setattr(core, "replace_catalog", publish)
    monkeypatch.setattr(core, "rclone", delete)
    core.apply_remote_prune(
        tmp_path, "r2:bucket/pingstore/runs", ["exp001"], plan["plan_hash"]
    )
    assert state["objects"] == other_objects


def test_remote_prune_retains_protected_and_cross_experiment_ancestry(
    tmp_path, monkeypatch
):
    old, _, other, state = remote_prune_fixture(tmp_path, monkeypatch)
    state["objects"][old.name + "/.protected"] = 0
    assert not core.remote_prune_plan(tmp_path, "r2:bucket/pingstore/runs", ["exp001"])[
        "prune"
    ]
    del state["objects"][old.name + "/.protected"]
    record = load_json(other / "run.json")
    record["inputs"] = {
        "source": {
            "run_id": old.name,
            "payload_digest": load_json(old / "run.json")["payload_digest"],
        }
    }
    (other / "run.json").write_text(json.dumps(record))
    state["catalog"] = fixture_catalog(tmp_path / ".pingstore/runs")
    state["objects"][other.name + "/run.json"] = (other / "run.json").stat().st_size
    assert not core.remote_prune_plan(tmp_path, "r2:bucket/pingstore/runs", ["exp001"])[
        "prune"
    ]


def test_remote_prune_publishes_valid_index_before_scoped_deletion(
    tmp_path, monkeypatch
):
    old, new, other, state = remote_prune_fixture(tmp_path, monkeypatch)
    plan = core.remote_prune_plan(tmp_path, "r2:bucket/pingstore/runs", ["exp001"])
    assert plan["prune"] == [old.name]
    calls = []

    def publish(destination, previous, revised):
        core.validate_catalog(revised)
        assert old.name not in revised["runs"]
        assert {new.name, other.name} <= revised["runs"].keys()
        state["catalog"] = revised
        calls.append("publish")

    def delete(*args):
        assert args[0] == "delete"
        assert calls == ["publish"]
        paths = args[-1].read_text().splitlines()
        assert paths and all(name.startswith(old.name + "/") for name in paths)
        for name in paths:
            del state["objects"][name]
        calls.append("delete")

    monkeypatch.setattr(core, "replace_catalog", publish)
    monkeypatch.setattr(core, "rclone", delete)
    core.apply_remote_prune(
        tmp_path, "r2:bucket/pingstore/runs", ["exp001"], plan["plan_hash"]
    )
    assert calls == ["publish", "delete"]


def test_remote_prune_stops_before_deletion_if_plan_or_catalog_changes(
    tmp_path, monkeypatch
):
    _, _, _, state = remote_prune_fixture(tmp_path, monkeypatch)
    plan = core.remote_prune_plan(tmp_path, "r2:bucket/pingstore/runs", ["exp001"])
    calls = []
    monkeypatch.setattr(core, "rclone", lambda *args: calls.append(args))
    state["catalog"]["updated_at"] = "2026-10-25T00:00:00Z"
    with pytest.raises(PingstoreError, match="plan changed"):
        core.apply_remote_prune(
            tmp_path, "r2:bucket/pingstore/runs", ["exp001"], plan["plan_hash"]
        )
    assert not calls
    plan = core.remote_prune_plan(tmp_path, "r2:bucket/pingstore/runs", ["exp001"])

    def fail(*args):
        raise PingstoreError("conditional publication failed")

    monkeypatch.setattr(core, "replace_catalog", fail)
    with pytest.raises(PingstoreError, match="publication failed"):
        core.apply_remote_prune(
            tmp_path, "r2:bucket/pingstore/runs", ["exp001"], plan["plan_hash"]
        )
    assert not calls
