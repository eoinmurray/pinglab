"""Authorized metadata correction separates runsets without changing evidence."""

import pytest
from pingstore import core

from .test_prune import make_run, reference


def test_old_records_detach_provenance_and_preserve_payloads(tmp_path):
    old = make_run(
        tmp_path,
        "exp110-r001-compute",
        stage="compute",
        created_at="2026-01-01T00:00:00Z",
    )
    compute = make_run(
        tmp_path,
        "exp110-r002-compute",
        stage="compute",
        created_at="2026-01-02T00:00:00Z",
        inputs={"previous_attempt": reference(old)},
    )
    present = make_run(
        tmp_path,
        "exp110-r003-present",
        stage="present",
        created_at="2026-01-03T00:00:00Z",
        inputs={"compute": reference(compute)},
    )
    paths = [old, compute, present]
    payloads = {p.name: core.payload_inventory(p) for p in paths}
    before = {p.name: core.load_json(p / "run.json") for p in paths}
    core.reporting_index(tmp_path, source="local")
    plan = core.detach_runsets_plan(tmp_path)
    assert all("runset_inputs" not in core.load_json(p / "run.json") for p in paths)
    result = core.apply_detach_runsets(tmp_path, plan["plan_hash"])
    assert result["phase"] == "complete"
    records = {p.name: core.load_json(p / "run.json") for p in paths}
    assert records[compute.name]["inputs"] == before[compute.name]["inputs"]
    assert records[compute.name]["runset_inputs"] == {}
    assert records[present.name]["runset_inputs"] == {"compute": reference(compute)}
    assert core._runset(records, present.name) == sorted([compute.name, present.name])
    assert {p.name: core.payload_inventory(p) for p in paths} == payloads
    assert "authorized metadata correction" in (compute / "README.md").read_text()
    assert core.apply_detach_runsets(tmp_path, plan["plan_hash"])["phase"] == "complete"
    assert not core.detach_runsets_plan(tmp_path)["changes"]
    assert not core.reporting_index(tmp_path, source="local")["selection_errors"]
    # The detached historical source may disappear without breaking this runset.
    del records[old.name]
    assert core._runset(records, present.name) == sorted(records)


def test_delegated_analysis_promotes_original_compute_pin(tmp_path):
    compute = make_run(
        tmp_path,
        "exp110-r001-compute",
        stage="compute",
        created_at="2026-01-01T00:00:00Z",
    )
    old = make_run(
        tmp_path,
        "exp110-r002-analyse",
        stage="analyse",
        created_at="2026-01-02T00:00:00Z",
        inputs={"compute": reference(compute)},
    )
    current = make_run(
        tmp_path,
        "exp110-r003-analyse",
        stage="analyse",
        created_at="2026-01-03T00:00:00Z",
        inputs={"frontier_analysis": reference(old)},
    )
    plan = core.detach_runsets_plan(tmp_path)
    core.apply_detach_runsets(tmp_path, plan["plan_hash"])
    record = core.load_json(current / "run.json")
    assert record["runset_inputs"] == {"compute": reference(compute)}
    assert record["inputs"] == {"frontier_analysis": reference(old)}


def test_modified_metadata_invalidates_plan(tmp_path):
    path = make_run(
        tmp_path,
        "exp110-r001-compute",
        stage="compute",
        created_at="2026-01-01T00:00:00Z",
    )
    plan = core.detach_runsets_plan(tmp_path)
    (path / "README.md").write_text("changed")
    with pytest.raises(core.PingstoreError, match="plan changed"):
        core.apply_detach_runsets(tmp_path, plan["plan_hash"])


def test_dependency_contract_rejects_cross_runset_edges(tmp_path):
    path = make_run(
        tmp_path,
        "exp110-r001-compute",
        stage="compute",
        created_at="2026-01-01T00:00:00Z",
    )
    record = core.load_json(path / "run.json")
    record["runset_inputs"] = {
        "other": {
            "run_id": "exp111-r001-compute",
            "payload_digest": "sha256:" + "0" * 64,
        }
    }
    with pytest.raises(core.PingstoreError, match="runset links"):
        core.validate_run(record)


def test_remote_correction_resumes_and_keeps_catalog_and_local_metadata_equal(
    tmp_path, monkeypatch
):
    import copy
    import hashlib
    import io
    from types import SimpleNamespace

    from .test_maintenance import DESTINATION, stores

    repo, paths, state = stores.__wrapped__(tmp_path, monkeypatch)
    objects = {
        f"pingstore/runs/{identity}/{name}": (
            repo / ".pingstore/runs" / identity / name
        ).read_bytes()
        for identity in state["catalog"]["runs"]
        for name in ("run.json", "README.md")
    }
    failed = False
    calls = 0

    def get_object(*, Bucket, Key):
        data = objects[Key]
        return {"Body": io.BytesIO(data), "ETag": hashlib.sha256(data).hexdigest()}

    def put_object(*, Bucket, Key, Body, IfMatch, **kwargs):
        nonlocal failed, calls
        calls += 1
        if calls == 2 and not failed:
            failed = True
            raise RuntimeError("injected metadata interruption")
        assert IfMatch == hashlib.sha256(objects[Key]).hexdigest()
        objects[Key] = Body
        state["objects"][Key.removeprefix("pingstore/runs/")] = len(Body)

    def download(*args, **kwargs):
        assert args[0] == "copy"
        root = args[2]
        for key, data in objects.items():
            path = root / key.removeprefix("pingstore/runs/")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)

    monkeypatch.setattr(core, "rclone", download)
    monkeypatch.setattr(
        core,
        "_s3_client",
        lambda _: SimpleNamespace(get_object=get_object, put_object=put_object),
    )
    before = copy.deepcopy(state["catalog"])
    plan = core.detach_runsets_plan(repo, DESTINATION)
    with pytest.raises(RuntimeError, match="injected metadata interruption"):
        core.apply_detach_runsets(repo, plan["plan_hash"], DESTINATION)
    assert state["control"]["operation"] is not None
    assert state["catalog"] == before
    result = core.apply_detach_runsets(repo, plan["plan_hash"], DESTINATION)
    assert result["phase"] == "complete"
    assert state["control"]["operation"] is None
    for identity, entry in state["catalog"]["runs"].items():
        assert entry["record"] == core.load_json(
            repo / ".pingstore/runs" / identity / "run.json"
        )
        for name in ("run.json", "README.md"):
            data = objects[f"pingstore/runs/{identity}/{name}"]
            assert data == (repo / ".pingstore/runs" / identity / name).read_bytes()
            assert entry["files"][name] == {
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        assert (
            entry["record"]["payload_digest"]
            == before["runs"][identity]["record"]["payload_digest"]
        )
    assert not core.detach_runsets_plan(repo, DESTINATION)["changes"]


def test_new_reanalysis_inherits_compute_dependency_without_old_chain(tmp_path):
    pin = {"run_id": "exp110-r001-compute", "payload_digest": "sha256:" + "a" * 64}
    prior = {
        "run_id": "exp110-r002-analyse",
        "experiment": "exp110",
        "stage": "analyse",
        "inputs": {"compute": pin},
        "runset_inputs": {"compute": pin},
        "payload_digest": "sha256:" + "b" * 64,
    }
    current = {
        "run_id": "exp110-r003-analyse",
        "experiment": "exp110",
        "stage": "analyse",
        "inputs": {
            "previous": {
                "run_id": prior["run_id"],
                "payload_digest": prior["payload_digest"],
            }
        },
    }
    assert core._detached_inputs(current, {prior["run_id"]: prior}) == {"compute": pin}
