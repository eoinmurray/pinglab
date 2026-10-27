"""Deployment-aware evidence retention and conditional metadata writes."""

import copy
import io
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from pingstore import core

from .test_maintenance import DESTINATION, manifest_for
from .test_maintenance import stores as storage_fixture


@pytest.fixture(name="stores")
def prepared_storage(tmp_path, monkeypatch):
    return storage_fixture.__wrapped__(tmp_path, monkeypatch)


def test_github_inventory_paginates_without_losing_older_pending_workflows(monkeypatch):
    calls = []

    def response(repository, endpoint):
        calls.append(endpoint)
        if "page=2" in endpoint:
            return {"workflow_runs": [{"id": 7, "status": "queued"}]}
        return {"workflow_runs": [{"id": i, "status": "completed"} for i in range(100)]}

    monkeypatch.setattr(core, "_github_json", response)
    rows = list(core._github_pages("owner/repo", "runs?per_page=100", "workflow_runs"))
    assert rows[-1] == {"id": 7, "status": "queued"}
    assert len(rows) == 101
    assert calls == ["runs?per_page=100", "runs?per_page=100&page=2"]


def test_s3_conditional_put_uses_the_exact_get_etag_and_create_once_condition(
    monkeypatch,
):
    previous = {"generation": 1}
    body = json.dumps(previous).encode()
    uploaded = []
    client = SimpleNamespace(
        get_object=lambda **kwargs: {
            "Body": io.BytesIO(body),
            "ETag": '"multipart-17"',
        },
        put_object=lambda **kwargs: uploaded.append(kwargs),
    )
    monkeypatch.setattr(core, "_s3_client", lambda _: client)
    monkeypatch.setattr(core, "rclone", lambda *args, **kwargs: uploaded[-1]["Body"])
    core.conditional_document(
        "r2:bucket/pingstore/control.json", previous, {"generation": 2}, exists=True
    )
    assert uploaded[-1]["IfMatch"] == '"multipart-17"'
    assert "IfNoneMatch" not in uploaded[-1]
    core.conditional_document(
        "r2:bucket/pingstore/new.json", {}, {"generation": 1}, exists=False
    )
    assert uploaded[-1]["IfNoneMatch"] == "*"
    assert "IfMatch" not in uploaded[-1]


def test_conditional_conflict_never_falls_back_to_an_unconditional_write(monkeypatch):
    calls = []

    class Conflict(Exception):
        response = {"Error": {"Code": "PreconditionFailed"}}

    def fail(**kwargs):
        calls.append(kwargs)
        raise Conflict("sensitive SDK detail must not appear")

    monkeypatch.setattr(core, "_s3_client", lambda _: SimpleNamespace(put_object=fail))
    with pytest.raises(core.PingstoreError, match="PreconditionFailed") as error:
        core.conditional_document(
            "r2:bucket/pingstore/control.json", {}, {}, exists=False
        )
    assert len(calls) == 1
    assert "sensitive" not in str(error.value)


def prepare(stores, monkeypatch, *, build="456/1"):
    repo, paths, state = stores
    monkeypatch.setattr(core, "public_json", lambda _: copy.deepcopy(state["catalog"]))
    return core.publication_prepare(
        repo, DESTINATION, repo / "build-evidence", build_id=build, revision="a" * 40
    )


def test_prepare_freezes_catalog_index_and_exact_incoming_ancestry(stores, monkeypatch):
    repo, paths, state = stores
    manifest = prepare(stores, monkeypatch)
    assert manifest == core.load_json(repo / "build-evidence/evidence.json")
    index = core.load_json(repo / "build-evidence/index.json")
    assert index["generation"] == manifest["index_generation"]
    assert core.load_json(repo / "build-evidence/catalog.json") == state["catalog"]
    assert state["control"]["publications"]["incoming"]["456/1"] == manifest
    plan = core.unified_prune_plan(repo, ["exp001"])
    current = next(row for row in plan["keep"] if row["run_id"] == paths["latest"].name)
    assert "incoming:456/1" in current["reasons"]


def test_successful_replacement_releases_old_evidence_only_after_deployed_bytes_match(
    stores, monkeypatch
):
    repo, paths, state = stores
    records = {
        identity: entry["record"]
        for identity, entry in state["catalog"]["runs"].items()
    }
    old = manifest_for(records, [paths["old_present"].name], "123/1")
    state["control"]["publications"]["active"] = old
    new = prepare(stores, monkeypatch)
    assert not core.unified_prune_plan(repo, ["exp001"])["prune"]
    github = dict(
        active=dict(run_id="456", revision="a" * 40, url="https://site.test"),
        pending=[],
    )
    monkeypatch.setattr(core, "github_pages_state", lambda _: github)
    monkeypatch.setattr(core, "public_json", lambda _: old)
    with pytest.raises(core.PingstoreError, match="not deployed"):
        core.publication_activate(repo, DESTINATION, new)
    assert state["control"]["publications"]["active"] == old
    monkeypatch.setattr(core, "public_json", lambda _: new)
    core.publication_activate(repo, DESTINATION, new)
    assert state["control"]["publications"]["active"] == new
    assert not state["control"]["publications"]["incoming"]
    assert paths["old_present"].name in {
        row["run_id"] for row in core.unified_prune_plan(repo, ["exp001"])["prune"]
    }


def test_failed_incoming_build_never_releases_previous_successful_deployment(
    stores, monkeypatch
):
    repo, paths, state = stores
    records = {
        identity: entry["record"]
        for identity, entry in state["catalog"]["runs"].items()
    }
    old = manifest_for(records, [paths["old_present"].name], "123/1")
    state["control"]["publications"]["active"] = old
    prepare(stores, monkeypatch)
    monkeypatch.setattr(
        core,
        "github_pages_state",
        lambda _: dict(active=dict(run_id="123"), pending=[]),
    )
    core.publication_failed(repo, DESTINATION, "456/1")
    assert state["control"]["publications"]["active"] == old
    assert not state["control"]["publications"]["incoming"]
    assert not core.unified_prune_plan(repo, ["exp001"])["prune"]


def test_deploy_only_retry_registers_frozen_evidence_again(stores, monkeypatch):
    repo, paths, state = stores
    records = {
        identity: entry["record"]
        for identity, entry in state["catalog"]["runs"].items()
    }
    frozen = manifest_for(records, [paths["old_present"].name], "456/1")
    monkeypatch.setattr(
        core, "github_pages_state", lambda _: dict(active=None, pending=[])
    )
    core.publication_retain(DESTINATION, frozen)
    core.publication_failed(repo, DESTINATION, "456/1")
    assert "456/1" not in state["control"]["publications"]["incoming"]
    assert paths["old_present"].name in {
        row["run_id"] for row in core.unified_prune_plan(repo, ["exp001"])["prune"]
    }
    core.publication_retain(DESTINATION, frozen)
    plan = core.unified_prune_plan(repo, ["exp001"])
    retained = next(
        row for row in plan["keep"] if row["run_id"] == paths["old_present"].name
    )
    assert "incoming:456/1" in retained["reasons"]
    state["control"]["operation"] = dict(kind="prune", plan_hash="held")
    with pytest.raises(core.PingstoreError, match="storage maintenance is active"):
        core.publication_retain(DESTINATION, frozen)


def test_a_deployed_build_cannot_be_released_as_failed(stores, monkeypatch):
    repo, paths, state = stores
    prepare(stores, monkeypatch)
    monkeypatch.setattr(
        core,
        "github_pages_state",
        lambda _: dict(active=dict(run_id="456"), pending=[]),
    )
    with pytest.raises(core.PingstoreError, match="deployed"):
        core.publication_failed(repo, DESTINATION, "456/1")
    assert "456/1" in state["control"]["publications"]["incoming"]


def test_publication_cas_cannot_overlap_storage_mutation(stores, monkeypatch):
    repo, paths, state = stores
    previous = copy.deepcopy(state["control"]["publications"])
    state["control"]["operation"] = {"kind": "prune", "token": "owned"}
    with pytest.raises(core.PingstoreError, match="maintenance is active"):
        core.update_publications(DESTINATION, previous, previous)


def test_status_classifies_unused_only_with_resolved_publication(stores):
    repo, paths, state = stores
    records = {
        identity: entry["record"]
        for identity, entry in state["catalog"]["runs"].items()
    }
    index = core.resolve_run_index(
        {}, records, source="remote", publications=state["control"]["publications"]
    )
    assert index["publication"] == "known"
    assert index["runs"][paths["old_present"].name]["status"] == "Unused"
    index = core.resolve_run_index({}, records, source="remote")
    assert index["runs"][paths["old_present"].name]["status"] == "Unresolved"


def test_legacy_bootstrap_refuses_missing_raw_evidence(stores, monkeypatch):
    repo, paths, state = stores
    monkeypatch.setattr(
        core,
        "github_pages_state",
        lambda _: dict(
            repository="owner/repo",
            active=dict(run_id="123", revision="a" * 40, deployment_id=1),
            pending=[],
        ),
    )

    def download(command, **kwargs):
        target = Path(command[-1]) / "artifact.tar"
        data = b'<tr class="selected-run"><td>exp999-r001-present</td></tr>'
        with tarfile.open(target, "w") as archive:
            member = tarfile.TarInfo("index.html")
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))

    monkeypatch.setattr(core.subprocess, "run", download)
    before = copy.deepcopy(state["control"])
    with pytest.raises(core.PingstoreError, match="unavailable evidence"):
        core.bootstrap_publication(repo, DESTINATION, repo / "bootstrap.json")
    assert state["control"] == before
