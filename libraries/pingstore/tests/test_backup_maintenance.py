"""Backup shares the deletion fence, refuses resurrection and resumes exact inputs."""

import copy

import pytest
from pingstore import core

from .test_maintenance import DESTINATION
from .test_maintenance import stores as storage_fixture
from .test_prune import make_run


@pytest.fixture
def backup_store(tmp_path, monkeypatch):
    repo, paths, state = storage_fixture.__wrapped__(tmp_path, monkeypatch)
    source = repo / ".pingstore/runs"
    state["bodies"] = {
        f"{directory.name}/{name}": (directory / name).read_bytes()
        for directory in source.iterdir()
        if core.STAGE_ID_RE.fullmatch(directory.name)
        for name in core._maintenance_files(directory)
    }
    state["fail_copy"] = False

    def remote(*args, **kwargs):
        command = args[0]
        if command == "cat":
            return state["bodies"][str(args[1]).removeprefix(DESTINATION + "/")]
        assert command in ("copy", "check")
        selected = args[args.index("--files-from-raw") + 1].read_text().splitlines()
        assert state["control"]["operation"] is not None
        for index, name in enumerate(selected):
            data = (source / name).read_bytes()
            if command == "copy":
                if name in state["bodies"]:
                    assert state["bodies"][name] == data
                state["bodies"][name] = data
                state["objects"][name] = len(data)
                if state["fail_copy"] and index == 0:
                    raise core.PingstoreError("injected upload failure")
            else:
                assert state["bodies"][name] == data
        return None

    monkeypatch.setattr(core, "rclone", remote)
    return repo, paths, state


def test_backups_and_pruning_share_one_remote_writer_fence(backup_store):
    repo, paths, state = backup_store
    source = repo / ".pingstore/runs"
    state["control"]["operation"] = {"kind": "prune", "plan_hash": "held"}
    with pytest.raises(core.PingstoreError, match="unfinished remote operation"):
        core.coordinated_backup(source, DESTINATION, dry_run=True, confirm=None)
    assert not state["calls"]


def test_upload_failure_is_journaled_and_original_plan_resumes(backup_store):
    repo, paths, state = backup_store
    new = make_run(
        repo, "exp001-r099-present", stage="present", created_at="2026-03-01T00:00:00Z"
    )
    source = repo / ".pingstore/runs"
    plan = core._coordinated_backup_plan(source, DESTINATION)
    state["fail_copy"] = True
    with pytest.raises(core.PingstoreError, match="backup incomplete.*resume"):
        core.coordinated_backup(
            source, DESTINATION, dry_run=False, confirm=plan["plan_hash"]
        )
    assert new.name not in state["catalog"]["runs"]
    assert state["control"]["operation"]["kind"] == "backup"
    state["fail_copy"] = False
    assert (
        core.coordinated_backup(
            source, DESTINATION, dry_run=False, confirm=plan["plan_hash"]
        )
        == 0
    )
    assert new.name in state["catalog"]["runs"]
    assert state["control"]["operation"] is None


def test_lost_backup_release_acknowledgement_is_reconciled_without_reupload(
    backup_store, monkeypatch
):
    repo, paths, state = backup_store
    source = repo / ".pingstore/runs"
    plan = core._coordinated_backup_plan(source, DESTINATION)
    release = core.release_remote_operation

    def lost_ack(*args, **kwargs):
        release(*args, **kwargs)
        raise OSError("lost backup release acknowledgement")

    monkeypatch.setattr(core, "release_remote_operation", lost_ack)
    with pytest.raises(core.PingstoreError, match="incomplete"):
        core.coordinated_backup(
            source, DESTINATION, dry_run=False, confirm=plan["plan_hash"]
        )
    assert state["control"]["operation"] is None
    assert (
        core.coordinated_backup(
            source, DESTINATION, dry_run=False, confirm=plan["plan_hash"]
        )
        == 0
    )
    assert state["calls"].count("catalog") == 1


def test_backup_skips_pruned_history_and_does_not_resurrect_it(backup_store):
    repo, paths, state = backup_store
    identity = paths["superseded"].name
    state["catalog"]["runs"].pop(identity)
    for name in list(state["objects"]):
        if name.startswith(identity + "/"):
            state["objects"].pop(name)
            state["bodies"].pop(name)
    state["control"]["tombstones"][identity] = {"plan_hash": "previous-prune"}
    plan = core._coordinated_backup_plan(repo / ".pingstore/runs", DESTINATION)
    assert identity in plan["excluded"]
    assert identity not in {row["run_id"] for row in plan["runs"]}
    assert not plan["new_runs"]


def test_current_runset_cannot_silently_restore_a_pruned_input(backup_store):
    repo, paths, state = backup_store
    identity = paths["hidden_source"].name
    state["control"]["tombstones"][identity] = {"plan_hash": "previous-prune"}
    with pytest.raises(
        core.PingstoreError, match="current or protected evidence was pruned"
    ):
        core._coordinated_backup_plan(repo / ".pingstore/runs", DESTINATION)


def test_changed_source_or_control_invalidates_backup_confirmation(backup_store):
    repo, paths, state = backup_store
    source = repo / ".pingstore/runs"
    plan = core._coordinated_backup_plan(source, DESTINATION)
    state["control"]["revision"] += 1
    with pytest.raises(core.PingstoreError, match="plan changed"):
        core.coordinated_backup(
            source, DESTINATION, dry_run=False, confirm=plan["plan_hash"]
        )
    assert not state["calls"]


def test_unchanged_allocation_watermarks_are_preserved_by_catalog_publication(
    backup_store,
):
    repo, paths, state = backup_store
    core.persist_watermarks(repo / ".pingstore", {"exp001": 999})
    source = repo / ".pingstore/runs"
    plan = core._coordinated_backup_plan(source, DESTINATION)
    core.coordinated_backup(
        source, DESTINATION, dry_run=False, confirm=plan["plan_hash"]
    )
    assert state["catalog"]["allocation_watermarks"]["exp001"] == 999


def test_catalog_publication_failure_keeps_uploaded_runs_and_the_recovery_lease(
    backup_store, monkeypatch
):
    repo, paths, state = backup_store
    source = repo / ".pingstore/runs"
    previous = copy.deepcopy(state["catalog"])
    plan = core._coordinated_backup_plan(source, DESTINATION)
    publish = core.replace_catalog
    monkeypatch.setattr(
        core,
        "replace_catalog",
        lambda *args: (_ for _ in ()).throw(
            core.PingstoreError("injected catalog conflict")
        ),
    )
    with pytest.raises(core.PingstoreError, match="incomplete"):
        core.coordinated_backup(
            source, DESTINATION, dry_run=False, confirm=plan["plan_hash"]
        )
    assert state["catalog"] == previous
    assert state["control"]["operation"]["kind"] == "backup"
    monkeypatch.setattr(core, "replace_catalog", publish)
    assert (
        core.coordinated_backup(
            source, DESTINATION, dry_run=False, confirm=plan["plan_hash"]
        )
        == 0
    )
