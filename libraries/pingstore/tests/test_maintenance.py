"""Fault-injected unified maintenance over both immutable evidence stores."""

import copy
import json
import shutil

import pytest
from pingstore import core

from .test_prune import fixture_store, make_run, reference, report

DESTINATION = "r2:bucket/pingstore/runs"


def known_publications():
    return dict(core.empty_publications(), known=True)


def catalog_for(repo):
    entries = {}
    for directory in (repo / ".pingstore/runs").iterdir():
        if not core.STAGE_ID_RE.fullmatch(directory.name):
            continue
        entries[directory.name] = dict(
            record=core.load_json(directory / "run.json"),
            files={
                name: row
                for name, row in core._maintenance_files(directory).items()
                if name != ".protected"
            },
        )
    return dict(schema=core.SCHEMA, updated_at="2026-10-27T00:00:00Z", runs=entries)


@pytest.fixture
def stores(tmp_path, monkeypatch):
    monkeypatch.setattr(
        core, "_verify_active_publication", lambda *args: dict(active=None, pending=[])
    )
    paths = fixture_store(tmp_path)
    state = dict(catalog=catalog_for(tmp_path), objects={}, calls=[], fail_delete=False)
    state["objects"] = {
        f"{identity}/{name}": row["bytes"]
        for identity, entry in state["catalog"]["runs"].items()
        for name, row in entry["files"].items()
    }
    state["control"] = dict(
        schema=core.CONTROL_SCHEMA,
        revision=1,
        operation=None,
        publications=known_publications(),
        tombstones={},
    )

    def read_catalog(destination, **kwargs):
        if not kwargs.get("allow_malformed"):
            core.validate_catalog(
                state["catalog"], complete=kwargs.get("complete", True)
            )
        return copy.deepcopy(state["catalog"])

    monkeypatch.setattr(core, "read_catalog", read_catalog)
    monkeypatch.setattr(core, "inventory", lambda _: dict(state["objects"]))
    monkeypatch.setattr(core, "read_control", lambda _: copy.deepcopy(state["control"]))

    def control(destination, previous, revised):
        assert previous == state["control"]
        state["control"] = copy.deepcopy(revised)

    monkeypatch.setattr(core, "replace_control", control)

    def catalog(destination, previous, revised):
        assert state["control"]["operation"] is not None
        assert previous == state["catalog"]
        core.validate_catalog(revised)
        state["catalog"] = copy.deepcopy(revised)
        state["calls"].append("catalog")

    monkeypatch.setattr(core, "replace_catalog", catalog)

    def remote(*args, **kwargs):
        assert args[0] == "delete"
        assert state["control"]["operation"] is not None
        assert state["calls"][0] == "catalog"
        for index, name in enumerate(args[-1].read_text().splitlines()):
            state["objects"].pop(name)
            if state["fail_delete"] and index == 0:
                raise core.PingstoreError("injected partial deletion")
        state["calls"].append("delete")

    monkeypatch.setattr(core, "rclone", remote)
    return tmp_path, paths, state


def test_default_both_plan_is_exact_and_dry_run_is_read_only(stores):
    repo, paths, state = stores
    plan = core.unified_prune_plan(repo, ["exp001"])
    assert plan["storage"] == "both"
    assert {row["run_id"] for row in plan["prune"]} == {
        paths["old_compute"].name,
        paths["old_present"].name,
    }
    assert set(plan["objects"]["local"]) == set(plan["objects"]["r2"])
    assert all(
        row["sha256"] and row["bytes"] >= 0 for row in plan["objects"]["r2"].values()
    )
    assert state["calls"] == []
    assert not (repo / ".pingstore/operations").exists()
    output = core.render_unified_prune_plan(plan)
    assert "--storage both" in output and "R2" not in output.splitlines()[0]
    assert paths["latest"].name in output and "current:auto:exp001" in output


def test_prune_both_publishes_before_deleting_then_verifies_both_inventories(stores):
    repo, paths, state = stores
    plan = core.unified_prune_plan(repo, ["exp001"])
    result = core.apply_unified_prune(repo, plan["plan_hash"], ["exp001"])
    assert result["phase"] == "complete"
    assert not paths["old_present"].exists() and not paths["old_compute"].exists()
    assert paths["latest"].exists()
    assert not set(plan["objects"]["r2"]) & state["objects"].keys()
    assert state["control"]["operation"] is None
    assert (
        state["control"]["tombstones"][paths["old_present"].name]["plan_hash"]
        == plan["plan_hash"]
    )
    assert state["catalog"]["allocation_watermarks"]["exp001"] == 3
    assert (
        core.apply_unified_prune(repo, plan["plan_hash"], ["exp001"])["phase"]
        == "complete"
    )


@pytest.mark.parametrize("extra_runs", [0, 8])
def test_archive_scan_count_does_not_grow_with_deletion_batch(
    stores, monkeypatch, extra_runs
):
    repo, _, _ = stores
    for number in range(10, 10 + extra_runs):
        make_run(
            repo,
            f"exp001-r{number:03d}-compute",
            stage="compute",
            created_at="2026-01-01T00:00:00Z",
        )
    options = dict(storage="local", publications=known_publications())
    plan = core.unified_prune_plan(repo, ["exp001"], **options)
    snapshot = core.maintenance_local_snapshot
    calls = []

    def counted_snapshot(root):
        calls.append(root)
        return snapshot(root)

    monkeypatch.setattr(core, "maintenance_local_snapshot", counted_snapshot)
    result = core.apply_unified_prune(repo, plan["plan_hash"], ["exp001"], **options)
    assert result["phase"] == "complete"
    assert len(result["deleted"]["local"]) == 2 + extra_runs
    # Rebuild the reviewed plan, validate before mutation, verify after the batch.
    assert len(calls) == 3


@pytest.mark.parametrize("change", ["payload", "protection", "missing-file"])
def test_next_candidate_is_checked_after_batch_starts(stores, monkeypatch, change):
    repo, paths, _ = stores
    options = dict(storage="local", publications=known_publications())
    plan = core.unified_prune_plan(repo, ["exp001"], **options)
    target = paths["old_present"]
    payload = target / "export/result.json"
    original = payload.read_bytes()
    remove = core.shutil.rmtree

    def change_next_target(directory, *args, **kwargs):
        remove(directory, *args, **kwargs)
        if directory.name == paths["old_compute"].name:
            if change == "payload":
                payload.write_bytes(b"changed")
            elif change == "protection":
                (target / ".protected").touch()
            else:
                payload.unlink()

    monkeypatch.setattr(core.shutil, "rmtree", change_next_target)
    with pytest.raises(core.PingstoreError, match="quarantined candidate changed"):
        core.apply_unified_prune(repo, plan["plan_hash"], ["exp001"], **options)
    journal = core.load_json(core._prune_journal_path(repo, plan["plan_hash"]))
    assert journal["deleted"]["local"] == [paths["old_compute"].name]
    staged = repo / ".pingstore/quarantine" / plan["plan_hash"][7:] / target.name
    assert staged.is_dir()
    assert paths["latest"].exists()
    monkeypatch.setattr(core.shutil, "rmtree", remove)
    (staged / "export/result.json").write_bytes(original)
    (staged / ".protected").unlink(missing_ok=True)
    save = core._journal_save
    checkpoint_counts = []

    def record_checkpoint(path, journal):
        if journal["phase"] == "local-deleting":
            checkpoint_counts.append(len(journal["deleted"]["local"]))
        save(path, journal)

    monkeypatch.setattr(core, "_journal_save", record_checkpoint)
    result = core.apply_unified_prune(repo, plan["plan_hash"], ["exp001"], **options)
    assert result["phase"] == "complete"
    assert not staged.exists()
    assert checkpoint_counts == [2]


def test_partial_local_deletion_resumes_quarantine_with_original_hash(
    stores, monkeypatch
):
    repo, paths, _ = stores
    options = dict(storage="local", publications=known_publications())
    plan = core.unified_prune_plan(repo, ["exp001"], **options)
    remove = core.shutil.rmtree

    def partial_delete(directory, *args, **kwargs):
        (directory / "export/result.json").unlink()
        raise OSError("injected partial local deletion")

    monkeypatch.setattr(core.shutil, "rmtree", partial_delete)
    with pytest.raises(core.PingstoreError, match="injected partial local deletion"):
        core.apply_unified_prune(repo, plan["plan_hash"], ["exp001"], **options)
    staged = (
        repo
        / ".pingstore/quarantine"
        / plan["plan_hash"][7:]
        / paths["old_compute"].name
    )
    assert staged.is_dir() and not (staged / "export/result.json").exists()
    monkeypatch.setattr(core.shutil, "rmtree", remove)
    result = core.apply_unified_prune(repo, plan["plan_hash"], ["exp001"], **options)
    assert result["phase"] == "complete"
    assert not staged.exists()
    assert paths["latest"].exists()


@pytest.mark.parametrize("change", ["report", "writer", "restored-source"])
def test_retention_controls_still_checked_between_deletions(
    stores, monkeypatch, change
):
    repo, paths, _ = stores
    if change == "restored-source":
        shutil.rmtree(repo / "experiments/exp001")
    options = dict(storage="local", publications=known_publications())
    plan = core.unified_prune_plan(repo, ["exp001"], **options)
    remove = core.shutil.rmtree

    def change_retention(directory, *args, **kwargs):
        remove(directory, *args, **kwargs)
        if directory.name == paths["old_compute"].name:
            if change == "report":
                report(repo, "exp001", ["exp001", "exp002"])
            elif change == "writer":
                import os

                hidden = repo / ".pingstore/runs/.exp005-r003-analyse.tmp"
                (hidden / ".writer.lock").write_text(str(os.getpid()))
            else:
                (repo / "experiments/exp001").mkdir()

    monkeypatch.setattr(core.shutil, "rmtree", change_retention)
    with pytest.raises(
        core.PingstoreError,
        match="report inputs changed|unfinished work changed|source restored",
    ):
        core.apply_unified_prune(repo, plan["plan_hash"], ["exp001"], **options)
    assert paths["old_present"].exists()


def test_prune_can_retire_all_runs_in_a_scope_without_current_evidence(stores):
    repo, paths, state = stores
    extra = make_run(
        repo,
        "exp003-r002-compute",
        stage="compute",
        created_at="2026-03-01T00:00:00Z",
    )
    state["catalog"] = catalog_for(repo)
    state["objects"] = {
        f"{identity}/{name}": row["bytes"]
        for identity, entry in state["catalog"]["runs"].items()
        for name, row in entry["files"].items()
    }
    other_objects = {
        name: size
        for name, size in state["objects"].items()
        if not name.startswith("exp003-")
    }
    other_runs = set(state["catalog"]["runs"]) - {paths["hpc"].name, extra.name}
    plan = core.unified_prune_plan(repo, ["exp003"])
    assert {row["run_id"] for row in plan["prune"]} == {
        paths["hpc"].name,
        extra.name,
    }
    result = core.apply_unified_prune(repo, plan["plan_hash"], ["exp003"])
    assert result["phase"] == "complete"
    assert not paths["hpc"].exists() and not extra.exists()
    assert state["objects"] == other_objects
    assert set(state["catalog"]["runs"]) == other_runs
    assert paths["latest"].exists()


def test_explicitly_removed_experiment_prunes_its_last_presentation_in_both_stores(
    stores,
):
    repo, paths, state = stores
    shutil.rmtree(repo / "experiments/exp001")
    plan = core.unified_prune_plan(repo, ["exp001"])
    identities = {paths[key].name for key in ("old_compute", "old_present", "latest")}
    assert plan["retired_experiments"] == ["exp001"]
    assert {row["run_id"] for row in plan["prune"]} == identities
    assert "Retired experiments: exp001" in core.render_unified_prune_plan(plan)
    result = core.apply_unified_prune(repo, plan["plan_hash"], ["exp001"])
    assert result["phase"] == "complete"
    assert all(
        not paths[key].exists() for key in ("old_compute", "old_present", "latest")
    )
    assert not identities & state["catalog"]["runs"].keys()
    assert not any(name.startswith("exp001-") for name in state["objects"])
    assert paths["exp2_latest"].exists()
    assert state["catalog"]["allocation_watermarks"]["exp001"] == 3


@pytest.mark.parametrize(
    "root",
    ["source", "report-input", "protection", "consumer", "deployment", "unscoped"],
)
def test_removed_experiment_keeps_other_retention_roots(stores, root):
    repo, paths, state = stores
    shutil.rmtree(repo / "experiments/exp001")
    experiments = ["exp001"]
    if root == "source":
        (repo / "experiments/exp001").mkdir()
    elif root == "report-input":
        report(repo, "exp002", ["exp002", "exp001"])
    elif root == "protection":
        (paths["latest"] / ".protected").write_text("keep")
    elif root == "consumer":
        make_run(
            repo,
            "exp002-r003-compute",
            stage="compute",
            created_at="2026-04-01T00:00:00Z",
            inputs={"source": reference(paths["latest"])},
        )
    elif root == "deployment":
        records = {
            identity: entry["record"]
            for identity, entry in state["catalog"]["runs"].items()
        }
        state["control"]["publications"]["active"] = manifest_for(
            records, [paths["latest"].name]
        )
    else:
        experiments = None
    plan = core.unified_prune_plan(repo, experiments)
    assert paths["latest"].name in {row["run_id"] for row in plan["keep"]}


def test_restored_source_invalidates_a_retirement_plan(stores):
    repo, paths, _ = stores
    shutil.rmtree(repo / "experiments/exp001")
    plan = core.unified_prune_plan(repo, ["exp001"])
    (repo / "experiments/exp001").mkdir()
    with pytest.raises(core.PingstoreError, match="source restored"):
        core._local_plan_controls(plan)
    assert paths["latest"].exists()


def test_partial_r2_deletion_retains_local_copies_and_resumes_original_hash(stores):
    repo, paths, state = stores
    plan = core.unified_prune_plan(repo, ["exp001"])
    state["fail_delete"] = True
    with pytest.raises(core.PingstoreError, match="cleanup incomplete.*resume"):
        core.apply_unified_prune(repo, plan["plan_hash"], ["exp001"])
    assert paths["old_present"].exists() and paths["old_compute"].exists()
    journal = core.load_json(core._prune_journal_path(repo, plan["plan_hash"]))
    assert len(journal["deleted"]["r2"]) == 1
    assert journal["phase"] == "catalog-published"
    assert state["control"]["operation"] is not None
    with pytest.raises(core.PingstoreError, match="unfinished remote operation"):
        core.unified_prune_plan(repo, ["exp001"])
    state["fail_delete"] = False
    assert (
        core.apply_unified_prune(repo, plan["plan_hash"], ["exp001"])["phase"]
        == "complete"
    )
    assert state["calls"].count("catalog") == 1


@pytest.mark.parametrize(
    "change", ["protection", "report", "objects", "catalog", "writer"]
)
def test_confirmation_revalidates_controls_before_any_mutation(stores, change):
    repo, paths, state = stores
    plan = core.unified_prune_plan(repo, ["exp001"])
    if change == "protection":
        (paths["old_present"] / ".protected").touch()
    elif change == "report":
        from .test_prune import report

        report(repo, "exp001", ["exp001", "exp002"])
    elif change == "objects":
        state["objects"][paths["old_present"].name + "/unexpected"] = 0
    elif change == "catalog":
        state["catalog"]["updated_at"] = "2026-10-28T00:00:00Z"
    else:
        import os

        hidden = repo / ".pingstore/runs/.exp005-r003-analyse.tmp"
        (hidden / ".writer.lock").write_text(str(os.getpid()))
    with pytest.raises(core.PingstoreError):
        core.apply_unified_prune(repo, plan["plan_hash"], ["exp001"])
    assert paths["old_present"].exists()
    assert state["calls"] == []


def test_ancestry_broken_only_in_deletion_candidates_is_allowed(stores):
    repo, paths, state = stores
    manifest = paths["old_present"] / "run.json"
    record = core.load_json(manifest)
    record["inputs"]["compute"]["run_id"] = "exp001-r999-compute"
    manifest.write_text(json.dumps(record))
    state["catalog"] = catalog_for(repo)
    state["objects"] = {
        f"{identity}/{name}": row["bytes"]
        for identity, entry in state["catalog"]["runs"].items()
        for name, row in entry["files"].items()
    }
    plan = core.unified_prune_plan(repo, ["exp001"])
    assert paths["old_present"].name in {row["run_id"] for row in plan["prune"]}
    result = core.apply_unified_prune(repo, plan["plan_hash"], ["exp001"])
    assert result["phase"] == "complete"
    core.validate_catalog(state["catalog"])


def manifest_for(records, identities, build="123/1"):
    identities = core.ancestry(records, set(identities))
    return dict(
        schema="pingstore.evidence/v1",
        build=dict(id=build, revision="a" * 40),
        selections={},
        runs={
            identity: {
                key: records[identity][key] for key in ("payload_digest", "inputs")
            }
            for identity in identities
        },
    )


def test_active_and_incoming_deployments_retain_exact_evidence(stores):
    repo, paths, state = stores
    records = {
        identity: entry["record"]
        for identity, entry in state["catalog"]["runs"].items()
    }
    state["control"]["publications"]["active"] = manifest_for(
        records, [paths["old_present"].name]
    )
    state["control"]["publications"]["incoming"] = {
        "456/1": manifest_for(records, [paths["latest"].name], "456/1")
    }
    plan = core.unified_prune_plan(repo, ["exp001"])
    assert not plan["prune"]
    kept = {row["run_id"]: row["reasons"] for row in plan["keep"]}
    assert "deployment:123/1" in kept[paths["old_compute"].name]
    assert "incoming:456/1" in kept[paths["latest"].name]
    state["control"]["publications"]["incoming"] = {}
    assert not core.unified_prune_plan(repo, ["exp001"])["prune"]


def test_unknown_publication_never_classifies_history_as_unused(stores):
    repo, paths, state = stores
    state["control"]["publications"]["known"] = False
    plan = core.unified_prune_plan(repo, ["exp001"])
    assert not plan["prune"] and not any(plan["objects"].values())
    assert any("unresolved-publication" in row["reasons"] for row in plan["keep"])


def test_out_of_scope_consumer_and_stage_scope_retain_shared_ancestry(stores):
    repo, paths, state = stores
    make_run(
        repo,
        "exp002-r099-compute",
        stage="compute",
        created_at="2026-03-01T00:00:00Z",
        inputs={"shared": reference(paths["old_compute"])},
    )
    plan = core.unified_prune_plan(repo, ["exp001"])
    assert paths["old_compute"].name in {row["run_id"] for row in plan["keep"]}
    assert not core.unified_prune_plan(repo, ["exp001"], ["compute"])["prune"]


def test_local_only_and_r2_only_are_explicit_and_leave_other_copy(stores):
    repo, paths, state = stores
    plan = core.unified_prune_plan(repo, ["exp001"], storage="r2")
    core.apply_unified_prune(repo, plan["plan_hash"], ["exp001"], storage="r2")
    assert paths["old_present"].exists()
    assert not set(plan["objects"]["r2"]) & state["objects"].keys()


def test_resume_rejects_scope_changes_and_new_protections(stores):
    repo, paths, state = stores
    plan = core.unified_prune_plan(repo, ["exp001"])
    state["fail_delete"] = True
    with pytest.raises(core.PingstoreError):
        core.apply_unified_prune(repo, plan["plan_hash"], ["exp001"])
    with pytest.raises(core.PingstoreError, match="scope differs"):
        core.apply_unified_prune(repo, plan["plan_hash"], ["exp001"], storage="r2")
    (paths["old_present"] / ".protected").touch()
    with pytest.raises(core.PingstoreError, match="candidate changed"):
        core.apply_unified_prune(repo, plan["plan_hash"], ["exp001"])
    assert paths["old_present"].exists()


def test_lost_lease_release_acknowledgement_is_resumable(stores, monkeypatch):
    repo, paths, state = stores
    plan = core.unified_prune_plan(repo, ["exp001"])
    release = core.release_remote_operation

    def lost_ack(*args, **kwargs):
        release(*args, **kwargs)
        raise OSError("release acknowledgement lost")

    monkeypatch.setattr(core, "release_remote_operation", lost_ack)
    with pytest.raises(core.PingstoreError, match="incomplete"):
        core.apply_unified_prune(repo, plan["plan_hash"], ["exp001"])
    assert state["control"]["operation"] is None
    assert (
        core.apply_unified_prune(repo, plan["plan_hash"], ["exp001"])["phase"]
        == "complete"
    )
    assert state["calls"].count("delete") == 1


def test_malformed_local_candidate_requires_an_explicit_bounded_identity(stores):
    repo, paths, state = stores
    malformed = repo / ".pingstore/runs/exp001-r099-compute"
    (malformed / "export").mkdir(parents=True)
    (malformed / "export/fragment.bin").write_bytes(b"incomplete")
    plan = core.unified_prune_plan(repo, ["exp001"])
    assert not plan["prune"]
    plan = core.unified_prune_plan(repo, ["exp001"], malformed_runs=[malformed.name])
    assert any(
        name.startswith(malformed.name + "/") for name in plan["objects"]["local"]
    )
    core.apply_unified_prune(
        repo, plan["plan_hash"], ["exp001"], malformed_runs=[malformed.name]
    )
    assert not malformed.exists()
    assert core.allocation_watermarks(repo / ".pingstore")["exp001"] == 99


def test_uncataloged_remote_objects_are_bounded_and_not_silently_deleted(stores):
    repo, paths, state = stores
    identity = "exp001-r099-compute"
    state["objects"][identity + "/export/partial.bin"] = 11
    plan = core.unified_prune_plan(repo, ["exp001"])
    assert not plan["prune"]
    plan = core.unified_prune_plan(repo, ["exp001"], malformed_runs=[identity])
    assert plan["objects"]["r2"][identity + "/export/partial.bin"] == {
        "bytes": 11,
        "sha256": None,
    }
    core.apply_unified_prune(
        repo, plan["plan_hash"], ["exp001"], malformed_runs=[identity]
    )
    assert identity + "/export/partial.bin" not in state["objects"]
    assert core.allocation_watermarks(repo / ".pingstore")["exp001"] == 99


def test_invalid_identity_never_becomes_a_directory_or_prefix_deletion(stores):
    repo, paths, state = stores
    for identity in ("../runs", "exp001", "exp001-r001-compute/extra"):
        with pytest.raises(core.PingstoreError, match="bounded"):
            core.unified_prune_plan(repo, ["exp001"], malformed_runs=[identity])
    assert not state["calls"]


def test_storage_scope_retains_the_only_input_copy_of_a_surviving_consumer(stores):
    repo, paths, state = stores
    make_run(
        repo,
        "exp001-r099-compute",
        stage="compute",
        created_at="2026-03-01T00:00:00Z",
        inputs={"shared": reference(paths["old_compute"])},
    )
    core.shutil.rmtree(paths["old_compute"])
    plan = core.unified_prune_plan(repo, ["exp001"], storage="r2")
    retained = next(
        row for row in plan["keep"] if row["run_id"] == paths["old_compute"].name
    )
    assert any("other-storage-copy" in reason for reason in retained["reasons"])
    assert all(
        not name.startswith(paths["old_compute"].name + "/")
        for name in plan["objects"]["r2"]
    )


def test_corrupt_surviving_catalog_in_a_journal_cannot_delete_retained_evidence(stores):
    repo, paths, state = stores
    plan = core.unified_prune_plan(repo, ["exp001"])
    state["fail_delete"] = True
    with pytest.raises(core.PingstoreError):
        core.apply_unified_prune(repo, plan["plan_hash"], ["exp001"])
    path = core._prune_journal_path(repo, plan["plan_hash"])
    journal = core.load_json(path)
    journal["revised_catalog"]["runs"].pop(paths["latest"].name)
    core.write_json_atomic(path, journal)
    before = dict(state["objects"])
    with pytest.raises(core.PingstoreError, match="surviving catalog differs"):
        core.apply_unified_prune(repo, plan["plan_hash"], ["exp001"])
    assert state["objects"] == before


def test_remote_protection_retains_local_copies_in_local_only_cleanup(stores):
    repo, paths, state = stores
    identity = paths["old_present"].name
    state["objects"][identity + "/.protected"] = 0
    snapshot = core._maintenance_remote_snapshot(DESTINATION, repo)
    plan = core.unified_prune_plan(
        repo, ["exp001"], storage="local", remote_snapshot=snapshot
    )
    assert not any(name.startswith(identity + "/") for name in plan["objects"]["local"])
    kept = next(row for row in plan["keep"] if row["run_id"] == identity)
    assert "remote-protection:" + identity in kept["reasons"]
