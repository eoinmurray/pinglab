from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from pingstore import core
from pingstore.cli import main
from pingstore.core import allocation_watermarks, apply_prune, prune_plan
from pingstore.library import PingstoreError, operation_lock, payload_digest


def report(repo, article, inputs, pins=None):
    path = repo / "experiments" / article / "report/report.mdx"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "---\n"
        + json.dumps({"id": article, "inputs": inputs, "runs": pins or {}})
        + "\n---\n"
    )


def make_run(
    repo: Path,
    run_id: str,
    *,
    stage: str,
    created_at: str,
    inputs: dict | None = None,
    **metadata,
) -> Path:
    directory = repo / ".pingstore/runs" / run_id
    (directory / "export").mkdir(parents=True)
    (directory / "README.md").write_text("Run history\n")
    (directory / "export/result.json").write_text(json.dumps({"run": run_id}))
    record = {
        "schema": "pingstore.run/v4",
        "run_id": run_id,
        "experiment": run_id.split("-")[0],
        "collection": "fixture",
        "stage": stage,
        "origin": "local",
        "created_at": created_at,
        "inputs": inputs or {},
        "execution": {"started_at": created_at, "completed_at": created_at},
        "provenance": {},
        **metadata,
    }
    record["payload_digest"] = payload_digest(directory)
    (directory / "run.json").write_text(json.dumps(record))
    return directory


def reference(directory: Path) -> dict:
    record = json.loads((directory / "run.json").read_text())
    return {"run_id": record["run_id"], "payload_digest": record["payload_digest"]}


def fixture_store(repo: Path) -> dict[str, Path]:
    old_compute = make_run(
        repo, "exp001-r001-compute", stage="compute", created_at="2026-01-01T00:00:00Z"
    )
    old_present = make_run(
        repo,
        "exp001-r002-present",
        stage="present",
        created_at="2026-01-02T00:00:00Z",
        inputs={"compute": reference(old_compute)},
    )
    latest = make_run(
        repo, "exp001-r003-present", stage="present", created_at="2026-01-03T00:00:00Z"
    )
    superseded = make_run(
        repo, "exp002-r001-compute", stage="compute", created_at="2026-01-01T00:00:00Z"
    )
    exp2_latest = make_run(
        repo, "exp002-r002-present", stage="present", created_at="2026-01-02T00:00:00Z"
    )
    hpc = make_run(
        repo,
        "exp003-r001-compute",
        stage="compute",
        created_at="2026-01-01T00:00:00Z",
        scientific_execution={"origin": "slurm", "scheduler": "slurm"},
    )
    imported_hpc = make_run(
        repo,
        "exp004-r001-compute",
        stage="compute",
        created_at="2026-01-01T00:00:00Z",
        historical_import={"producer": {"host": "gpu-q-13"}},
    )
    hidden_source = make_run(
        repo, "exp005-r001-compute", stage="compute", created_at="2026-01-01T00:00:00Z"
    )
    make_run(
        repo, "exp005-r002-present", stage="present", created_at="2026-01-02T00:00:00Z"
    )
    hidden = repo / ".pingstore/runs/.exp005-r003-analyse.tmp"
    hidden.mkdir()
    (hidden / "run.json").write_text(
        json.dumps({"inputs": {"compute": reference(hidden_source)}})
    )
    report(repo, "exp001", ["exp001"])
    report(repo, "exp002", ["exp002"])
    return locals()


def test_plan_keeps_used_and_incomplete_inputs_but_not_unused_hpc(tmp_path):
    paths = fixture_store(tmp_path)
    plan = prune_plan(tmp_path)
    kept = {row["run_id"]: row["reasons"] for row in plan["keep"]}
    pruned = {row["run_id"] for row in plan["prune"]}

    assert "used-by:exp001" in kept[paths["latest"].name]
    assert paths["hpc"].name in pruned
    assert paths["imported_hpc"].name in pruned
    assert any(
        reason.startswith("incomplete-input:")
        for reason in kept[paths["hidden_source"].name]
    )
    assert paths["old_compute"].name in pruned
    assert paths["old_present"].name in pruned
    assert paths["superseded"].name in pruned
    assert plan["plan_hash"].startswith("sha256:")
    assert len(plan["plan_hash"]) == 71


def test_scoped_plan_only_prunes_selected_experiment(tmp_path):
    paths = fixture_store(tmp_path)
    plan = prune_plan(tmp_path, ["exp001"])
    kept = {row["run_id"]: row["reasons"] for row in plan["keep"]}
    pruned = {row["run_id"] for row in plan["prune"]}

    assert plan["experiments"] == ["exp001"]
    assert pruned == {paths["old_compute"].name, paths["old_present"].name}
    assert "used-by:exp001" in kept[paths["latest"].name]
    assert "out-of-scope" in kept[paths["superseded"].name]
    assert "out-of-scope" in kept[paths["exp2_latest"].name]


def test_scoped_prune_allows_unrelated_missing_report_pin(tmp_path):
    paths = fixture_store(tmp_path)
    missing = "exp124-r164-present"
    report(tmp_path, "exp124", ["exp124"], {"exp124": missing})

    with pytest.raises(PingstoreError, match="names missing run"):
        prune_plan(tmp_path)
    plan = prune_plan(tmp_path, ["exp001"])
    assert plan["report_selections"]["exp124"]["runs"] == {"exp124": missing}
    assert {row["run_id"] for row in plan["prune"]} == {
        paths["old_compute"].name,
        paths["old_present"].name,
    }

    apply_prune(tmp_path, plan["plan_hash"], ["exp001"])
    assert not paths["old_compute"].exists()
    assert not paths["old_present"].exists()
    assert paths["latest"].exists()
    assert paths["superseded"].exists()


def test_scoped_plan_keeps_shared_report_inputs_despite_unrelated_missing_pin(tmp_path):
    paths = fixture_store(tmp_path)
    report(
        tmp_path,
        "exp110",
        ["exp001", "exp124"],
        {
            "exp001": paths["old_present"].name,
            "exp124": "exp124-r164-present",
        },
    )

    plan = prune_plan(tmp_path, ["exp001"])
    kept = {row["run_id"]: row for row in plan["keep"]}
    for name in ("old_compute", "old_present"):
        assert "exp110" in kept[paths[name].name]["used_by"]
    assert not plan["prune"]


@pytest.mark.parametrize(
    "experiment,target",
    [
        ("exp001", "exp999-r001-present"),
        ("exp999", "exp001-r999-present"),
    ],
)
def test_scoped_plan_rejects_missing_pin_declaring_or_naming_scoped_experiment(
    tmp_path,
    experiment,
    target,
):
    fixture_store(tmp_path)
    report(tmp_path, "exp110", [experiment], {experiment: target})
    with pytest.raises(PingstoreError, match="names missing run"):
        prune_plan(tmp_path, ["exp001"])


def test_scoped_confirm_detects_changes_to_unrelated_missing_pins(tmp_path):
    paths = fixture_store(tmp_path)
    report(tmp_path, "exp124", ["exp124"], {"exp124": "exp124-r164-present"})
    plan = prune_plan(tmp_path, ["exp001"])
    report(tmp_path, "exp124", ["exp124"], {"exp124": "exp124-r165-present"})
    with pytest.raises(PingstoreError, match="plan changed"):
        apply_prune(tmp_path, plan["plan_hash"], ["exp001"])
    assert paths["old_present"].exists()


def test_scoped_plan_keeps_selected_ancestor_of_out_of_scope_run(tmp_path):
    selected_parent = make_run(
        tmp_path,
        "exp001-r001-compute",
        stage="compute",
        created_at="2026-01-01T00:00:00Z",
    )
    make_run(
        tmp_path,
        "exp001-r002-present",
        stage="present",
        created_at="2026-01-02T00:00:00Z",
    )
    make_run(
        tmp_path,
        "exp002-r001-present",
        stage="present",
        created_at="2026-01-03T00:00:00Z",
        inputs={"shared": reference(selected_parent)},
    )

    plan = prune_plan(tmp_path, ["exp001"])
    kept = {row["run_id"]: row["reasons"] for row in plan["keep"]}

    assert "current:exp002" in kept[selected_parent.name]
    assert not plan["prune"]


def test_scoped_plan_rejects_unknown_or_invalid_experiment(tmp_path):
    fixture_store(tmp_path)

    with pytest.raises(PingstoreError, match="invalid experiment filter"):
        prune_plan(tmp_path, ["experiment-one"])
    with pytest.raises(PingstoreError, match="experiment filter has no runs"):
        prune_plan(tmp_path, ["exp999"])


def test_confirm_applies_exact_plan_and_preserves_hidden_directory(tmp_path):
    paths = fixture_store(tmp_path)
    plan = prune_plan(tmp_path)
    applied = apply_prune(tmp_path, plan["plan_hash"])

    assert applied == plan
    assert not paths["old_compute"].exists()
    assert not paths["old_present"].exists()
    assert paths["latest"].exists()
    assert (tmp_path / ".pingstore/runs/.exp005-r003-analyse.tmp").exists()
    assert not list((tmp_path / ".pingstore").glob(".prune-*-runs.*"))


def test_scoped_confirm_applies_only_matching_scoped_plan(tmp_path):
    paths = fixture_store(tmp_path)
    plan = prune_plan(tmp_path, ["exp001"])

    with pytest.raises(PingstoreError, match="plan changed"):
        apply_prune(tmp_path, plan["plan_hash"])
    applied = apply_prune(tmp_path, plan["plan_hash"], ["exp001"])

    assert applied == plan
    assert not paths["old_compute"].exists()
    assert not paths["old_present"].exists()
    assert paths["superseded"].exists()


def test_confirm_rejects_plan_drift_without_removing_runs(tmp_path):
    paths = fixture_store(tmp_path)
    plan = prune_plan(tmp_path)
    make_run(
        tmp_path,
        "exp006-r001-present",
        stage="present",
        created_at="2026-01-04T00:00:00Z",
    )

    with pytest.raises(PingstoreError, match="plan changed"):
        apply_prune(tmp_path, plan["plan_hash"])
    assert paths["old_present"].exists()


def test_confirm_rejects_active_writer(tmp_path):
    fixture_store(tmp_path)
    hidden = tmp_path / ".pingstore/runs/.exp005-r003-analyse.tmp"
    (hidden / ".writer.lock").write_text(f"{os.getpid()}\n")
    plan = prune_plan(tmp_path)

    with pytest.raises(PingstoreError, match="active writer"):
        apply_prune(tmp_path, plan["plan_hash"])


def test_failed_post_swap_validation_restores_original_store(tmp_path, monkeypatch):
    paths = fixture_store(tmp_path)
    plan = prune_plan(tmp_path)
    validate = core._validate_survivors
    calls = 0

    def fail_after_swap(runs, expected):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise PingstoreError("post-swap failure")
        return validate(runs, expected)

    monkeypatch.setattr(core, "_validate_survivors", fail_after_swap)
    with pytest.raises(PingstoreError, match="post-swap failure"):
        apply_prune(tmp_path, plan["plan_hash"])
    assert paths["old_compute"].exists()
    assert paths["old_present"].exists()
    assert paths["latest"].exists()
    assert not list((tmp_path / ".pingstore").glob(".prune-*-runs.*"))


def test_cli_requires_dry_run_or_complete_hash(tmp_path, capsys):
    fixture_store(tmp_path)
    lock = tmp_path / ".pingstore/.operation.lock"
    assert not lock.exists()
    assert main(["prune", "--root", str(tmp_path), "--dry-run"]) == 0
    output = capsys.readouterr().out
    assert "KEEP" in output and "PRUNE" in output and "Confirm with:" in output
    before = {path.name for path in (tmp_path / ".pingstore/runs").iterdir()}
    assert not lock.exists()

    assert main(["prune", "--root", str(tmp_path), "--confirm", "yes"]) == 1
    assert "complete sha256 plan hash" in capsys.readouterr().err
    assert before == {path.name for path in (tmp_path / ".pingstore/runs").iterdir()}


def test_cli_renders_repeatable_experiment_scope_in_confirmation(tmp_path, capsys):
    fixture_store(tmp_path)

    assert (
        main(
            [
                "prune",
                "--root",
                str(tmp_path),
                "--experiment",
                "exp002",
                "--experiment",
                "exp001",
                "--dry-run",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "Scope: exp001, exp002" in output
    assert "--experiment exp001 --experiment exp002 --confirm sha256:" in output


def test_prune_lock_blocks_reservation_and_execution(tmp_path):
    store = tmp_path / ".pingstore"
    identity = core.reserve_stage(store, "exp001", "compute")
    with operation_lock(store, exclusive=True):
        with pytest.raises(PingstoreError, match="another Pingstore operation"):
            core.reserve_stage(store, "exp001", "compute")
        with (
            pytest.raises(PingstoreError, match="another Pingstore operation"),
            core.stage_run(tmp_path, "exp001", "compute", run_id=identity),
        ):
            pass


def test_used_is_transitive_global_and_independent_of_manual_protection(tmp_path):
    paths = fixture_store(tmp_path)
    report(tmp_path, "exp110", ["exp001"], {"exp001": paths["old_present"].name})
    (paths["exp2_latest"] / ".protected").write_text("Keep this historical result\n")
    plan = prune_plan(tmp_path, ["exp001"])
    kept = {row["run_id"]: row for row in plan["keep"]}
    assert "exp110" in kept[paths["old_compute"].name]["used_by"]
    assert kept[paths["old_present"].name]["used"]
    assert kept[paths["exp2_latest"].name]["protected"]
    assert not plan["prune"]


def test_manual_protection_retains_ancestry_without_marking_used(tmp_path):
    paths = fixture_store(tmp_path)
    (paths["old_present"] / ".protected").touch()
    plan = prune_plan(tmp_path)
    kept = {row["run_id"]: row for row in plan["keep"]}
    for name in ("old_compute", "old_present"):
        assert kept[paths[name].name]["protected"]
        assert not kept[paths[name].name]["used"]
    apply_prune(tmp_path, plan["plan_hash"])
    assert (paths["old_present"] / ".protected").is_file()


@pytest.mark.parametrize("change", ["add", "remove", "edit"])
def test_protection_changes_invalidate_confirmation(tmp_path, change):
    paths = fixture_store(tmp_path)
    marker = paths["old_present"] / ".protected"
    if change != "add":
        marker.write_text("Keep\n")
    plan = prune_plan(tmp_path)
    if change == "remove":
        marker.unlink()
    else:
        marker.write_text("Revised\n")
    with pytest.raises(PingstoreError, match="plan changed"):
        apply_prune(tmp_path, plan["plan_hash"])
    assert paths["old_present"].exists()


@pytest.mark.parametrize("kind", ["directory", "symlink", "broken-symlink"])
def test_invalid_protection_marker_fails_closed(tmp_path, kind):
    paths = fixture_store(tmp_path)
    marker = paths["old_present"] / ".protected"
    if kind == "directory":
        marker.mkdir()
    else:
        marker.symlink_to("README.md" if kind == "symlink" else "missing")
    with pytest.raises(PingstoreError):
        prune_plan(tmp_path)


def test_report_pin_changes_invalidate_confirmation(tmp_path):
    paths = fixture_store(tmp_path)
    plan = prune_plan(tmp_path)
    report(tmp_path, "exp001", ["exp001"], {"exp001": paths["old_present"].name})
    with pytest.raises(PingstoreError, match="plan changed"):
        apply_prune(tmp_path, plan["plan_hash"])


@pytest.mark.parametrize(
    "target", ["exp999-r001-present", "exp001-r001-compute", "exp002-r002-present"]
)
@pytest.mark.parametrize("experiments", [None, ["exp001"]])
def test_missing_wrong_stage_or_wrong_experiment_pins_fail_closed(
    tmp_path, target, experiments
):
    fixture_store(tmp_path)
    report(tmp_path, "exp001", ["exp001"], {"exp001": target})
    with pytest.raises(PingstoreError, match="report"):
        prune_plan(tmp_path, experiments)


def test_unpinned_report_uses_completion_time_and_can_have_unavailable_inputs(tmp_path):
    paths = fixture_store(tmp_path)
    record_path = paths["old_present"] / "run.json"
    record = json.loads(record_path.read_text())
    record["execution"]["completed_at"] = "2026-02-01T00:00:00Z"
    record_path.write_text(json.dumps(record))
    report(tmp_path, "exp001", ["exp001", "exp999"])
    plan = prune_plan(tmp_path)
    assert plan["report_selections"]["exp001"]["runs"] == {
        "exp001": paths["old_present"].name
    }
    assert paths["latest"].name in {row["run_id"] for row in plan["prune"]}


def test_prune_all_unused_keeps_counters_outside_runs_and_dry_run_is_read_only(
    tmp_path,
):
    run = make_run(
        tmp_path,
        "exp006-r099-compute-phase-train",
        stage="compute",
        scope="phase",
        phase="train",
        created_at="2026-01-01T00:00:00Z",
    )
    plan = prune_plan(tmp_path)
    assert not plan["keep"]
    assert not (tmp_path / ".pingstore/.allocation").exists()
    apply_prune(tmp_path, plan["plan_hash"])
    assert not run.exists()
    assert (
        core.reserve_stage(tmp_path / ".pingstore", "exp006", "analyse")
        == "exp006-r100-analyse"
    )
    assert allocation_watermarks(tmp_path / ".pingstore")["exp006"] == 100


def test_reservation_counter_survives_removal_of_last_hidden_run(tmp_path):
    import shutil

    store = tmp_path / ".pingstore"
    identity = core.reserve_stage(store, "exp006", "compute")
    shutil.rmtree(store / "runs" / f".{identity}.tmp")
    assert (
        core.reserve_stage(store, "exp006", "compute", phase="train")
        == "exp006-r002-compute-phase-train"
    )


def test_stage_scope_retains_dependencies_of_runs_outside_stage_scope(tmp_path):
    paths = fixture_store(tmp_path)
    plan = prune_plan(tmp_path, stages=["compute"])
    assert paths["old_compute"].name in {row["run_id"] for row in plan["keep"]}
    assert all(row["stage"] == "compute" for row in plan["prune"])


def test_latest_unreported_presentation_is_current_without_collection_promotion(
    tmp_path,
):
    run = make_run(
        tmp_path,
        "exp006-r001-present",
        stage="present",
        created_at="2026-01-01T00:00:00Z",
    )
    (tmp_path / ".pingstore/collections.json").write_text(
        json.dumps({"views": [run.name]})
    )
    plan = prune_plan(tmp_path)
    assert not plan["prune"]
    assert [row["run_id"] for row in plan["keep"]] == [run.name]
    assert plan["keep"][0]["reasons"] == ["current:exp006"]


def test_protection_marker_does_not_change_payload_or_backup_inventory(
    tmp_path, monkeypatch
):
    from pingstore import core
    from pingstore.library import validate_operational_run_directory

    paths = fixture_store(tmp_path)
    run = paths["old_present"]
    before = validate_operational_run_directory(run)["payload_digest"]
    files = core.local_files(run.parent, run.name)
    (run / ".protected").write_text("Historical evidence\n")
    assert validate_operational_run_directory(run)["payload_digest"] == before
    assert core.local_files(run.parent, run.name) == files


@pytest.mark.parametrize("value", [-1, True, "99", None])
def test_corrupt_allocation_state_blocks_prune_and_reservation(tmp_path, value):
    fixture_store(tmp_path)
    state = tmp_path / ".pingstore/.allocation/exp001.json"
    state.parent.mkdir()
    state.write_text(json.dumps({"high_watermark": value}))
    with pytest.raises(PingstoreError, match="allocation counter"):
        prune_plan(tmp_path)
    with pytest.raises(PingstoreError, match="allocation counter"):
        core.reserve_stage(tmp_path / ".pingstore", "exp001", "compute")


def test_protection_added_during_confirmation_prevents_deletion(tmp_path, monkeypatch):
    paths = fixture_store(tmp_path)
    plan = prune_plan(tmp_path)
    validate = core._validate_survivors

    def add_marker(runs, expected):
        validate(runs, expected)
        (paths["old_present"] / ".protected").touch()

    monkeypatch.setattr(core, "_validate_survivors", add_marker)
    with pytest.raises(PingstoreError, match="protection marker changed"):
        apply_prune(tmp_path, plan["plan_hash"])
    assert paths["old_present"].exists()
    assert paths["old_compute"].exists()


def test_manuscript_uses_all_selected_experiments_and_shared_ancestors(tmp_path):
    paths = fixture_store(tmp_path)
    report(
        tmp_path,
        "exp110",
        ["exp001", "exp002"],
        {"exp001": paths["old_present"].name, "exp002": paths["exp2_latest"].name},
    )
    plan = prune_plan(tmp_path, ["exp001"])
    manuscript = {row["run_id"] for row in plan["keep"] if "exp110" in row["used_by"]}
    assert manuscript == {
        paths[name].name for name in ("old_present", "old_compute", "exp2_latest")
    }
    assert not manuscript.intersection(row["run_id"] for row in plan["prune"])


def test_cli_all_matches_default_and_rejects_experiment_combination(tmp_path, capsys):
    fixture_store(tmp_path)
    assert main(["prune", "--root", str(tmp_path), "--dry-run"]) == 0
    default = capsys.readouterr().out
    assert main(["prune", "--all", "--root", str(tmp_path), "--dry-run"]) == 0
    assert capsys.readouterr().out == default
    with pytest.raises(SystemExit):
        main(["prune", "--all", "--experiment", "exp001", "--dry-run"])


def test_backup_ignores_local_and_remote_protection_markers(tmp_path, monkeypatch):
    from pingstore import core

    fixture_store(tmp_path)
    source = tmp_path / ".pingstore/runs"
    records, _ = core.discover_store(source)
    remote = {}
    for identity in records:
        remote.update(core.local_files(source, identity))
    remote["exp001-r003-present/.protected"] = 99
    (source / "exp001-r002-present/.protected").touch()
    monkeypatch.setattr(core, "inventory", lambda _: remote)

    def remote_cat(*args, **kwargs):
        assert args[0] == "cat"
        identity, filename = args[1].split("/")[-2:]
        return (source / identity / filename).read_bytes()

    monkeypatch.setattr(core, "rclone", remote_cat)
    plan = core.backup_plan(source, "remote:pingstore/runs")
    assert not plan["new_runs"]
    assert len(plan["existing_runs"]) == len(records)


@pytest.mark.parametrize("broken", ["missing", "changed"])
def test_unused_broken_input_can_be_pruned_but_retained_input_must_validate(
    tmp_path, broken
):
    paths = fixture_store(tmp_path)
    parent = reference(paths["old_compute"])
    if broken == "missing":
        parent["run_id"] = "exp001-r999-compute"
    else:
        parent["payload_digest"] = "sha256:" + "0" * 64
    orphan = make_run(
        tmp_path,
        "exp001-r004-compute",
        stage="compute",
        created_at="2026-01-04T00:00:00Z",
        inputs={"source": parent},
    )
    plan = prune_plan(tmp_path)
    assert orphan.name in {row["run_id"] for row in plan["prune"]}
    (orphan / ".protected").touch()
    with pytest.raises(PingstoreError, match="missing|changed"):
        prune_plan(tmp_path)
    (orphan / ".protected").unlink()
    apply_prune(tmp_path, plan["plan_hash"])
    assert not orphan.exists()
    assert paths["latest"].exists()
