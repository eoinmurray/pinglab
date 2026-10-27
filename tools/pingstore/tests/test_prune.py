from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from pingstore import core
from pingstore.cli import main
from pingstore.core import allocation_watermarks
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


def local_plan(repo, experiments=None, stages=None):
    return core.unified_prune_plan(
        repo,
        experiments,
        stages,
        storage="local",
        publications=dict(core.empty_publications(), known=True),
    )


def local_apply(repo, plan, experiments=None, stages=None):
    return core.apply_unified_prune(
        repo,
        plan["plan_hash"],
        experiments,
        stages,
        storage="local",
        publications=dict(core.empty_publications(), known=True),
    )


def test_local_plan_retains_current_protected_unfinished_and_cross_scope_inputs(
    tmp_path,
):
    paths = fixture_store(tmp_path)
    (paths["old_present"] / ".protected").write_text("Historical control")
    plan = local_plan(tmp_path, ["exp001"])
    kept = {row["run_id"]: row["reasons"] for row in plan["keep"]}
    assert not plan["prune"]
    assert any(
        reason.startswith("required-by:") for reason in kept[paths["old_compute"].name]
    )
    assert "current:auto:exp001" in kept[paths["latest"].name]
    assert "out-of-scope" in kept[paths["superseded"].name]
    assert any(
        reason.startswith("unfinished:") for reason in kept[paths["hidden_source"].name]
    )


def test_local_prune_preserves_hidden_work_and_watermarks(tmp_path):
    paths = fixture_store(tmp_path)
    plan = local_plan(tmp_path, ["exp001"])
    assert not (tmp_path / ".pingstore/operations").exists()
    result = local_apply(tmp_path, plan, ["exp001"])
    assert result["phase"] == "complete"
    assert not paths["old_present"].exists()
    assert paths["latest"].exists()
    assert (tmp_path / ".pingstore/runs/.exp005-r003-analyse.tmp").exists()
    assert allocation_watermarks(tmp_path / ".pingstore")["exp001"] == 3


@pytest.mark.parametrize("marker", ["directory", "symlink", "broken-symlink"])
def test_unsafe_protection_markers_fail_closed(tmp_path, marker):
    paths = fixture_store(tmp_path)
    path = paths["old_present"] / ".protected"
    if marker == "directory":
        path.mkdir()
    else:
        path.symlink_to("README.md" if marker == "symlink" else "missing")
    with pytest.raises(PingstoreError):
        local_plan(tmp_path)


def test_new_records_and_scope_changes_invalidate_confirmation(tmp_path):
    fixture_store(tmp_path)
    plan = local_plan(tmp_path, ["exp001"])
    with pytest.raises(PingstoreError, match="plan changed"):
        local_apply(tmp_path, plan)
    make_run(
        tmp_path,
        "exp006-r001-present",
        stage="present",
        created_at="2026-03-01T00:00:00Z",
    )
    with pytest.raises(PingstoreError, match="plan changed"):
        local_apply(tmp_path, plan, ["exp001"])


def test_active_writer_blocks_deletion_without_losing_inputs(tmp_path):
    paths = fixture_store(tmp_path)
    hidden = tmp_path / ".pingstore/runs/.exp005-r003-analyse.tmp"
    (hidden / ".writer.lock").write_text(str(os.getpid()))
    plan = local_plan(tmp_path)
    with pytest.raises(PingstoreError, match="active writer"):
        local_apply(tmp_path, plan)
    assert paths["old_present"].exists()


def test_highest_removed_identity_never_reused(tmp_path):
    run = make_run(
        tmp_path,
        "exp006-r099-compute-phase-train",
        stage="compute",
        scope="phase",
        phase="train",
        created_at="2026-01-01T00:00:00Z",
    )
    plan = local_plan(tmp_path)
    local_apply(tmp_path, plan)
    assert not run.exists()
    assert (
        core.reserve_stage(tmp_path / ".pingstore", "exp006", "analyse")
        == "exp006-r100-analyse"
    )


def test_operation_lock_blocks_scientific_reservation(tmp_path):
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


def test_stage_scope_retains_dependencies_of_retained_presentations(tmp_path):
    paths = fixture_store(tmp_path)
    plan = local_plan(tmp_path, stages=["compute"])
    assert paths["old_compute"].name in {row["run_id"] for row in plan["keep"]}
    assert all(row["run_id"].endswith("-compute") for row in plan["prune"])


def test_unrelated_legacy_report_pin_does_not_block_current_reporting_or_pruning(
    tmp_path,
):
    paths = fixture_store(tmp_path)
    report(tmp_path, "exp124", ["exp124"], {"exp124": "exp124-r999-present"})
    plan = local_plan(tmp_path, ["exp001"])
    local_apply(tmp_path, plan, ["exp001"])
    assert not paths["old_present"].exists()


def test_partial_local_quarantine_deletion_resumes_original_plan(tmp_path, monkeypatch):
    paths = fixture_store(tmp_path)
    plan = local_plan(tmp_path, ["exp001"])
    remove = core.shutil.rmtree
    calls = []

    def fail_once(path):
        if not calls:
            calls.append(path)
            next(path.glob("export/*.json")).unlink()
            raise OSError("injected local deletion failure")
        remove(path)

    monkeypatch.setattr(core.shutil, "rmtree", fail_once)
    with pytest.raises(PingstoreError, match="cleanup incomplete"):
        local_apply(tmp_path, plan, ["exp001"])
    assert paths["latest"].exists()
    assert local_apply(tmp_path, plan, ["exp001"])["phase"] == "complete"
    assert not paths["old_present"].exists()


@pytest.mark.parametrize(
    "arguments",
    [
        ["prune", "--dry-run"],
        ["prune-remote", "--experiment", "exp001", "--dry-run"],
        ["backup-papers", "--dry-run"],
    ],
)
def test_cli_requires_explicit_scope_and_removes_retired_commands(arguments):
    with pytest.raises(SystemExit):
        main(arguments)
