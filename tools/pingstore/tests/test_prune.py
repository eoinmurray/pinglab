from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from pingstore import prune as prune_module
from pingstore import stages
from pingstore.cli import main
from pingstore.contracts import PingstoreError, payload_digest
from pingstore.locking import operation_lock
from pingstore.prune import apply_plan, build_plan, is_hpc_run


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
        "execution": {},
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
    return locals()


def test_plan_keeps_hpc_latest_visible_incomplete_inputs_and_ancestry(tmp_path):
    paths = fixture_store(tmp_path)
    plan = build_plan(tmp_path)
    kept = {row["run_id"]: row["reasons"] for row in plan["keep"]}
    pruned = {row["run_id"] for row in plan["prune"]}

    assert "latest-visible" in kept[paths["latest"].name]
    assert "hpc" in kept[paths["hpc"].name]
    assert "hpc" in kept[paths["imported_hpc"].name]
    assert any(
        reason.startswith("incomplete-input:")
        for reason in kept[paths["hidden_source"].name]
    )
    assert paths["old_compute"].name in pruned
    assert paths["old_present"].name in pruned
    assert paths["superseded"].name in pruned
    assert plan["plan_hash"].startswith("sha256:")
    assert len(plan["plan_hash"]) == 71


@pytest.mark.parametrize(
    "record",
    [
        {"origin": "slurm-wilkes"},
        {"origin": "local", "execution": {"host": "slurm-csd3-1"}},
        {"origin": "local", "historical_import": {"producer_host": "gpu-q-9"}},
    ],
)
def test_hpc_recognition_uses_authoritative_provenance(record):
    assert is_hpc_run(record)


def test_hpc_recognition_does_not_use_commands_paths_or_notes():
    assert not is_hpc_run(
        {
            "origin": "local",
            "execution": {"command": ["load-exp001-slurm"]},
            "historical_import": {"producer": {"note": "copied from HPC"}},
        }
    )


def test_confirm_applies_exact_plan_and_preserves_hidden_directory(tmp_path):
    paths = fixture_store(tmp_path)
    plan = build_plan(tmp_path)
    applied = apply_plan(tmp_path, plan["plan_hash"])

    assert applied == plan
    assert not paths["old_compute"].exists()
    assert not paths["old_present"].exists()
    assert paths["latest"].exists()
    assert (tmp_path / ".pingstore/runs/.exp005-r003-analyse.tmp").exists()
    assert not list((tmp_path / ".pingstore").glob(".prune-*-runs.*"))


def test_confirm_rejects_plan_drift_without_removing_runs(tmp_path):
    paths = fixture_store(tmp_path)
    plan = build_plan(tmp_path)
    make_run(
        tmp_path,
        "exp006-r001-present",
        stage="present",
        created_at="2026-01-04T00:00:00Z",
    )

    with pytest.raises(PingstoreError, match="plan changed"):
        apply_plan(tmp_path, plan["plan_hash"])
    assert paths["old_present"].exists()


def test_confirm_rejects_active_writer(tmp_path):
    fixture_store(tmp_path)
    hidden = tmp_path / ".pingstore/runs/.exp005-r003-analyse.tmp"
    (hidden / ".writer.lock").write_text(f"{os.getpid()}\n")
    plan = build_plan(tmp_path)

    with pytest.raises(PingstoreError, match="active writer"):
        apply_plan(tmp_path, plan["plan_hash"])


def test_failed_post_swap_validation_restores_original_store(tmp_path, monkeypatch):
    paths = fixture_store(tmp_path)
    plan = build_plan(tmp_path)
    validate = prune_module._validate_survivors
    calls = 0

    def fail_after_swap(runs, expected):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise PingstoreError("post-swap failure")
        return validate(runs, expected)

    monkeypatch.setattr(prune_module, "_validate_survivors", fail_after_swap)
    with pytest.raises(PingstoreError, match="post-swap failure"):
        apply_plan(tmp_path, plan["plan_hash"])
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


def test_prune_lock_blocks_reservation_and_execution(tmp_path):
    store = tmp_path / ".pingstore"
    identity = stages.reserve_stage(store, "exp001", "compute")
    with operation_lock(store, exclusive=True):
        with pytest.raises(PingstoreError, match="another Pingstore operation"):
            stages.reserve_stage(store, "exp001", "compute")
        with pytest.raises(PingstoreError, match="another Pingstore operation"):
            with stages.stage_run(tmp_path, "exp001", "compute", run_id=identity):
                pass
