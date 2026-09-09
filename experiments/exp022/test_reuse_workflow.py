"""Exercise bank execution boundaries with small, structurally real scientific data."""

from __future__ import annotations

import copy
import json
import shutil
import socket
import subprocess
from pathlib import Path

import pytest
import torch
from experiments.exp022 import campaign, compute, recipe, reuse
from experiments.exp022.test_reuse_contract import bank as bank
from experiments.helpers.checkpoints import public_provenance, resolve_checkpoint
from pingstore.contracts import (
    PingstoreError,
    file_sha256,
    load_json,
    validate_operational_run_directory,
    write_json_atomic,
)
from pingstore.prune import _hidden_inputs
from pingstore.stages import reserve_stage, source_run


@pytest.fixture
def workspace(bank: Path, monkeypatch) -> Path:
    registry = bank / "experiments/collections/registry.json"
    registry.parent.mkdir(parents=True)
    write_json_atomic(registry, {
        "schema": "pingstore.experiment-registry/v1",
        "experiments": {"exp022": "gamma-gated-sparsity"},
        "historical": {},
    })
    (bank / "uv.lock").write_text("fixture lockfile\n")
    monkeypatch.setattr(campaign, "git_identity", lambda repo: ("a" * 40, False))
    monkeypatch.setattr(compute, "REPO", bank)
    monkeypatch.setattr(recipe, "SNN_TOOL", bank / "tools/snnsim/tool.py")
    monkeypatch.delenv("PINGLAB_NB022_PLUMBING", raising=False)
    return bank


def _writer(repo: Path, run_id: str) -> Path:
    return repo / ".pingstore/runs" / f".{run_id}.tmp"


def _manifest(repo: Path, run_id: str) -> dict:
    return campaign.load_manifest(_writer(repo, run_id) / ".scratch/reuse/campaign.json")


def _write_training(manifest: dict, row: dict) -> Path:
    """A complete production-shape checkpoint with compact tensor storage."""
    directory = Path(row["output_directory"])
    directory.mkdir(parents=True, exist_ok=True)
    expected = campaign._expected_config(row)
    identity = {
        "training_cell_name": row["name"], "training_run_id": row["training_run_id"],
        "campaign_resolved_parameters": row["parameters"],
    }
    roles = ("W_in", "W_out", "W_EE_1", "W_EI_1", "W_IE_1", "W_II_1")
    initialization = {
        role: {"distribution": "constant", "zeros_remain_trainable": True,
               "requested_initial_zero_fraction": 0.0, "statistics": {"n_parameters": 1}}
        for role in roles
    }
    initialization["W_in"]["requested_initial_zero_fraction"] = expected.get(
        "w_in_initial_zero_fraction", 0.0
    )
    config = {**expected, **identity, "weight_initialization": initialization,
              "git_sha": "fixture-training-commit", "device": "cuda:0"}
    epochs = row["parameters"]["epochs"]
    history = [{"ep": ep, "samples": round(row["parameters"]["max_samples"] * 0.9),
                "acc": 10.0} for ep in range(1, epochs + 1)]
    write_json_atomic(directory / "config.json", config)
    (directory / "metrics.jsonl").write_text("".join(json.dumps(ep) + "\n" for ep in history))
    shapes = {"W_ff.0": (784, 1024), "W_ff.1": (1024, 10),
              "W_ei.1": (1024, 256), "W_ie.1": (256, 1024)}
    checkpoints = {}
    for role, filename, epoch, value in (
        ("best_validation", "weights.pth", 1, 1.0),
        ("final_epoch", "weights_final.pth", epochs, 2.0),
    ):
        torch.save({key: torch.tensor(value).expand(shape) for key, shape in shapes.items()},
                   directory / filename)
        checkpoints[role] = {"filename": filename, "epoch": epoch,
                             "sha256": file_sha256(directory / filename)}
    write_json_atomic(directory / "metrics.json", {
        **identity, "config": {**expected, "weight_initialization": initialization},
        "best_epoch": 1, "checkpoints": checkpoints,
        "weight_final": {role: {"zero_fraction": 0.0} for role in roles},
    })
    attempt = campaign.run_record_base(manifest, row)
    attempt.update(state="complete", exit_code=0, attempt_id=f"fixture-{row['name']}",
                   ended_at_utc=campaign.utc_now())
    write_json_atomic(campaign.status_path(manifest, row["name"]), attempt)
    return directory


def _complete_training(repo: Path, run_id: str) -> dict:
    manifest = _manifest(repo, run_id)
    for row in manifest["cells"]:
        _write_training(manifest, row)
    return manifest


@pytest.fixture
def diagnostics(monkeypatch) -> list[dict]:
    calls = []

    def generate(export: Path, destination: Path) -> None:
        calls.append({"export": export, "destination": destination})
        for cell in recipe.CANONICAL_CELLS:
            if cell["seed"] != 42:
                continue
            target = destination / cell["name"]
            target.mkdir(parents=True)
            (target / "recording.npz").write_bytes(b"fixture recording " + cell["name"].encode())
            write_json_atomic(target / "probe-command.json", {
                "command": ["fixture-simulation", "--dt", str(cell["dt_ms"])],
                "checkpoint": public_provenance(resolve_checkpoint(
                    export / cell["name"], recipe.RESULT_CHECKPOINT_ROLE)),
            })

    monkeypatch.setattr(compute, "generate_snapshots", generate)
    # The scientific NPZ contract has separate tests; these exercise writer boundaries.
    monkeypatch.setattr(reuse, "_validate_recording", lambda path, name: None)
    return calls


def test_allocation_pins_source_and_only_twelve_workers_remain_ineligible(workspace: Path) -> None:
    run_id = reuse.reserve(workspace)
    directory = _writer(workspace, run_id)
    record = load_json(directory / "run.json")
    manifest = _manifest(workspace, run_id)
    expected = {cell["name"] for cell in reuse.replacement_cells()}
    assert len(expected) == 12
    assert {row["name"] for row in manifest["cells"]} == expected
    assert manifest["selection"] == {"tier": "refractory-replacement"}
    assert record["origin"] == "slurm-wilkes"
    source = source_run(workspace / ".pingstore", "exp022-r001-compute")
    assert record["inputs"] == {"retained_bank": source.reference}
    reasons = {source.record["run_id"]: set()}
    _hidden_inputs(directory.parent, {source.record["run_id"]: source.record}, reasons)
    assert f"incomplete-input:{directory.name}" in reasons[source.record["run_id"]]
    result = reuse.status(workspace, run_id)
    assert not result["consumable"]
    assert len(result["reused_cells"]) == 90
    assert len(result["new_cells"]) == 12
    assert len(result["diagnostics"]) == 34
    assert not (directory.parent / run_id).exists()
    with pytest.raises((PingstoreError, FileNotFoundError)):
        source_run(workspace / ".pingstore", run_id)
    with pytest.raises(PingstoreError):
        validate_operational_run_directory(directory)


def _contents(directory: Path) -> dict[str, bytes | None]:
    return {path.relative_to(directory).as_posix(): path.read_bytes() if path.is_file() else None
            for path in directory.rglob("*")}


def test_preallocated_identity_is_initialized_without_allocating_another(
    workspace: Path, capsys,
) -> None:
    run_id = reserve_stage(workspace / ".pingstore", "exp022", "compute", origin="slurm-wilkes")
    directory = _writer(workspace, run_id)
    reservation = (directory / ".reservation.json").read_bytes()
    before_ids = {path.name for path in directory.parent.iterdir()}
    assert reuse.handle_cli(["--reuse-reserve", "--run-id", run_id], workspace)
    assert capsys.readouterr().out.strip() == run_id
    assert {path.name for path in directory.parent.iterdir()} == before_ids
    assert (directory / ".reservation.json").read_bytes() == reservation
    record = load_json(directory / "run.json")
    source = source_run(workspace / ".pingstore", "exp022-r001-compute")
    assert record["run_id"] == run_id
    assert record["inputs"] == {"retained_bank": source.reference}
    assert _manifest(workspace, run_id)["pingstore_run_id"] == run_id
    assert len(reuse.status(workspace, run_id)["new_cells"]) == 12


@pytest.mark.parametrize("damage", ["origin", "experiment", "stage", "identity", "schema"])
def test_preallocated_reservation_contract_mismatch_preserves_bytes(
    workspace: Path, damage: str,
) -> None:
    run_id = reserve_stage(workspace / ".pingstore", "exp022", "compute", origin="slurm-wilkes")
    directory = _writer(workspace, run_id)
    path = directory / ".reservation.json"
    reservation = load_json(path)
    key, value = {
        "origin": ("origin", "local"), "experiment": ("experiment", "exp023"),
        "stage": ("stage", "analyse"), "identity": ("run_id", "exp022-r999-compute"),
        "schema": ("schema", "pingstore.run/v3"),
    }[damage]
    reservation[key] = value
    write_json_atomic(path, reservation)
    before = _contents(directory)
    with pytest.raises(PingstoreError, match="reservation|identity"):
        reuse.reserve(workspace, run_id=run_id)
    assert _contents(directory) == before


@pytest.mark.parametrize("damage", ["run_record", "scratch", "export", "unexpected", "readme_type", "export_type"])
def test_preallocated_reservation_must_be_empty_and_untouched(
    workspace: Path, damage: str,
) -> None:
    run_id = reserve_stage(workspace / ".pingstore", "exp022", "compute", origin="slurm-wilkes")
    directory = _writer(workspace, run_id)
    if damage == "run_record":
        write_json_atomic(directory / "run.json", {"partial": True})
    elif damage == "scratch":
        (directory / ".scratch").mkdir()
    elif damage == "export":
        (directory / "export/partial.npz").write_bytes(b"incomplete scientific output")
    elif damage == "unexpected":
        (directory / "unexpected.txt").write_text("retained writer state")
    elif damage == "readme_type":
        (directory / "README.md").unlink()
        (directory / "README.md").mkdir()
    else:
        (directory / "export").rmdir()
        (directory / "export").write_bytes(b"invalid export entry")
    before = _contents(directory)
    with pytest.raises(PingstoreError, match="fresh unused reservation"):
        reuse.reserve(workspace, run_id=run_id)
    assert _contents(directory) == before


def test_preallocated_identity_cannot_be_reinitialized(workspace: Path) -> None:
    run_id = reuse.reserve(workspace)
    directory = _writer(workspace, run_id)
    before = _contents(directory)
    with pytest.raises(PingstoreError, match="fresh unused reservation"):
        reuse.reserve(workspace, run_id=run_id)
    assert _contents(directory) == before


def test_plan_rejects_existing_identity_without_initializing(workspace: Path, capsys) -> None:
    run_id = reserve_stage(workspace / ".pingstore", "exp022", "compute", origin="slurm-wilkes")
    directory = _writer(workspace, run_id)
    before = _contents(directory)
    with pytest.raises(SystemExit) as exc:
        reuse.handle_cli(["--reuse-plan", "--run-id", run_id], workspace)
    assert exc.value.code == 2
    assert "plan does not accept" in capsys.readouterr().err
    assert _contents(directory) == before


@pytest.mark.parametrize("reason", ["dirty", "plumbing"])
def test_invalid_allocation_does_not_claim_an_identity(workspace: Path, monkeypatch, reason: str) -> None:
    before = sorted(path.name for path in (workspace / ".pingstore/runs").iterdir())
    if reason == "dirty":
        monkeypatch.setattr(campaign, "git_identity", lambda repo: ("a" * 40, True))
    else:
        monkeypatch.setenv("PINGLAB_NB022_PLUMBING", "1")
    with pytest.raises(PingstoreError):
        reuse.reserve(workspace)
    assert sorted(path.name for path in (workspace / ".pingstore/runs").iterdir()) == before


def test_full_finalization_preserves_roles_and_exports_exact_102_plus_34(
    workspace: Path, diagnostics,
) -> None:
    run_id = reuse.reserve(workspace)
    manifest = _complete_training(workspace, run_id)
    recovered_name = manifest["cells"][0]["name"]
    original_attempt = load_json(campaign.status_path(manifest, recovered_name))
    write_json_atomic(campaign.lock_path(manifest, recovered_name),
                      {"attempt_id": original_attempt["attempt_id"]})
    assert reuse.train_cell(workspace, run_id, recovered_name, recover_stale=True) == 0
    original = workspace / ".pingstore/runs/exp022-r001-compute"
    original_bytes = {str(p.relative_to(original)): p.read_bytes()
                      for p in original.rglob("*") if p.is_file()}
    plan = load_json(_writer(workspace, run_id) / "run.json")["bank_reuse"]["plan"]
    assert reuse.finalize(workspace, run_id) == run_id
    finished = original.parent / run_id
    record = validate_operational_run_directory(finished)
    assert {p.name for p in finished.iterdir()} == {"run.json", "README.md", "export"}
    assert len([p for p in (finished / "export").iterdir() if p.is_dir()]) == 102
    assert len([p for p in (finished / "export").iterdir() if p.is_file()]) == 34
    assert len([p for p in (finished / "export").rglob("*") if p.is_file()]) == 442
    assert len(diagnostics) == 1
    for name in plan["reused_cells"]:
        for role in reuse.CELL_FILES:
            assert (finished / "export" / name / role).read_bytes() == (original / "export" / name / role).read_bytes()
    for cell in recipe.CANONICAL_CELLS:
        unit = finished / "export" / cell["name"]
        assert {p.name for p in unit.iterdir()} == set(reuse.CELL_FILES)
        best = resolve_checkpoint(unit, "best_validation")
        final = resolve_checkpoint(unit, "final_epoch")
        assert best["sha256"] != final["sha256"]
        assert final["epoch"] == 50
    for row in manifest["cells"]:
        provenance = record["bank_reuse"]["cell_provenance"][row["name"]]
        assert provenance["operation"] == "train"
        assert provenance["attempt"]["state"] == "complete"
        unit = finished / "export" / row["name"]
        config = load_json(unit / "config.json")
        metrics = load_json(unit / "metrics.json")
        assert len(metrics["epochs"]) == 50
        assert "campaign_resolved_parameters" not in config
        assert "git_sha" not in config
        assert "device" not in config
        assert config["refractory_e_ms"] == 1.2
        assert config["refractory_i_ms"] == 0.6
        assert provenance["execution_metadata"]["config.json"]["root"]["git_sha"] == "fixture-training-commit"
        assert provenance["training_output_files"]["config.json"]["sha256"] != provenance["files"]["config.json"]["sha256"]
    recovered = record["bank_reuse"]["cell_provenance"][recovered_name]
    assert recovered["attempt"] == original_attempt
    assert recovered["recovery"]["operation"] == "recover-completed-cell"
    assert recovered["recovery"]["attempt_id"] != original_attempt["attempt_id"]
    assert all(row["checkpoint"]["role"] == "final_epoch"
               for row in record["bank_reuse"]["diagnostic_provenance"].values())
    assert original_bytes == {str(p.relative_to(original)): p.read_bytes()
                              for p in original.rglob("*") if p.is_file()}
    assert not _writer(workspace, run_id).exists()
    for operation in (reuse.status, reuse.finalize):
        with pytest.raises(PingstoreError, match="immutable"):
            operation(workspace, run_id)


@pytest.mark.parametrize("damage", ["missing", "unexpected", "partial", "missing_attempt", "failed_attempt"])
def test_incomplete_or_unproven_training_cannot_finalize(workspace: Path, diagnostics, damage: str) -> None:
    run_id = reuse.reserve(workspace)
    manifest = _complete_training(workspace, run_id)
    row = manifest["cells"][0]
    if damage == "missing":
        shutil.rmtree(Path(row["output_directory"]))
    elif damage == "unexpected":
        (Path(manifest["campaign_root"]) / "cells/unexpected").mkdir()
    elif damage == "partial":
        (Path(row["output_directory"]) / "weights_final.pth").unlink()
    elif damage == "missing_attempt":
        campaign.status_path(manifest, row["name"]).unlink()
    else:
        path = campaign.status_path(manifest, row["name"])
        attempt = load_json(path)
        attempt.update(state="failed", exit_code=1)
        write_json_atomic(path, attempt)
    with pytest.raises(PingstoreError):
        reuse.finalize(workspace, run_id)
    assert not diagnostics
    assert _writer(workspace, run_id).is_dir()
    assert not (_writer(workspace, run_id).parent / run_id).exists()


def test_reused_cells_are_never_dispatched(workspace: Path, monkeypatch) -> None:
    run_id = reuse.reserve(workspace)
    dispatched = []
    monkeypatch.setattr(compute, "_campaign_train", lambda *args, **kw: dispatched.append(args))
    with pytest.raises(PingstoreError, match="only the 12"):
        reuse.train_cell(workspace, run_id, "ping__canonical__seed42")
    assert not dispatched


def test_failed_worker_records_failure_and_remains_ineligible(workspace: Path, monkeypatch) -> None:
    run_id = reuse.reserve(workspace)
    manifest = _manifest(workspace, run_id)
    name = manifest["cells"][0]["name"]
    monkeypatch.setattr(compute, "_gpu_metadata", lambda: {})
    monkeypatch.setattr(compute.subprocess, "run", lambda command, **kw: subprocess.CompletedProcess(command, 7))
    assert reuse.train_cell(workspace, run_id, name) == 1
    attempt = load_json(campaign.status_path(manifest, name))
    assert attempt["state"] == "failed"
    assert attempt["exit_code"] == 7
    assert not campaign.lock_path(manifest, name).exists()
    assert not reuse.status(workspace, run_id)["consumable"]


def test_complete_cell_with_stale_owner_needs_explicit_recovery(workspace: Path, monkeypatch) -> None:
    run_id = reuse.reserve(workspace)
    manifest = _manifest(workspace, run_id)
    row = manifest["cells"][0]
    _write_training(manifest, row)
    lock = campaign.lock_path(manifest, row["name"])
    write_json_atomic(lock, {"attempt_id": "old"})
    attempt = campaign.run_record_base(manifest, row)
    attempt.update(attempt_id="old", state="running", hostname=socket.gethostname(), pid=999_999_999)
    write_json_atomic(campaign.status_path(manifest, row["name"]), attempt)
    monkeypatch.setattr(compute, "_campaign_train", lambda *a, **kw: pytest.fail("completed cell retrained"))
    with pytest.raises(RuntimeError, match="stale"):
        reuse.train_cell(workspace, run_id, row["name"])
    assert lock.exists()
    assert reuse.train_cell(workspace, run_id, row["name"], recover_stale=True) == 0
    assert not lock.exists()
    recovered = load_json(campaign.status_path(manifest, row["name"]))
    assert recovered["state"] == "complete"
    assert recovered["operation"] == "recover-completed-cell"
    assert recovered["previous_attempt"]["attempt_id"] == "old"


def test_active_attempt_cannot_be_recovered(workspace: Path) -> None:
    run_id = reuse.reserve(workspace)
    manifest = _manifest(workspace, run_id)
    row = manifest["cells"][0]
    _write_training(manifest, row)
    attempt, lock = campaign.acquire_attempt(manifest, row)
    try:
        with pytest.raises(RuntimeError, match="active"):
            reuse.train_cell(workspace, run_id, row["name"], recover_stale=True)
    finally:
        campaign.release_attempt(lock, attempt["attempt_id"])


def test_scheduler_ownership_query_includes_suspended_and_stopped_jobs(monkeypatch) -> None:
    def query(command, **kwargs):
        assert command[0] == "squeue"
        assert "--states=all" in command
        return subprocess.CompletedProcess(command, 0, stdout="123\n")

    monkeypatch.setattr(campaign.subprocess, "run", query)
    assert campaign.attempt_is_active({"state": "running", "slurm_job_id": "123"}) is True


def test_worker_and_finalizer_are_mutually_exclusive(workspace: Path, monkeypatch) -> None:
    run_id = reuse.reserve(workspace)
    name = _manifest(workspace, run_id)["cells"][0]["name"]

    def worker(*args, **kwargs):
        with pytest.raises(PingstoreError, match="active"):
            reuse.finalize(workspace, run_id)
        return 0

    monkeypatch.setattr(compute, "_campaign_train", worker)
    assert reuse.train_cell(workspace, run_id, name) == 0
    with reuse._bank_lock(workspace, exclusive=True):
        with pytest.raises(PingstoreError, match="active"):
            reuse.train_cell(workspace, run_id, name)


def test_snapshot_failure_retains_training_and_explicit_retry_regenerates(
    workspace: Path, diagnostics, monkeypatch,
) -> None:
    run_id = reuse.reserve(workspace)
    manifest = _complete_training(workspace, run_id)
    generate = compute.generate_snapshots

    def fail(export: Path, destination: Path):
        destination.mkdir(parents=True)
        (destination / "partial").write_bytes(b"interrupted")
        raise RuntimeError("injected simulation failure")

    monkeypatch.setattr(compute, "generate_snapshots", fail)
    with pytest.raises(RuntimeError, match="injected simulation"):
        reuse.finalize(workspace, run_id)
    for row in manifest["cells"]:
        assert campaign.validate_cell(row)["valid"]
    record = load_json(_writer(workspace, run_id) / "run.json")
    assert not record["bank_reuse"].get("prepared_export")
    with pytest.raises(PingstoreError, match="recover-finalization"):
        reuse.finalize(workspace, run_id)
    monkeypatch.setattr(compute, "generate_snapshots", generate)
    assert reuse.finalize(workspace, run_id, recover=True) == run_id
    assert len(diagnostics) == 1


@pytest.mark.parametrize("failure", ["scratch_cleanup", "validation_after_reservation_removal", "rename"])
def test_prepared_bank_recovers_after_late_failure_without_rerunning_diagnostics(
    workspace: Path, diagnostics, monkeypatch, failure: str,
) -> None:
    run_id = reuse.reserve(workspace)
    _complete_training(workspace, run_id)
    directory = _writer(workspace, run_id)
    with monkeypatch.context() as failure_patch:
        if failure == "scratch_cleanup":
            original_rmtree = shutil.rmtree

            def remove(path, *args, **kwargs):
                if Path(path) == directory / ".scratch":
                    original_rmtree(path, *args, **kwargs)
                    raise OSError("injected late failure")
                return original_rmtree(path, *args, **kwargs)

            failure_patch.setattr(reuse.shutil, "rmtree", remove)
        elif failure == "validation_after_reservation_removal":
            def validate(path):
                assert not (directory / ".scratch").exists()
                assert not (directory / ".reservation.json").exists()
                raise OSError("injected late failure")

            failure_patch.setattr(reuse, "validate_operational_run_directory", validate)
        else:
            def rename(source, destination):
                raise OSError("injected late failure")

            failure_patch.setattr(reuse.os, "rename", rename)
        with pytest.raises(OSError, match="injected late"):
            reuse.finalize(workspace, run_id)
    record = load_json(directory / "run.json")
    assert record["bank_reuse"]["prepared_export"]
    assert not (directory / ".scratch").exists()
    assert not (directory.parent / run_id).exists()
    assert reuse.status(workspace, run_id)["training"] is None
    with pytest.raises(PingstoreError, match="only be finalized"):
        reuse.train_cell(workspace, run_id, record["bank_reuse"]["plan"]["new_cells"][0])
    assert reuse.finalize(workspace, run_id, recover=True) == run_id
    assert len(diagnostics) == 1
    final = validate_operational_run_directory(directory.parent / run_id)
    attempts = final["bank_reuse"]["finalization_attempts"]
    assert len(attempts) == 2
    assert "error" in attempts[0]
    assert attempts[1]["recovery"]


def test_tampered_prepared_export_is_never_published(workspace: Path, diagnostics, monkeypatch) -> None:
    run_id = reuse.reserve(workspace)
    _complete_training(workspace, run_id)
    directory = _writer(workspace, run_id)
    with monkeypatch.context() as fail_patch:
        fail_patch.setattr(reuse, "validate_operational_run_directory",
                           lambda path: (_ for _ in ()).throw(OSError("late failure")))
        with pytest.raises(OSError, match="late failure"):
            reuse.finalize(workspace, run_id)
    name = load_json(directory / "run.json")["bank_reuse"]["plan"]["new_cells"][0]
    (directory / "export" / name / "weights_final.pth").write_bytes(b"changed")
    with pytest.raises(PingstoreError, match="prepared export changed"):
        reuse.finalize(workspace, run_id, recover=True)
    assert len(diagnostics) == 1
    assert not (directory.parent / run_id).exists()


def test_source_drift_after_reservation_blocks_workers_and_finalization(workspace: Path) -> None:
    run_id = reuse.reserve(workspace)
    directory = _writer(workspace, run_id)
    source = directory.parent / "exp022-r001-compute/export/ping__canonical__seed42/weights.pth"
    source.write_bytes(b"changed source")
    for operation in (reuse.status, reuse.finalize):
        with pytest.raises(PingstoreError, match="checksum"):
            operation(workspace, run_id)
    assert not (directory.parent / run_id).exists()


@pytest.mark.parametrize("damage", ["duplicate", "unexpected"])
def test_rehashed_worker_manifest_drift_is_rejected(workspace: Path, damage: str) -> None:
    run_id = reuse.reserve(workspace)
    directory = _writer(workspace, run_id)
    path = directory / ".scratch/reuse/campaign.json"
    manifest = _manifest(workspace, run_id)
    if damage == "duplicate":
        manifest["cells"][1] = copy.deepcopy(manifest["cells"][0])
    else:
        manifest["cells"][0]["name"] = "ping__canonical__seed42"
    manifest.pop("manifest_sha256")
    campaign.write_manifest(path, manifest)
    record = load_json(directory / "run.json")
    record["bank_reuse"]["campaign"] = campaign.load_manifest(path)
    write_json_atomic(directory / "run.json", record)
    with pytest.raises(SystemExit, match="duplicate|declared selection"):
        reuse.status(workspace, run_id)
