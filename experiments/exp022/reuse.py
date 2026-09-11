"""Exp022's pinned 84-cell reuse and 18-cell COBA damping replacement.

This is deliberately an experiment workflow, not a general bank merger. Training
and finalization are separate commands; only finalization makes a run visible.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path

from experiments.exp022 import compute as campaign
from experiments.exp022 import recipe
from experiments.exp022.checkpoints import public_provenance, resolve_checkpoint
from experiments.exp022.reuse_contract import inspect_source, replacement_cells
from pingstore.contracts import (
    RUN_SCHEMA,
    PingstoreError,
    file_sha256,
    load_json,
    payload_digest,
    validate_operational_run_directory,
    write_json_atomic,
)
from pingstore.layout import canonical_export_file, normalize_export_layout
from pingstore.membership import memberships
from pingstore.stages import (
    execution_origin,
    operation_lock,
    reserve_stage,
    stage_reservation,
)
from snnsim.timing import duration_steps

SCHEMA = "pinglab.exp022.gradient-damping-bank/v1"
CELL_FILES = ("config.json", "metrics.json", "weights.pth", "weights_final.pth")


@contextmanager
def _bank_lock(repo: Path, *, exclusive: bool):
    """Workers may fan out, but reservation/finalization owns the bank alone."""
    with operation_lock(repo / ".pingstore", exclusive=False):
        path = repo / ".pingstore/.exp022-reuse.lock"
        if path.is_symlink():
            raise PingstoreError("reuse lock must not be a symlink")
        descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            try:
                mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
                fcntl.flock(descriptor, mode | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise PingstoreError("another bank worker or finalizer is active") from exc
            yield
        finally:
            os.close(descriptor)


def _directory(repo: Path, run_id: str) -> Path:
    if not re.fullmatch(r"exp022-r[0-9]{3,}-compute", run_id):
        raise PingstoreError("reuse requires a reserved exp022 compute identity")
    directory = repo / ".pingstore/runs" / f".{run_id}.tmp"
    if (directory.parent / run_id).exists():
        raise PingstoreError("completed runs are immutable; this identity is already visible")
    if not directory.is_dir() or any(p.is_symlink() for p in (directory, *directory.parents)):
        raise PingstoreError("reuse requires a real incomplete reserved writer")
    if any(p.is_symlink() for p in directory.rglob("*")):
        raise PingstoreError("reuse writer contains a symlink")
    return directory


def _history(directory: Path, message: str) -> None:
    with (directory / "README.md").open("a") as handle:
        handle.write(f"- {campaign.utc_now()}: {message}\n")


def reserve(repo: Path, *, origin: str = "slurm-wilkes", run_id: str | None = None) -> str:
    """Initialize a fresh reservation, pin its source, and create 18 worker rows.

    A supplied identity must already have been allocated with ``reserve_stage``;
    this permits copying that exact reservation into the frozen HPC checkout.
    """
    from experiments.exp022 import compute

    with _bank_lock(repo, exclusive=True):
        source, plan = inspect_source(repo)
        # Resolve the clean production contract before allocating an identity.
        if os.environ.get("PINGLAB_NB022_PLUMBING"):
            raise PingstoreError("the replacement bank requires production settings")
        commit, dirty = campaign.git_identity(repo)
        if dirty:
            raise PingstoreError("bank reservation requires a clean source worktree")
        if run_id is None:
            run_id = reserve_stage(repo / ".pingstore", recipe.SLUG, "compute", origin=origin)
        directory = _directory(repo, run_id)
        if ({path.name for path in directory.iterdir()} != {"README.md", "export", ".reservation.json"}
                or not (directory / "README.md").is_file()
                or not (directory / ".reservation.json").is_file()
                or not (directory / "export").is_dir()
                or any((directory / "export").iterdir())):
            raise PingstoreError("bank initialization requires a fresh unused reservation with an empty export")
        reservation = stage_reservation(directory)
        if (reservation["run_id"] != run_id or reservation["experiment"] != recipe.SLUG
                or reservation["stage"] != "compute" or reservation["origin"] != origin):
            raise PingstoreError("preallocated reservation identity or execution origin mismatch")
        record = {
            "schema": RUN_SCHEMA, "run_id": run_id, "experiment": recipe.SLUG,
            "collection": memberships(repo)[recipe.SLUG], "stage": "compute",
            "origin": origin, "created_at": reservation["reserved_at"],
            "inputs": {"retained_bank": source.reference},
            "execution": {
                "operation": "gradient-damping-bank-reuse", "command": [sys.executable, *sys.argv],
                "cwd": str(repo), "host": execution_origin(),
                "started_at": campaign.utc_now(), "configuration": recipe.SCALE,
            },
            "provenance": {"git_commit": commit, "dirty": False, "code_dirty": False,
                           "lockfile_sha256": campaign.lock_identity(repo)["sha256"]},
            "bank_reuse": {"schema": SCHEMA, "plan": plan, "reservation": reservation},
        }
        # Even an allocation interrupted before manifest creation protects its parent.
        write_json_atomic(directory / "run.json", record)
        root = directory / ".scratch/reuse"
        manifest = campaign.create_manifest(
            repo=repo, bank_root=root, bank_id=run_id,
            cells=replacement_cells(), tier_for=recipe.cell_resource_tier,
            samples_epochs=recipe.cell_samples_epochs, build_args=recipe.build_train_args,
            scientific_contract_for=recipe.scientific_contract,
            selection_tier="exp110-coba-damping-replacement",
        )
        manifest["pingstore_run_id"] = run_id
        campaign.write_manifest(root / "bank.json", manifest)
        manifest = campaign.load_manifest(root / "bank.json")
        record["bank_reuse"]["campaign"] = manifest
        write_json_atomic(directory / "run.json", record)
        _history(directory, f"allocated `{run_id}` with origin `{origin}` from clean Git commit "
                 f"`{commit}`; retained input `{source.reference['run_id']}` at "
                 f"`{source.reference['payload_digest']}`. The 84 reused cells retain their "
                 "source-bank bytes and training origins. Reserved a complete 102-cell "
                 "replacement bank: 84 cells will be copied byte-for-byte from the pinned "
                 "retained bank; 18 TR-02 COBA cells will be trained with gradient damping "
                 "1000 and the recurrent loop disabled. Both checkpoint roles and original "
                 "training origins are retained. "
                 "All 34 seed-42 diagnostics will be regenerated during explicit finalization.")
        compute._checked_bank_manifest(root / "bank.json")
        source.check_unchanged()
        return run_id


def _load(repo: Path, run_id: str):
    directory = _directory(repo, run_id)
    record = load_json(directory / "run.json")
    reuse = record.get("bank_reuse", {})
    if (record.get("schema") != RUN_SCHEMA or record.get("run_id") != run_id
            or record.get("experiment") != recipe.SLUG or record.get("stage") != "compute"
            or reuse.get("schema") != SCHEMA):
        raise PingstoreError("not an exp022 gradient-damping-bank reservation")
    source, plan = inspect_source(repo)
    if record.get("inputs") != {"retained_bank": source.reference} or reuse.get("plan") != plan:
        raise PingstoreError("reserved source or bank assembly contract changed")
    saved_reservation = reuse["reservation"]
    if (saved_reservation.get("schema") != RUN_SCHEMA
            or saved_reservation.get("run_id") != run_id
            or saved_reservation.get("origin") != record.get("origin")):
        raise PingstoreError("reuse reservation identity mismatch")
    if (directory / ".reservation.json").exists():
        if stage_reservation(directory) != saved_reservation:
            raise PingstoreError("reuse reservation changed")
    elif not reuse.get("prepared_export"):
        raise PingstoreError("training writer has lost its reservation")
    manifest = reuse.get("campaign")
    if not isinstance(manifest, dict):
        raise PingstoreError("allocation was interrupted before campaign creation; reserve a new identity")
    commit, dirty = campaign.git_identity(repo)
    if dirty or manifest["repository"] != {"commit": commit, "dirty": False}:
        raise PingstoreError("reuse execution requires the original clean source commit")
    if manifest["environment"]["lockfile"] != campaign.lock_identity(repo):
        raise PingstoreError("reuse environment lockfile changed")
    if not reuse.get("prepared_export"):
        path = directory / ".scratch/reuse/bank.json"
        if campaign.load_manifest(path) != manifest:
            raise PingstoreError("working campaign differs from the reserved contract")
        from experiments.exp022 import compute

        manifest = compute._checked_bank_manifest(path)
    return directory, record, source, plan, manifest


def status(repo: Path, run_id: str) -> dict:
    with _bank_lock(repo, exclusive=False):
        _directory_, record, source, plan, manifest = _load(repo, run_id)
        prepared = record["bank_reuse"].get("prepared_export")
        result = {
            "run_id": run_id, "source": source.reference,
            "reused_cells": plan["reused_cells"], "new_cells": plan["new_cells"],
            "diagnostics": plan["diagnostics"], "prepared_export": prepared,
            "training": None if prepared else campaign.summarize_status(manifest),
            "consumable": False,
        }
        source.check_unchanged()
        return result


def train_cell(repo: Path, run_id: str, name: str, *, recover_stale: bool = False) -> int:
    from experiments.exp022 import compute

    with _bank_lock(repo, exclusive=False):
        directory, record, source, plan, manifest = _load(repo, run_id)
        if record["bank_reuse"].get("prepared_export"):
            raise PingstoreError("assembled bank can only be finalized")
        if name not in plan["new_cells"]:
            raise PingstoreError("only the 18 replacement cells may be trained")
        row = campaign.manifest_cell(manifest, name)
        # A killed worker may have completed its files before releasing its lock.
        if campaign.lock_path(manifest, name).exists() and campaign.validate_cell(row)["valid"]:
            status_file = campaign.status_path(manifest, name)
            if not status_file.is_file():
                raise PingstoreError("cannot confirm the owner of a completed cell's attempt lock")
            attempt, lock = campaign.acquire_attempt(manifest, row, recover_stale=recover_stale)
            try:
                attempt.update(state="complete", ended_at_utc=campaign.utc_now(), exit_code=0)
                attempt["operation"] = "recover-completed-cell"
                attempt["note"] = "validated completed files; no training was performed by this recovery"
                campaign.atomic_json(campaign.status_path(manifest, name), attempt)
            finally:
                campaign.release_attempt(lock, attempt["attempt_id"])
            result = 0
        else:
            result = compute._train_bank_cell(
                directory / ".scratch/reuse/bank.json", name, recover_stale=recover_stale
            )
        source.check_unchanged()
        return result


def _file_record(path: Path) -> dict:
    return {"sha256": file_sha256(path), "size_bytes": path.stat().st_size}


def _inventory(export: Path) -> dict:
    if any(p.is_symlink() for p in export.rglob("*")):
        raise PingstoreError("assembled export contains a symlink")
    return {p.relative_to(export).as_posix(): _file_record(p)
            for p in sorted(export.rglob("*")) if p.is_file()}


def _check_reused(export: Path, plan: dict) -> None:
    for name in plan["reused_cells"]:
        for filename, expected in plan["per_cell"][name]["files"].items():
            path = export / name / filename
            if not path.is_file() or _file_record(path) != expected:
                raise PingstoreError(f"reused cell bytes changed: {name}/{filename}")


def _validate_recording(path: Path, name: str) -> None:
    import numpy as np

    cell = next(cell for cell in recipe.CANONICAL_CELLS if cell["name"] == name)
    steps = duration_steps(recipe.T_MS, cell["dt_ms"])
    try:
        with np.load(path, allow_pickle=False) as recording:
            if not np.isclose(float(recording["dt"]), cell["dt_ms"], rtol=1e-6, atol=0):
                raise ValueError("recorded timestep disagrees with the cell")
            for key, wanted in (("n_e", recipe.N_EXCITATORY), ("n_i", recipe.N_INHIBITORY), ("label", 0)):
                if int(recording[key]) != wanted:
                    raise ValueError(f"recorded {key} disagrees with the fixed digit-0 probe")
            for key, neurons in (("spk_e", recipe.N_EXCITATORY), ("spk_i", recipe.N_INHIBITORY)):
                spikes = recording[key]
                if spikes.shape != (steps, neurons) or not np.all((spikes == 0) | (spikes == 1)):
                    raise ValueError(f"invalid {key} raster dimensions or spike values")
    except (OSError, ValueError, KeyError) as exc:
        raise PingstoreError(f"invalid diagnostic recording for {name}: {exc}") from exc


def _project_new_cell(trained: Path, target: Path) -> dict:
    """Keep new scientific definitions in exports and execution records in run.json."""
    operational = {
        "resource_tier", "git_sha", "git_dirty", "torch_version", "device",
        "python_env_hash", "run_id", "started_at", "run_started_at", "run_finished_at",
        "total_elapsed_s", "perf", "hostname", "execution_origin",
    }
    removed = {}
    for filename in ("config.json", "metrics.json"):
        payload = load_json(target / filename)
        captures = {}
        for label, value in (("root", payload), ("config", payload.get("config", {}))):
            if not isinstance(value, dict):
                continue
            metadata = {
                key: value.pop(key)
                for key in list(value)
                if key in operational
                or key.startswith("campaign_")
                or key.startswith("bank_")
            }
            if metadata:
                captures[label] = metadata
        if filename == "metrics.json" and "epochs" not in payload:
            payload["epochs"] = [json.loads(line) for line in
                                 (trained / "metrics.jsonl").read_text().splitlines() if line]
        if captures:
            removed[filename] = captures
        write_json_atomic(target / filename, payload)
    return removed


def _assemble(directory: Path, record: dict, source, plan: dict, manifest: dict) -> None:
    from experiments.exp022 import compute

    root = directory / ".scratch/reuse"
    cells = root / "cells"
    actual = {p.name for p in cells.iterdir()} if cells.exists() else set()
    if actual != set(plan["new_cells"]):
        raise PingstoreError("training directories do not exactly match the 18 replacement cells")
    for row in manifest["cells"]:
        if campaign.lock_path(manifest, row["name"]).exists():
            raise PingstoreError(f"cell attempt still owns {row['name']}; recover it before finalization")
        validation = campaign.validate_cell(row)
        if not validation["valid"]:
            raise PingstoreError(f"invalid replacement {row['name']}: {validation['reasons']}")
        status_file = campaign.status_path(manifest, row["name"])
        if not status_file.is_file():
            raise PingstoreError(f"missing training execution record: {row['name']}")
        status_record = load_json(status_file)
        if (status_record.get("state") != "complete" or status_record.get("exit_code") != 0
                or status_record.get("bank_manifest_sha256") != manifest["manifest_sha256"]
                or status_record.get("cell_name") != row["name"]):
            raise PingstoreError(f"training execution record mismatch: {row['name']}")
    export = directory / "export"
    normalization = directory / ".normalized-export.tmp"
    if normalization.exists():
        shutil.rmtree(normalization)
    if export.exists():
        shutil.rmtree(export)
    export.mkdir()
    origins = dict(plan["per_cell"])
    for name in [*plan["reused_cells"], *plan["new_cells"]]:
        trained = source.unit(name) if name in plan["reused_cells"] else cells / name
        target = export / name
        target.mkdir()
        for filename in CELL_FILES:
            shutil.copy2(trained / filename, target / filename)
        if name in plan["new_cells"]:
            row = campaign.manifest_cell(manifest, name)
            status_file = campaign.status_path(manifest, name)
            attempt = load_json(status_file)
            recovered = attempt.get("operation") == "recover-completed-cell"
            metadata = _project_new_cell(trained, target)
            origins[name] = {
                "operation": "train", "source_run_id": record["run_id"],
                "parameters": row["parameters"], "command": row["command"],
                "repository": manifest["repository"], "environment": manifest["environment"],
                "attempt": attempt["previous_attempt"] if recovered else attempt,
                "recovery": attempt if recovered else None,
                "training_output_files": {filename: _file_record(trained / filename) for filename in CELL_FILES},
                "execution_metadata": metadata,
                "checkpoint_roles": {role: public_provenance(resolve_checkpoint(target, role))
                                     for role in ("best_validation", "final_epoch")},
                "files": {filename: _file_record(target / filename) for filename in CELL_FILES},
            }
    probes = root / "diagnostics"
    if probes.exists():
        shutil.rmtree(probes)
    compute.generate_snapshots(export, probes)
    if {p.name for p in probes.iterdir()} != set(plan["diagnostics"]):
        raise PingstoreError("diagnostics must contain exactly the 34 seed-42 cells")
    diagnostic_records = {}
    for name in plan["diagnostics"]:
        probe = probes / name
        recording = probe / "recording.npz"
        if not recording.is_file() or recording.stat().st_size == 0:
            raise PingstoreError(f"missing diagnostic recording: {name}")
        _validate_recording(recording, name)
        command = load_json(probe / "probe-command.json")
        checkpoint = public_provenance(resolve_checkpoint(export / name, recipe.RESULT_CHECKPOINT_ROLE))
        if command.get("checkpoint") != checkpoint:
            raise PingstoreError(f"diagnostic checkpoint mismatch: {name}")
        target = canonical_export_file(export, "snapshots", name, "recording.npz")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(recording, target)
        diagnostic_records[name] = {
            "checkpoint": checkpoint, "command": command["command"],
            "recording": target.relative_to(export).as_posix(),
            "execution_records": {p.name: load_json(p) for p in probe.glob("*.json")
                                  if p.name != "probe-command.json"},
        }
    record["bank_reuse"]["cell_provenance"] = origins
    record["bank_reuse"]["diagnostic_provenance"] = diagnostic_records
    normalize_export_layout(directory, record)
    _check_reused(export, plan)
    source.check_unchanged()
    # Everything needed to recover a failure during cleanup is now in export/run.json.
    record["bank_reuse"]["prepared_export"] = {
        "payload_digest": payload_digest(directory), "files": _inventory(export),
        "prepared_at": campaign.utc_now(),
    }
    write_json_atomic(directory / "run.json", record)
    _history(directory, "assembled and verified 102 cells with both checkpoint roles and "
             "34 regenerated seed-42 recordings; original source bytes remain unchanged.")


def finalize(repo: Path, run_id: str, *, recover: bool = False) -> str:
    """Explicitly assemble diagnostics and atomically expose the complete compute bank."""
    with _bank_lock(repo, exclusive=True):
        directory, record, source, plan, manifest = _load(repo, run_id)
        reuse = record["bank_reuse"]
        attempts = reuse.setdefault("finalization_attempts", [])
        if attempts and not recover:
            raise PingstoreError("interrupted finalization requires --recover-finalization")
        attempt = {"started_at": campaign.utc_now(), "command": [sys.executable, *sys.argv],
                   "host": execution_origin(), "recovery": recover}
        attempts.append(attempt)
        write_json_atomic(directory / "run.json", record)
        try:
            if not reuse.get("prepared_export"):
                _assemble(directory, record, source, plan, manifest)
                reuse = record["bank_reuse"]
                attempt = reuse["finalization_attempts"][-1]
            prepared = reuse["prepared_export"]
            if (_inventory(directory / "export") != prepared["files"]
                    or payload_digest(directory) != prepared["payload_digest"]):
                raise PingstoreError("prepared export changed; refusing finalization")
            _check_reused(directory / "export", plan)
            source.check_unchanged()
            record["payload_digest"] = prepared["payload_digest"]
            attempt["completed_at"] = campaign.utc_now()
            record["execution"]["completed_at"] = attempt["completed_at"]
            write_json_atomic(directory / "run.json", record)
            scratch = directory / ".scratch"
            if scratch.exists():
                shutil.rmtree(scratch)
            (directory / ".reservation.json").unlink(missing_ok=True)
            validate_operational_run_directory(directory)
            source.check_unchanged()
            _history(directory, "compute finalization completed; the complete standalone bank "
                     "is now eligible for explicit downstream analysis. No publication performed.")
            destination = directory.parent / run_id
            if destination.exists():
                raise PingstoreError("completed destination already exists")
            os.rename(directory, destination)
        except BaseException as exc:
            if directory.exists():
                attempt = record["bank_reuse"]["finalization_attempts"][-1]
                attempt["error"] = f"{type(exc).__name__}: {exc}"
                write_json_atomic(directory / "run.json", record)
                _history(directory, "finalization interrupted; retained as incomplete for explicit recovery.")
            raise
        return run_id


def handle_cli(argv: list[str], repo: Path) -> bool:
    flags = {"--reuse-plan", "--reuse-reserve", "--reuse-status", "--reuse-train-cell", "--reuse-finalize"}
    if not flags.intersection(argv):
        return False
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--reuse-plan", action="store_true", help="inspect the exact source and 84/18 assembly without writing")
    modes.add_argument("--reuse-reserve", action="store_true", help="reserve the incomplete compute writer before dispatch")
    modes.add_argument("--reuse-status", action="store_true")
    modes.add_argument("--reuse-train-cell", metavar="NAME")
    modes.add_argument("--reuse-finalize", action="store_true", help="assemble 102 cells and regenerate 34 diagnostics")
    parser.add_argument("--run-id")
    parser.add_argument("--execution-origin", default="slurm-wilkes")
    parser.add_argument("--recover-stale", action="store_true")
    parser.add_argument("--recover-finalization", action="store_true")
    args = parser.parse_args(argv)
    if args.reuse_plan and args.run_id:
        parser.error("plan does not accept an existing run ID")
    if not (args.reuse_plan or args.reuse_reserve) and not args.run_id:
        parser.error("--run-id is required")
    if args.recover_stale and not args.reuse_train_cell:
        parser.error("--recover-stale applies only to a replacement worker")
    if args.recover_finalization and not args.reuse_finalize:
        parser.error("--recover-finalization applies only to finalization")
    if args.reuse_plan:
        source, plan = inspect_source(repo)
        source.check_unchanged()
        print(json.dumps(plan, indent=2, sort_keys=True))
    elif args.reuse_reserve:
        print(reserve(repo, origin=args.execution_origin, run_id=args.run_id))
    elif args.reuse_status:
        print(json.dumps(status(repo, args.run_id), indent=2, sort_keys=True))
    elif args.reuse_train_cell:
        raise SystemExit(train_cell(repo, args.run_id, args.reuse_train_cell, recover_stale=args.recover_stale))
    else:
        print(finalize(repo, args.run_id, recover=args.recover_finalization))
    return True
