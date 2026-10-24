"""Run the independent compute stage for exp022.

Invoke this CLI from the repository root. The experiment-local implementation
owns argument validation, explicit source pins and Pingstore stage completion.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

import argparse
import hashlib
import json
import os
import platform
import random
import shlex
import shutil
import socket
import subprocess
import time
import uuid
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from experiments.exp022 import recipe
from experiments.exp022.custom import bank as bank_io
from experiments.exp022.custom.checkpoints import checkpoint_tensors, resolve_checkpoint
from experiments.exp022.custom.network_artifacts import (
    model_execution,
    record_snapshot_policy,
)
from pingstore.contracts import PingstoreError, file_sha256, write_json_atomic
from pingstore.stages import reserve_stage, source_run, stage_run
from snnlab.analysis import (
    active_fraction,
    firing_rate,
    population_spike_count_cv,
    rhythmicity_metrics,
)
from snnlab.lang import duration_steps
from snnlab.sim import (
    GraphExecutor,
    RecordingSpec,
    SignalRecording,
    encode_images_poisson,
    load_dataset,
    plan_graph,
    resolve_device,
)
from torch.utils.data import DataLoader, TensorDataset

SCHEMA = "pinglab.exp022.bank"
SCHEMA_VERSION = 2
REQUIRED_CELL_FILES = (
    "config.json",
    "metrics.json",
    "metrics.jsonl",
    "weights.pth",
    "weights_final.pth",
)


def python_executable() -> str:
    """Return a stable venv shim across aliases such as /tmp and /private/tmp."""
    executable = Path(sys.executable)
    canonical = executable.parent.resolve() / "python"
    return str(canonical if canonical.is_file() else executable)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_identity(repo: Path) -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return commit, dirty


def lock_identity(repo: Path) -> dict[str, Any]:
    path = repo / "uv.lock"
    return {
        "path": "uv.lock",
        "sha256": sha256_file(path) if path.exists() else None,
    }


def resolved_parameters(cell, settings, max_samples, epochs, scientific_contract=None):
    result = {
        "training_run_id": cell["training_run_id"],
        "family": cell["family"],
        "model_recipe": cell["model"],
        "seed": cell["seed"],
        "max_samples": max_samples,
        "epochs": epochs,
        "settings": settings,
    }
    if scientific_contract is not None:
        result["scientific_contract"] = scientific_contract
    return result


def create_manifest(
    *,
    repo: Path,
    bank_root: Path,
    bank_id: str,
    cells: list[dict[str, Any]],
    tier_for: Callable[[dict[str, Any]], str],
    samples_epochs: Callable[[dict[str, Any]], tuple[int, int]],
    build_settings: Callable[[dict[str, Any], int, int], dict],
    scientific_contract_for: Callable[[dict[str, Any], int, int], dict[str, Any]]
    | None = None,
    plumbing: bool = False,
    selection_tier: str = "all",
) -> dict[str, Any]:
    root = bank_root.resolve()
    if root == repo.resolve():
        raise ValueError("bank working root may not be the repository root")
    commit, dirty = git_identity(repo)
    if dirty:
        raise ValueError("refusing to create a bank manifest from a dirty worktree")

    native_identity = execution_identity()
    native_digest = execution_digest(native_identity)
    rows = []
    for cell in cells:
        max_samples, epochs = samples_epochs(cell)
        spec = (
            {k: v for k, v in cell.items() if k != "max_samples"} if plumbing else cell
        )
        out = root / "cells" / cell["name"]
        settings = build_settings(spec, max_samples, epochs)
        settings["execution_digest"] = native_digest
        command = [
            python_executable(),
            "-m",
            "experiments.exp022.custom.compute_parts.native_worker",
            "--native-cell",
            cell["name"],
            "--native-output",
            str(out),
            "--native-samples",
            str(max_samples),
            "--native-epochs",
            str(epochs),
        ]
        rows.append(
            {
                "name": cell["name"],
                "training_run_id": cell["training_run_id"],
                "family": cell["family"],
                "resource_tier": tier_for(cell),
                "parameters": resolved_parameters(
                    cell,
                    settings,
                    max_samples,
                    epochs,
                    scientific_contract=(
                        scientific_contract_for(cell, max_samples, epochs)
                        if scientific_contract_for
                        else None
                    ),
                ),
                "command": command,
                "command_shell": shlex.join(command),
                "output_directory": str(out),
                "required_outputs": list(REQUIRED_CELL_FILES),
            }
        )
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "bank_id": bank_id,
        "created_at_utc": utc_now(),
        "repository": {"commit": commit, "dirty": dirty},
        "environment": {
            "lockfile": lock_identity(repo),
            "python": platform.python_version(),
            "native_identity": native_identity,
        },
        "bank_root": str(root),
        "plumbing": plumbing,
        "selection": {"tier": selection_tier},
        "cells": rows,
    }


def manifest_hash(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    material = dict(payload)
    material["manifest_sha256"] = manifest_hash(payload)
    atomic_json(path, material)


def load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    supplied = payload.pop("manifest_sha256", None)
    actual = manifest_hash(payload)
    if supplied != actual:
        raise ValueError(
            f"bank manifest hash mismatch: expected {supplied}, got {actual}"
        )
    payload["manifest_sha256"] = supplied
    if (
        payload.get("schema") != SCHEMA
        or payload.get("schema_version") != SCHEMA_VERSION
    ):
        raise ValueError("unsupported exp022 bank manifest schema")
    return payload


def manifest_cell(manifest: dict[str, Any], name: str) -> dict[str, Any]:
    matches = [cell for cell in manifest["cells"] if cell["name"] == name]
    if len(matches) != 1:
        raise ValueError(f"manifest contains {len(matches)} cells named {name!r}")
    return matches[0]


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} is not a JSON object")
    return payload


def _same(actual: Any, expected: Any) -> bool:
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return abs(float(actual) - float(expected)) <= 1e-9
    if isinstance(actual, tuple):
        actual = list(actual)
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _same(observed, wanted)
            for observed, wanted in zip(actual, expected, strict=True)
        )
    return actual == expected


def _expected_config(cell):
    return dict(cell["parameters"]["settings"])


def validate_cell(
    cell: dict[str, Any], *, load_checkpoint: bool = True
) -> dict[str, Any]:
    directory = Path(cell["output_directory"])
    missing = [name for name in REQUIRED_CELL_FILES if not (directory / name).is_file()]
    if missing:
        state = "missing" if len(missing) == len(REQUIRED_CELL_FILES) else "partial"
        return {
            "valid": False,
            "state": state,
            "reasons": [f"missing {name}" for name in missing],
        }
    reasons: list[str] = []
    try:
        config = _json(directory / "config.json")
        metrics = _json(directory / "metrics.json")
        history = [
            json.loads(line)
            for line in (directory / "metrics.jsonl").read_text().splitlines()
            if line
        ]
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {"valid": False, "state": "invalid", "reasons": [str(exc)]}
    for payload, label in ((config, "config"), (metrics, "metrics")):
        if payload.get("training_cell_name") != cell["name"]:
            reasons.append(f"{label} cell name mismatch")
        if payload.get("training_run_id") != cell["training_run_id"]:
            reasons.append(f"{label} TR ID mismatch")
        if payload.get("bank_resolved_parameters") != cell["parameters"]:
            reasons.append(f"{label} resolved scientific parameters mismatch")
    nested = metrics.get("config", {})
    try:
        expected = _expected_config(cell)
    except (KeyError, TypeError, ValueError) as exc:
        return {
            "valid": False,
            "state": "invalid",
            "reasons": [f"unresolved manifest contract: {exc}"],
        }
    for key, wanted in expected.items():
        actual = config.get(key, nested.get(key))
        if not _same(actual, wanted):
            reasons.append(f"config {key} mismatch: {actual!r} != {wanted!r}")
    initialization = config.get("weight_initialization")
    metrics_initialization = nested.get("weight_initialization")
    required_roles = {"W_in", "W_out", "W_EE_1", "W_EI_1", "W_IE_1", "W_II_1"}
    if not isinstance(initialization, dict) or not required_roles <= set(
        initialization
    ):
        reasons.append("config missing complete weight initialization provenance")
    elif initialization != metrics_initialization:
        reasons.append("config/metrics weight initialization provenance mismatch")
    else:
        for role, record in initialization.items():
            if record.get("zeros_remain_trainable") is not True:
                reasons.append(
                    f"{role} does not declare trainable initialization zeros"
                )
            if record.get("distribution") not in {
                "lower_clamped_normal",
                "signed_normal",
                "kaiming_uniform_signed",
                "uniform",
                "constant",
                "zeros",
            }:
                reasons.append(f"{role} has unknown initialization distribution")
            if not isinstance(record.get("statistics"), dict):
                reasons.append(f"{role} missing initialization statistics")
        if not _same(
            initialization["W_in"].get("requested_initial_zero_fraction"),
            expected.get("w_in_initial_zero_fraction", 0.0),
        ):
            reasons.append("W_in initial-zero fraction mismatch")
    final_weights = metrics.get("weight_final")
    if not isinstance(final_weights, dict) or not required_roles <= set(final_weights):
        reasons.append("metrics missing final weight/regrowth provenance")
    epochs = int(cell["parameters"]["epochs"])
    samples = int(cell["parameters"]["max_samples"])
    if len(history) < epochs or int(history[-1].get("ep", -1)) < epochs:
        reasons.append(f"history did not reach epoch {epochs}")
    observed_samples = [
        row.get("samples") for row in history if row.get("samples") is not None
    ]
    expected_train_samples = round(samples * 0.9)  # fixed MNIST validation split
    if len(observed_samples) < epochs or any(
        int(value) != expected_train_samples for value in observed_samples[:epochs]
    ):
        reasons.append(
            f"history does not record {expected_train_samples} training samples "
            f"for each of {epochs} epochs"
        )
    if load_checkpoint:
        checkpoint_specs = {
            "best_validation": ("weights.pth", metrics.get("best_epoch")),
            "final_epoch": ("weights_final.pth", epochs),
        }
        recorded_checkpoints = metrics.get("checkpoints", {})
        for role, (filename, expected_epoch) in checkpoint_specs.items():
            record = recorded_checkpoints.get(role)
            if not isinstance(record, dict):
                reasons.append(f"missing {role} checkpoint metadata")
                continue
            if record.get("filename") != filename:
                reasons.append(f"{role} checkpoint filename mismatch")
            if record.get("epoch") != expected_epoch:
                reasons.append(f"{role} checkpoint epoch mismatch")
            path = directory / filename
            if record.get("sha256") != sha256_file(path):
                reasons.append(f"{role} checkpoint hash mismatch")
            try:
                from experiments.exp022.custom.checkpoints import checkpoint_tensors

                checkpoint_tensors(path, config)
            except Exception as exc:  # noqa: BLE001
                reasons.append(
                    f"{role} checkpoint load failed: {type(exc).__name__}: {exc}"
                )
    return {
        "valid": not reasons,
        "state": "complete" if not reasons else "invalid",
        "reasons": reasons,
    }


def preserve_partial(directory: Path) -> Path | None:
    if not directory.exists() or not any(directory.iterdir()):
        return None
    failed_root = directory.parents[1] / "failed" / directory.name
    destination = failed_root / utc_now().replace(":", "-")
    suffix = 0
    while destination.exists():
        suffix += 1
        destination = failed_root / f"{utc_now().replace(':', '-')}-{suffix}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    directory.replace(destination)
    return destination


def run_record_base(manifest: dict[str, Any], cell: dict[str, Any]) -> dict[str, Any]:
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES")
    return {
        "schema": "pinglab.exp022.cell-attempt",
        "bank_id": manifest["bank_id"],
        "bank_manifest_sha256": manifest["manifest_sha256"],
        "repository_commit": manifest["repository"]["commit"],
        "repository_dirty": manifest["repository"]["dirty"],
        "cell_name": cell["name"],
        "training_run_id": cell["training_run_id"],
        "resource_tier": cell["resource_tier"],
        "command": manifest.get("_runtime_commands", {}).get(
            cell["name"],
            cell.get("command"),
        ),
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "pid": os.getpid(),
        "gpu": {"cuda_visible_devices": gpu} if gpu is not None else {},
        "started_at_utc": utc_now(),
        "state": "running",
    }


def status_path(manifest: dict[str, Any], cell_name: str) -> Path:
    return Path(manifest["bank_root"]) / "status" / f"{cell_name}.json"


def lock_path(manifest: dict[str, Any], cell_name: str) -> Path:
    return Path(manifest["bank_root"]) / "status" / f"{cell_name}.lock"


def attempt_is_active(record: dict[str, Any]) -> bool | None:
    """Return True/False when activity is provable, otherwise None."""
    if record.get("state") != "running":
        return False
    job_id = record.get("slurm_job_id")
    if job_id:
        try:
            query = subprocess.run(
                [
                    "squeue",
                    "--noheader",
                    "--states=all",
                    "--jobs",
                    str(job_id),
                    "--format",
                    "%A",
                ],
                capture_output=True,
                text=True,
            )
        except FileNotFoundError:
            return None
        if query.returncode != 0:
            return None
        return str(job_id) in query.stdout.split()
    if record.get("hostname") != socket.gethostname():
        return None
    pid = record.get("pid")
    if not isinstance(pid, int):
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def acquire_attempt(
    manifest: dict[str, Any],
    cell: dict[str, Any],
    *,
    recover_stale: bool = False,
) -> tuple[dict[str, Any], Path]:
    """Atomically claim a cell, refusing active or unconfirmed stale owners."""
    lock = lock_path(manifest, cell["name"])
    lock.parent.mkdir(parents=True, exist_ok=True)
    record_file = status_path(manifest, cell["name"])
    for _ in range(2):
        attempt_id = str(uuid.uuid4())
        record = run_record_base(manifest, cell)
        record["attempt_id"] = attempt_id
        try:
            descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            prior = _json(record_file) if record_file.exists() else {}
            active = attempt_is_active(prior)
            if active is True:
                raise RuntimeError(f"cell {cell['name']} is owned by an active attempt")
            if not recover_stale:
                state = "stale" if active is False else "unconfirmed"
                raise RuntimeError(
                    f"cell {cell['name']} has a {state} attempt lock; "
                    "use --recover-stale only after confirming its job is inactive"
                )
            if active is not False:
                raise RuntimeError(
                    f"cannot confirm that the prior attempt for {cell['name']} is inactive"
                )
            recovery = lock.with_suffix(".recovery")
            try:
                recovery_fd = os.open(
                    recovery,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                )
            except FileExistsError as exc:
                raise RuntimeError(
                    f"stale recovery is already in progress for {cell['name']}"
                ) from exc
            os.close(recovery_fd)
            try:
                current = _json(record_file) if record_file.exists() else {}
                if attempt_is_active(current) is not False:
                    raise RuntimeError(
                        f"prior attempt for {cell['name']} changed during stale recovery"
                    )
                if current:
                    record["previous_attempt"] = current
                lock.unlink()
                descriptor = os.open(
                    lock,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                )
                with os.fdopen(descriptor, "w") as handle:
                    json.dump(
                        {"attempt_id": attempt_id, "cell_name": cell["name"]}, handle
                    )
                    handle.write("\n")
                atomic_json(record_file, record)
                return record, lock
            finally:
                recovery.unlink(missing_ok=True)
        with os.fdopen(descriptor, "w") as handle:
            json.dump({"attempt_id": attempt_id, "cell_name": cell["name"]}, handle)
            handle.write("\n")
        if record_file.exists():
            record["previous_attempt"] = _json(record_file)
        atomic_json(record_file, record)
        return record, lock
    raise RuntimeError(f"could not acquire attempt lock for {cell['name']}")


def release_attempt(lock: Path, attempt_id: str) -> None:
    try:
        owner = _json(lock)
    except (OSError, ValueError, json.JSONDecodeError):
        return
    if owner.get("attempt_id") == attempt_id:
        lock.unlink(missing_ok=True)


def summarize_status(
    manifest: dict[str, Any], *, load_checkpoint: bool = True
) -> dict[str, Any]:
    rows = []
    for cell in manifest["cells"]:
        result = validate_cell(cell, load_checkpoint=load_checkpoint)
        record_paths = (
            status_path(manifest, cell["name"]),
            Path(cell["output_directory"]) / "attempt.json",
        )
        record_path = next((path for path in record_paths if path.exists()), None)
        active = False
        stale = False
        owned = lock_path(manifest, cell["name"]).exists()
        if not result["valid"] and owned and record_path is None:
            result["state"] = "running"
            active = True
        elif not result["valid"] and record_path is not None:
            try:
                attempt = _json(record_path)
                if attempt.get("state") == "running":
                    activity = attempt_is_active(attempt)
                    if activity is True:
                        result["state"] = "running"
                        active = True
                    else:
                        result["state"] = "stale"
                        stale = True
                elif attempt.get("state") == "failed" and result["state"] != "invalid":
                    result["state"] = "failed"
            except Exception:  # noqa: BLE001
                pass
        rows.append(
            {
                "name": cell["name"],
                "training_run_id": cell["training_run_id"],
                "resource_tier": cell["resource_tier"],
                "active": active,
                "stale": stale,
                **result,
            }
        )
    counts: dict[str, int] = {}
    by_tier: dict[str, dict[str, int]] = {}
    by_tr: dict[str, dict[str, int]] = {}
    for row in rows:
        state = str(row["state"])
        counts[state] = counts.get(state, 0) + 1
        tier = str(row["resource_tier"])
        training_run_id = str(row["training_run_id"])
        for grouping, key in ((by_tier, tier), (by_tr, training_run_id)):
            bucket = grouping.setdefault(key, {})
            bucket[state] = bucket.get(state, 0) + 1
    return {
        "bank_id": manifest["bank_id"],
        "counts": counts,
        "by_tier": by_tier,
        "by_training_run_id": by_tr,
        "retry_cells": [
            row["name"]
            for row in rows
            if not row["valid"] and not row["active"] and not row["stale"]
        ],
        "recoverable_cells": [row["name"] for row in rows if row["stale"]],
        "cells": rows,
    }


def cell_dir(name: str) -> Path:
    """Shared per-cell artifact directory."""
    return bank_io.training_root() / name


def load_cell(name: str) -> Path:
    """Return a trained cell's directory, or fail loudly if this notebook has
    not been run. Analysis notebooks call this instead of training."""
    d = cell_dir(name)
    if not (d / "weights.pth").exists() or not (d / "weights_final.pth").exists():
        raise SystemExit(
            f"missing trained cell '{name}' at {bank_io._display_path(d)}; "
            "run exp022 (Training) first to produce the shared cells."
        )
    return d


def _train_one_cell(cell: dict, plumbing: bool) -> None:
    """Train one registered cell under the configured working root."""
    _writable_compute_root(bank_io.training_root())
    ms, ep = recipe.cell_samples_epochs(cell)  # honours PINGLAB_NB022_PLUMBING
    spec = cell
    if plumbing:
        # The plumbing profile explicitly caps every family, including canonical cells.
        spec = {k: v for k, v in cell.items() if k != "max_samples"}

    train_cell(spec, cell_dir(cell["name"]), ms, ep)
    _stamp_training_run_identity(cell)


def _stamp_training_run_identity(cell: dict) -> None:
    """Attach the public training-run ID to one completed cell's artifacts."""
    directory = cell_dir(cell["name"])
    _writable_compute_root(directory)
    for filename in ("config.json", "metrics.json"):
        path = directory / filename
        if not path.exists():
            raise RuntimeError(f"completed cell is missing {path}")
        payload = json.loads(path.read_text())
        payload["training_run_id"] = cell["training_run_id"]
        payload["training_cell_name"] = cell["name"]
        nested_config = payload.get("config")
        if isinstance(nested_config, dict):
            nested_config["training_run_id"] = cell["training_run_id"]
            nested_config["training_cell_name"] = cell["name"]
        path.write_text(json.dumps(payload, indent=2) + "\n")


def _cell_by_name(name: str) -> dict | None:
    return next((c for c in recipe.CANONICAL_CELLS if c["name"] == name), None)


def _bank_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--bank-create", type=Path, metavar="ROOT")
    action.add_argument("--bank-status", type=Path, metavar="MANIFEST")
    action.add_argument("--bank-list", type=Path, metavar="MANIFEST")
    action.add_argument("--bank-train-cell", metavar="NAME")
    action.add_argument("--bank-validate", type=Path, metavar="MANIFEST")
    action.add_argument("--bank-finalize", type=Path, metavar="MANIFEST")
    parser.add_argument("--bank", type=Path, metavar="MANIFEST")
    parser.add_argument(
        "--execution-origin",
        default="slurm-wilkes",
        choices=("local", "slurm-wilkes"),
    )
    parser.add_argument("--tier", default="all")
    parser.add_argument("--retry-only", action="store_true")
    parser.add_argument("--recover-stale", action="store_true")
    return parser


def _checked_bank_manifest(path: Path) -> dict:
    manifest_path = path.resolve()
    manifest = load_manifest(manifest_path)
    root = Path(manifest["bank_root"])
    if not root.is_absolute() or root.resolve() != root:
        raise SystemExit("bank working root must be an absolute resolved path")
    if manifest_path != root / "bank.json":
        raise SystemExit("bank manifest must be <working-root>/bank.json")
    commit, dirty = git_identity(REPO)
    if dirty:
        raise SystemExit("bank execution requires a clean source worktree")
    if manifest["repository"] != {"commit": commit, "dirty": False}:
        raise SystemExit(
            "bank manifest does not match the clean checked-out commit: "
            f"manifest={manifest['repository']['commit']} checkout={commit}"
        )
    if manifest.get("environment", {}).get("lockfile") != lock_identity(REPO):
        raise SystemExit("bank lockfile identity does not match the checkout")

    native_identity = execution_identity()
    if manifest["environment"].get("native_identity") != native_identity:
        raise SystemExit("training data, execution request or runtime source changed")
    tier = manifest.get("selection", {}).get("tier")
    try:
        if tier == "exp110-coba-damping-replacement":
            from experiments.exp022.custom.reuse_contract import replacement_cells

            selected_cells = replacement_cells()
        else:
            selected_cells = recipe.cells_in_resource_tier(tier)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    manifest_names_list = [row.get("name") for row in manifest.get("cells", [])]
    if len(manifest_names_list) != len(set(manifest_names_list)):
        raise SystemExit("bank manifest contains duplicate cell names")
    expected_names_list = [cell["name"] for cell in selected_cells]
    if manifest_names_list != expected_names_list:
        raise SystemExit("bank cell list does not exactly match its declared selection")
    previous_plumbing = os.environ.get("PINGLAB_NB022_PLUMBING")
    runtime_commands = {}
    try:
        if manifest.get("plumbing"):
            os.environ["PINGLAB_NB022_PLUMBING"] = "1"
        else:
            os.environ.pop("PINGLAB_NB022_PLUMBING", None)
        for row in manifest["cells"]:
            spec = _cell_by_name(row["name"])
            assert spec is not None
            samples, epochs = recipe.cell_samples_epochs(spec)
            command_spec = (
                {k: v for k, v in spec.items() if k != "max_samples"}
                if manifest.get("plumbing")
                else spec
            )
            settings = recipe.training_settings(command_spec, samples, epochs)
            settings["execution_digest"] = execution_digest(native_identity)
            resolved = resolved_parameters(
                spec,
                settings,
                samples,
                epochs,
                scientific_contract=recipe.scientific_contract(spec, samples, epochs),
            )
            command = [
                python_executable(),
                "-m",
                "experiments.exp022.custom.compute_parts.native_worker",
                "--native-cell",
                spec["name"],
                "--native-output",
                str(root / "cells" / spec["name"]),
                "--native-samples",
                str(samples),
                "--native-epochs",
                str(epochs),
            ]
            output_directory = (root / "cells" / spec["name"]).resolve()
            expected = {
                "name": spec["name"],
                "training_run_id": spec["training_run_id"],
                "family": spec["family"],
                "resource_tier": recipe.cell_resource_tier(spec),
                "parameters": resolved,
                "command": command,
                "command_shell": shlex.join(command),
                "output_directory": str(output_directory),
                "required_outputs": list(REQUIRED_CELL_FILES),
            }
            if row != expected:
                raise SystemExit(f"bank manifest registry drift for {row['name']}")
            if output_directory.parent != (root / "cells").resolve():
                raise SystemExit(
                    f"bank output path escapes the cells root: {row['name']}"
                )
            runtime_commands[row["name"]] = command
    finally:
        if previous_plumbing is None:
            os.environ.pop("PINGLAB_NB022_PLUMBING", None)
        else:
            os.environ["PINGLAB_NB022_PLUMBING"] = previous_plumbing
    manifest["_runtime_commands"] = runtime_commands
    return manifest


def _stamp_bank_identity(directory: Path, manifest: dict, row: dict) -> None:
    _writable_compute_root(directory)
    for filename in ("config.json", "metrics.json"):
        path = directory / filename
        payload = json.loads(path.read_text())
        payload.update(
            {
                "bank_id": manifest["bank_id"],
                "bank_manifest_sha256": manifest["manifest_sha256"],
                "resource_tier": row["resource_tier"],
                "bank_repository_commit": manifest["repository"]["commit"],
                "bank_resolved_parameters": row["parameters"],
            }
        )
        nested = payload.get("config")
        if isinstance(nested, dict):
            nested.update(
                {
                    "bank_id": manifest["bank_id"],
                    "bank_manifest_sha256": manifest["manifest_sha256"],
                    "resource_tier": row["resource_tier"],
                    "bank_repository_commit": manifest["repository"]["commit"],
                    "bank_resolved_parameters": row["parameters"],
                }
            )
        atomic_json(path, payload)


def _gpu_metadata() -> dict:
    try:
        query = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return {"available": False}
    if query.returncode != 0:
        return {"available": False}
    return {
        "available": True,
        "devices": [line.strip() for line in query.stdout.splitlines()],
    }


def _train_bank_cell(
    manifest_path: Path, name: str, *, recover_stale: bool = False
) -> int:
    manifest = _checked_bank_manifest(manifest_path)
    row = manifest_cell(manifest, name)
    directory = Path(row["output_directory"])
    _writable_compute_root(directory)
    existing = validate_cell(row)
    if existing["valid"]:
        print(f"[skip-valid] {name} is complete and will not be touched")
        return 0
    record, attempt_lock = acquire_attempt(
        manifest,
        row,
        recover_stale=recover_stale,
    )
    cell_status_path = status_path(manifest, name)
    exit_code = 1
    attempt_started = time.monotonic()
    try:
        existing = validate_cell(row)
        if existing["valid"]:
            record.update(
                {
                    "ended_at_utc": utc_now(),
                    "exit_code": 0,
                    "elapsed_seconds": round(time.monotonic() - attempt_started, 3),
                    "state": "complete",
                    "validation": existing,
                    "note": "became valid before training ownership was acquired",
                }
            )
            atomic_json(cell_status_path, record)
            print(f"[skip-valid] {name} became complete and will not be touched")
            return 0
        preserved = preserve_partial(directory)
        if preserved:
            print(f"[preserve-partial] {directory} -> {preserved}")
        record["gpu"] = _gpu_metadata()
        atomic_json(cell_status_path, record)
        directory.parent.mkdir(parents=True, exist_ok=True)

        spec = _cell_by_name(name)
        train_cell(
            spec,
            directory,
            row["parameters"]["max_samples"],
            row["parameters"]["epochs"],
        )
        exit_code = 0
        if exit_code == 0:
            spec = _cell_by_name(name)
            assert spec is not None
            old_root = bank_io.TRAINING_ROOT
            try:
                bank_io.TRAINING_ROOT = Path(manifest["bank_root"]) / "cells"
                _stamp_training_run_identity(spec)
            finally:
                bank_io.TRAINING_ROOT = old_root
            _stamp_bank_identity(directory, manifest, row)
        validation = validate_cell(row)
        try:
            metrics_payload = bank_io.load_metrics(directory)
        except (OSError, ValueError, json.JSONDecodeError):
            metrics_payload = {}
        record.update(
            {
                "ended_at_utc": utc_now(),
                "exit_code": exit_code,
                "elapsed_seconds": round(time.monotonic() - attempt_started, 3),
                "state": "complete"
                if exit_code == 0 and validation["valid"]
                else "failed",
                "validation": validation,
                "gpu_after": _gpu_metadata(),
                "training_performance": metrics_payload.get("perf"),
                "output_bytes": sum(
                    path.stat().st_size
                    for path in directory.rglob("*")
                    if path.is_file()
                ),
            }
        )
        directory.mkdir(parents=True, exist_ok=True)
        atomic_json(directory / "attempt.json", record)
        atomic_json(cell_status_path, record)
        return 0 if record["state"] == "complete" else 1
    except BaseException as exc:
        record.update(
            {
                "ended_at_utc": utc_now(),
                "exit_code": exit_code,
                "elapsed_seconds": round(time.monotonic() - attempt_started, 3),
                "state": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        directory.mkdir(parents=True, exist_ok=True)
        atomic_json(directory / "attempt.json", record)
        atomic_json(cell_status_path, record)
        raise
    finally:
        release_attempt(attempt_lock, record["attempt_id"])


def _handle_bank_cli(argv: list[str]) -> bool:
    actions = {
        "--bank-create",
        "--bank-status",
        "--bank-list",
        "--bank-train-cell",
        "--bank-validate",
        "--bank-finalize",
    }
    if not actions.intersection(argv):
        return False
    args = _bank_parser().parse_args(argv)
    if args.bank_create:
        root = args.bank_create.resolve()
        _writable_compute_root(root)
        if root.exists():
            raise SystemExit(
                f"bank destination already exists and will not be modified: {root}"
            )
        # Reject predictable setup failures before allocating a run identity.
        _commit, dirty = git_identity(REPO)
        if dirty:
            raise SystemExit("bank creation requires a clean source worktree")
        reserved = reserve_stage(
            REPO / ".pingstore",
            recipe.SLUG,
            "compute",
            origin=args.execution_origin,
        )
        manifest = create_manifest(
            repo=REPO,
            bank_root=root,
            bank_id=reserved,
            cells=recipe.CANONICAL_CELLS,
            tier_for=recipe.cell_resource_tier,
            samples_epochs=recipe.cell_samples_epochs,
            build_settings=recipe.training_settings,
            scientific_contract_for=recipe.scientific_contract,
            plumbing=False,
            selection_tier="all",
        )
        manifest["pingstore_run_id"] = reserved
        root.mkdir(parents=True)
        for child in ("cells", "logs", "status", "submissions"):
            (root / child).mkdir()
        write_manifest(root / "bank.json", manifest)
        print(root / "bank.json")
        return True
    manifest_path = (
        args.bank
        or args.bank_status
        or args.bank_list
        or args.bank_validate
        or args.bank_finalize
    )
    if manifest_path is None:
        raise SystemExit("--bank MANIFEST is required")
    manifest = _checked_bank_manifest(manifest_path)
    if args.bank_train_cell:
        raise SystemExit(
            _train_bank_cell(
                manifest_path,
                args.bank_train_cell,
                recover_stale=args.recover_stale,
            )
        )
    if args.bank_validate:
        print(f"valid bank {manifest['bank_id']} {manifest['manifest_sha256']}")
        return True
    # Login-node listing/status remains metadata-only. Finalization, which must
    # run in an allocation, additionally hashes and loads every checkpoint.
    status = summarize_status(manifest, load_checkpoint=bool(args.bank_finalize))
    if args.bank_finalize:
        _finalize_bank(manifest_path, manifest, status)
        return True
    if args.bank_list:
        cells = [
            cell
            for cell in manifest["cells"]
            if args.tier == "all" or cell["resource_tier"] == args.tier
        ]
        if args.retry_only:
            retry = set(status["retry_cells"])
            cells = [cell for cell in cells if cell["name"] in retry]
        print("\n".join(cell["name"] for cell in cells))
    else:
        print(json.dumps(status, indent=2, sort_keys=True))
    return True


def _writable_compute_root(directory: Path) -> None:
    resolved = directory.resolve()
    runs = (REPO / ".pingstore/runs").resolve()
    if resolved == runs or resolved == REPO.resolve():
        raise PingstoreError("compute output must be a dedicated working directory")
    if runs in resolved.parents:
        identity = resolved.relative_to(runs).parts[0]
        if not (identity.startswith(".") and identity.endswith(".tmp")):
            raise PingstoreError("compute cannot modify a completed Pingstore run")


def generate_snapshots(bank: Path, output: Path) -> None:
    """Retain fixed digit-0/sample-0 probes; no plotting or discarded recordings."""
    expected = {cell["name"] for cell in recipe.CANONICAL_CELLS}
    actual = {
        path.name
        for path in bank.iterdir()
        if path.is_dir() and (path / "config.json").is_file()
    }
    if actual != expected:
        raise PingstoreError(
            "diagnostics require the current 102-cell bank with the exact-refractory timestep grid; "
            "preserve historical diagnostics until the replacement bank is available"
        )
    for cell in recipe.CANONICAL_CELLS:
        if cell["seed"] != 42:
            continue
        trained = bank / cell["name"]
        checkpoint = resolve_checkpoint(trained, recipe.RESULT_CHECKPOINT_ROLE)
        destination = output / cell["name"]
        if destination.exists():
            raise PingstoreError(f"probe output already exists: {destination}")

        training = json.loads((trained / "config.json").read_text())
        request = recipe.diagnostic_request(training, cell)
        run_inference(
            trained,
            destination,
            output.parent / ".scratch/diagnostic-requests" / cell["name"],
            training,
            request,
            recipe.author_network,
        )
        write_json_atomic(
            destination / "probe-command.json",
            {
                "command": request,
                "checkpoint": {
                    key: value for key, value in checkpoint.items() if key != "path"
                },
            },
        )


def promote_cells(export: Path) -> None:
    """Promote a tool-native cells/ bank to canonical unit directories."""
    cells = export / "cells"
    for source in sorted(cells.iterdir()):
        target = export / source.name
        if target.exists():
            raise PingstoreError(f"cell export already exists: {target}")
        source.rename(target)
    cells.rmdir()


def copy_bank(bank: Path, destination: Path) -> list[dict]:
    """Copy scientific evidence without restamping configs or checkpoint roles."""
    from pingstore.contracts import file_sha256

    actual = {
        path.name
        for path in bank.iterdir()
        if path.is_dir() and (path / "config.json").is_file()
    }
    try:
        cells = recipe.cells_for_names(actual)
    except ValueError as exc:
        raise PingstoreError(str(exc)) from exc
    expected = {cell["name"] for cell in cells}
    inventory = []
    for name in sorted(expected):
        cell = bank / name
        for required in (
            "config.json",
            "metrics.json",
            "weights.pth",
            "weights_final.pth",
        ):
            if not (cell / required).is_file():
                raise PingstoreError(f"missing scientific payload: {cell / required}")
        for role in ("best_validation", "final_epoch"):
            resolve_checkpoint(cell, role)
        for path in sorted(cell.rglob("*")):
            if path.is_symlink() or not (path.is_file() or path.is_dir()):
                raise PingstoreError(f"unsupported bank entry: {path}")
            if not path.is_file():
                continue
            if path.name == "attempt.json":
                continue
            relative = path.relative_to(bank)
            digest = file_sha256(path)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            if file_sha256(target) != digest:
                raise PingstoreError(f"import checksum mismatch: {relative}")
            inventory.append(
                {
                    "path": relative.as_posix(),
                    "sha256": digest,
                    "size_bytes": target.stat().st_size,
                }
            )
    for unit in sorted(bank.iterdir()):
        if not unit.is_dir() or unit.name in expected:
            continue
        if not any(unit.name.startswith(name + "--training--") for name in expected):
            continue
        for path in sorted(unit.iterdir()):
            if not path.is_file() or path.is_symlink():
                raise PingstoreError(f"invalid native bank artifact: {path}")
            relative = path.relative_to(bank)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            inventory.append(
                {
                    "path": relative.as_posix(),
                    "sha256": file_sha256(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    return inventory


def _finalize_bank(manifest_path: Path, manifest: dict, status: dict) -> str:
    """Validate a complete parallel bank and atomically finish its compute run."""
    if len(manifest["cells"]) != len(recipe.CANONICAL_CELLS):
        raise SystemExit("bank finalization requires the complete 102-cell recipe")
    incomplete = [row["name"] for row in status["cells"] if not row["valid"]]
    if incomplete:
        raise SystemExit(
            f"bank finalization refused: {len(incomplete)} cells are invalid"
        )
    attempts = {}
    for row in manifest["cells"]:
        name = row["name"]
        if lock_path(manifest, name).exists():
            raise SystemExit(
                f"bank finalization refused: {name} still has an attempt lock"
            )
        record_path = status_path(manifest, name)
        if not record_path.is_file():
            raise SystemExit(f"bank finalization refused: {name} has no attempt record")
        attempt = json.loads(record_path.read_text())
        if (
            attempt.get("state") != "complete"
            or attempt.get("exit_code") != 0
            or attempt.get("cell_name") != name
            or attempt.get("bank_manifest_sha256") != manifest["manifest_sha256"]
        ):
            raise SystemExit(
                f"bank finalization refused: invalid attempt record for {name}"
            )
        attempts[name] = attempt
    reserved = manifest.get("pingstore_run_id")
    if not reserved:
        raise PingstoreError("bank manifest has no preallocated compute identity")
    bank = Path(manifest["bank_root"]) / "cells"
    with stage_run(
        REPO,
        recipe.SLUG,
        "compute",
        run_id=reserved,
        configuration=recipe.configuration(
            plumbing=os.environ.get("PINGLAB_NB022_PLUMBING") == "1"
        ),
        operation="parallel-bank",
    ) as run:
        inventory = copy_bank(bank, run.export / "cells")
        shutil.copy2(manifest_path, run.scratch / "bank.json")
        run.record["execution"]["bank"] = {
            "manifest_sha256": manifest["manifest_sha256"],
            "trained_cells": len(manifest["cells"]),
            "scientific_files": len(inventory),
            "attempts": attempts,
        }
        generate_snapshots(run.export / "cells", run.export / "snapshots")
        promote_cells(run.export)
        record_snapshot_policy(run)
        final = summarize_status(_checked_bank_manifest(manifest_path))
        if any(not row["valid"] for row in final["cells"]):
            raise PingstoreError("bank changed during compute finalization")
        (run.directory / "README.md").write_text(
            "# Exp022 compute — parallel model bank\n\n"
            f"Completed `{reserved}` from the validated exp022 bank manifest "
            f"`{manifest['manifest_sha256']}`. Cells were independently trained "
            "and validated before atomic finalization; diagnostic probes were then "
            "generated for the complete bank.\n"
        )
    return run.run_id


def import_bank(identity: str, *, run_id: str | None = None) -> str:
    """Copy an explicit v4 compute bank without training."""
    source = source_run(
        REPO / ".pingstore",
        identity,
        stage="compute",
        experiment=recipe.SLUG,
    )
    with stage_run(
        REPO,
        recipe.SLUG,
        "compute",
        inputs={"import": source},
        run_id=run_id,
        configuration=source.record["execution"].get("configuration"),
        operation="import",
    ) as run:
        inventory = copy_bank(source.export, run.export / "cells")
        promote_cells(run.export)
        record_snapshot_policy(run)
        native = source.record.get("scientific_artifacts", {}).get("native_paths", {})
        retained = {
            original: stored
            for original, stored in native.items()
            if (run.export / stored).is_file()
        }
        if retained:
            run.record["scientific_artifacts"] = {"native_paths": retained}
        run.record["execution"]["imported_files"] = len(inventory)
        run.record["execution"]["imported_bytes"] = sum(
            row["size_bytes"] for row in inventory
        )
        run.record["historical_evidence"] = {
            "source": source.reference,
            "note": "Historical cell attempts and inherited/repaired lineage are preserved; "
            "this execution copied evidence and did not train or simulate.",
        }
        (run.directory / "README.md").write_text(
            "# Exp022 compute — imported model bank\n\n"
            f"Imported byte-preserving scientific evidence from `{identity}`. "
            "The original remains unchanged.\n\n"
            "The 102 cells are direct unit directories under `export/`. "
            "Source identity is retained in `run.json`; this run's local operation is an import, "
            "not historical SLURM execution or retraining.\n\n"
            "Raw raster snapshots were not retained in this historical bank. "
            "Analysis can recover training curves from metrics; presentation must either "
            "explicitly carry historical rasters from the original run or use a new "
            "compute diagnostics run. No simulation is triggered by analyse/present.\n"
        )
    return run.run_id


def main() -> None:
    from experiments.exp022.custom import reuse

    if reuse.handle_cli(sys.argv[1:], REPO):
        return
    if _handle_bank_cli(sys.argv[1:]):
        return
    retired = {"--skip-training", "--plot-only", "--only-missing"} & set(sys.argv[1:])
    if retired:
        raise SystemExit(
            "combined lifecycle flags are retired: use analyse.py --source COMPUTE_RUN "
            "or present.py --source ANALYSIS_RUN"
        )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", help="identity already reserved before dispatch")
    parser.add_argument(
        "--import-source", help="copy an explicit v4 compute bank without computation"
    )
    parser.add_argument(
        "--source", help="compute bank for new retained diagnostic probes"
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="simulate fixed probes only; requires --source",
    )
    args = parser.parse_args()
    if args.import_source:
        if args.source or args.diagnostics:
            parser.error("--import-source cannot be combined with diagnostic execution")
        import_bank(args.import_source, run_id=args.run_id)
        return
    if bool(args.source) != args.diagnostics:
        parser.error("--source and --diagnostics must be used together")
    inputs = {}
    if args.source:
        inputs["bank"] = source_run(
            REPO / ".pingstore", args.source, stage="compute", experiment=recipe.SLUG
        )
        if not (inputs["bank"].export / recipe.CANONICAL_CELLS[0]["name"]).is_dir():
            parser.error("--source must be a compute run exporting a model bank")
    with stage_run(
        REPO,
        recipe.SLUG,
        "compute",
        inputs=inputs,
        run_id=args.run_id,
        configuration=recipe.configuration(
            plumbing=os.environ.get("PINGLAB_NB022_PLUMBING") == "1"
        ),
    ) as run:
        if inputs:
            bank = inputs["bank"].export
        else:
            bank = run.export / "cells"
            previous = bank_io.TRAINING_ROOT
            try:
                bank_io.TRAINING_ROOT = bank
                for cell in recipe.CANONICAL_CELLS:
                    _train_one_cell(
                        cell, os.environ.get("PINGLAB_NB022_PLUMBING") == "1"
                    )
            finally:
                bank_io.TRAINING_ROOT = previous
        generate_snapshots(bank, run.export / "snapshots")
        if not inputs:
            promote_cells(run.export)
        record_snapshot_policy(run)


PARAMETERS = {
    "input_to_E.weight": "W_ff.0",
    "readout_projection.weight": "W_ff.1",
    "E_to_I.weight": "W_ei.1",
    "I_to_E.weight": "W_ie.1",
}


def bind_model(bundle, training, state, *, scale=1.0, fresh_loop=None, device=None):
    model = GraphExecutor(
        plan_graph(bundle.graph),
        seed=training["seed"],
        surrogate_slope=training["surrogate_slope"],
    )
    parameters = model.parameter_map()
    if set(parameters) != set(PARAMETERS):
        raise PingstoreError("graph parameter roles differ from checkpoint contract")
    with torch.no_grad():
        for name, key in PARAMETERS.items():
            value = state[key].clamp(min=0) if key.startswith("W_ff") else state[key]
            if fresh_loop is not None and name in ("E_to_I.weight", "I_to_E.weight"):
                value = fresh_loop[name]
            if name == "input_to_E.weight":
                value = value * scale
            if parameters[name].shape != value.shape:
                raise PingstoreError(f"graph/checkpoint shape mismatch: {name}")
            parameters[name].copy_(value)
    return model.to(device or resolve_device("auto")).eval()


def run_inference(
    train, output, attachments, training, request, author, *, forward=None
):
    """Each batch begins with fresh state; input and checkpoint order are explicit."""
    checkpoint = resolve_checkpoint(train, request["checkpoint_role"])
    state = checkpoint_tensors(checkpoint["path"], training)
    device = torch.device(
        resolve_device(request.get("device", os.environ.get("PINGLAB_DEVICE", "auto")))
    )
    cursor, fresh = initial_draws(training, ei_strength=request.get("ei_strength"))
    bundle = author(training, observables=request["observables"])
    model = bind_model(
        bundle,
        training,
        state,
        scale=request.get("scale", 1),
        fresh_loop=fresh if "ei_strength" in request else None,
        device=device,
    )
    output.mkdir(parents=True, exist_ok=False)
    attachments.mkdir(parents=True, exist_ok=True)
    with model_execution(
        model,
        bundle,
        output / "inference",
        kind="infer",
        seed=training["seed"],
        settings=request,
    ):
        steps = duration_steps(request["t_ms"], training["dt"])
        snapshot = request["input"] == "snapshot"
        synthetic = request["input"] == "synthetic"
        if synthetic:
            pixels, labels = None, None
            selected = np.arange(request["trials"])
            generator = torch.Generator().manual_seed(training["seed"] + 1)
        else:
            _, pixels, _, labels = load_dataset(
                "mnist", split=True, evaluation_only=True, evaluation_split="test"
            )
            if (
                pixels.shape != (10000, training["n_in"])
                or labels.shape != (10000,)
                or not np.isfinite(pixels).all()
                or (pixels < 0).any()
                or (pixels > 1).any()
            ):
                raise PingstoreError("invalid official MNIST test partition")
            if snapshot:
                index = request.get("sample_index")
                if index is None:
                    indices = np.flatnonzero(labels == request["digit"])
                    index = int(indices[request["sample"]])
                if not 0 <= index < len(labels):
                    raise PingstoreError("snapshot index out of range")
                selected = np.array([index])
                # Historical snapshots encode on the device stream after CPU
                # initialization; non-CPU initialization consumes no device draws.
                generator = (
                    cursor
                    if device.type == "cpu"
                    else torch.Generator(device=device).manual_seed(training["seed"])
                )
            else:
                samples = request["samples"]
                selected = (
                    np.random.RandomState(request["subset_seed"]).choice(
                        len(labels), samples, replace=False
                    )
                    if samples < len(labels)
                    else np.arange(len(labels))
                )
                generator = torch.Generator().manual_seed(request["encoder_seed"])
        recording = (
            RecordingSpec(
                tuple(
                    SignalRecording(f"{label}.spikes", kind="spike_events")
                    for label in ("E", "I")
                )
            )
            if any(k in request["products"] for k in ("rasters", "population"))
            else None
        )
        counts = {"e": [], "i": []}
        events = {"e": [], "i": []}
        correct = 0
        ce_sum = 0.0
        perturb = {}
        with torch.inference_mode():
            for start in range(0, len(selected), request["batch_size"]):
                indices = selected[start : start + request["batch_size"]]
                if synthetic:
                    probability = request["input_rate_hz"] * training["dt"] / 1000
                    drive = (
                        (
                            torch.rand(
                                steps,
                                len(indices),
                                training["n_in"],
                                generator=generator,
                            )
                            < probability
                        )
                        .float()
                        .to(device)
                    )
                else:
                    images = (
                        torch.from_numpy(pixels[indices]).to(device)
                        if snapshot
                        else torch.from_numpy(pixels[indices])
                    )
                    drive = encode_images_poisson(
                        images,
                        steps,
                        training["dt"],
                        request["input_rate_hz"],
                        generator=generator,
                    ).to(device)
                if forward:
                    result, accounting = forward(model, drive, request)
                    for key, row in accounting.items():
                        saved = perturb.setdefault(key, {field: 0 for field in row})
                        for field, value in row.items():
                            saved[field] += value
                else:
                    result = model(
                        {"drive": drive},
                        diagnostics=snapshot,
                        recording=recording,
                        batch_offset=start,
                    )
                for label in ("e", "i"):
                    value = result.outputs[f"spk_{label}_count"]
                    population = training["n_hidden" if label == "e" else "n_inh"]
                    if (
                        value.shape != (len(indices), population)
                        or not torch.isfinite(value).all()
                        or (value < 0).any()
                        or (value > steps).any()
                        or not torch.equal(value, value.floor())
                    ):
                        raise PingstoreError("invalid population count reduction")
                    counts[label].append(
                        result.outputs[f"spk_{label}_count"].detach().cpu().numpy()
                    )
                    if recording:
                        events[label].append(
                            result.recorded_signals[f"{label.upper()}.spikes"]
                            .cpu()
                            .numpy()
                        )
                if snapshot:
                    raw = {
                        k: v.detach().cpu().numpy()[:, 0]
                        for k, v in result.diagnostics.items()
                    }
                    np.savez_compressed(
                        output / "recording.npz",
                        dt=np.float32(training["dt"]),
                        n_e=np.int32(training["n_hidden"]),
                        n_i=np.int32(training["n_inh"]),
                        label=np.int64(labels[indices[0]]),
                        **raw,
                    )
                elif not synthetic:
                    scores = result.outputs["class_scores"]
                    if (
                        scores.shape != (len(indices), training["n_out"])
                        or not torch.isfinite(scores).all()
                    ):
                        raise PingstoreError("invalid class score reduction")
                    targets = torch.from_numpy(labels[indices]).to(device)
                    correct += int((scores.argmax(1) == targets).sum())
                    ce_sum += float(F.cross_entropy(scores, targets, reduction="sum"))
        count = {k: np.concatenate(v) for k, v in counts.items()}
        seconds = steps * training["dt"] / 1000
        rates = {
            key: float(count[label].sum() / (len(selected) * training[size] * seconds))
            for label, key, size in (("e", "hid", "n_hidden"), ("i", "inh", "n_inh"))
        }
        config = {
            k: training[k]
            for k in (
                "dt",
                "t_ms",
                "seed",
                "tau_gaba_ms",
                "n_in",
                "n_hidden",
                "n_inh",
                "ei_strength",
                "ei_ratio",
            )
        }
        config.update(
            t_ms=request["t_ms"],
            load_weights=str(checkpoint["path"]),
            ei_strength=request.get("ei_strength", training["ei_strength"]),
        )
        if synthetic:
            config.update(input_rate_hz=request["input_rate_hz"], n_batch=len(selected))
        else:
            config.update(
                dataset="mnist",
                evaluation_partition="official_mnist_test",
                evaluation_samples=len(selected),
            )
        metrics = {
            "config": config,
            "rates_hz": rates,
            "rate_e_hz": rates["hid"],
            "rate_i_hz": rates["inh"],
        }
        if not synthetic:
            metrics.update(
                best_acc=100 * correct / len(selected),
                n_correct=correct,
                n_total=len(selected),
                ce_loss=ce_sum / len(selected),
            )
        if perturb:
            metrics["perturbation"] = {
                "mode": request["perturb_mode"],
                "level": request["perturb_level"],
                "dt_ms": training["dt"],
                "populations": perturb,
            }
        if not snapshot:
            write_json_atomic(output / "metrics.json", metrics)
        if recording:
            raster = {
                "dt": np.float32(training["dt"]),
                "T": np.int32(steps),
                "n_trials": np.int32(len(selected)),
                "n_e": np.int32(training["n_hidden"]),
                "n_i": np.int32(training["n_inh"]),
            }
            for label in ("e", "i"):
                coordinates = np.concatenate(events[label])
                for suffix, axis in (("t", 0), ("trial", 1), ("cell", 2)):
                    raster[f"{label}_{suffix}"] = coordinates[:, axis].astype(np.int32)
            if "rasters" in request["products"]:
                np.savez_compressed(output / "rasters.npz", **raster)
            if "population" in request["products"]:
                linear = raster["e_trial"].astype(np.int64) * steps + raster["e_t"]
                pop = (
                    np.bincount(linear, minlength=len(selected) * steps)
                    .reshape(len(selected), steps)
                    .astype(np.float32)
                    / training["n_hidden"]
                )
                np.savez_compressed(
                    output / "pop_traces.npz", pop_e=pop, dt=np.float32(training["dt"])
                )
        if "rates" in request["products"]:
            np.savez_compressed(
                output / "per_cell_rates.npz",
                rate_e_per_sample=count["e"].mean(1) / seconds,
                rate_e_per_cell=count["e"].mean(0) / seconds,
            )
        write_json_atomic(
            attachments / "request.json",
            {
                "request": request,
                "training": training,
                "checkpoint": {k: v for k, v in checkpoint.items() if k != "path"},
                "graph_digest": bundle.manifest["graph_digest"],
                "selected_indices": selected.tolist(),
                "input_content_sha256": None
                if synthetic
                else hashlib.sha256(pixels.tobytes() + labels.tobytes()).hexdigest(),
                "device": str(device),
            },
        )
        if file_sha256(checkpoint["path"]) != checkpoint["sha256"]:
            raise PingstoreError("checkpoint changed during inference")
        return metrics


PARAMETERS = {
    "input_to_E.weight": "W_ff.0",
    "readout_projection.weight": "W_ff.1",
    "E_to_I.weight": "W_ei.1",
    "I_to_E.weight": "W_ie.1",
}


def initial_draws(training, *, ei_strength=None):
    """Ordered CPU weight draws, including unused zero recurrence.

    Return the generator cursor needed for CPU snapshots and the fresh loop
    needed for transfer probes. No model is rebuilt to recover these draws.
    """
    generator = torch.Generator().manual_seed(training["seed"])
    n_in, n_e, n_i, n_out = (
        training[k] for k in ("n_in", "n_hidden", "n_inh", "n_out")
    )

    def normal(shape, mean, sd, zero=0, scale=True):
        value = (
            torch.randn(*shape, generator=generator).mul_(sd).add_(mean).clamp_(min=0)
        )
        if zero:
            value *= (torch.rand(*shape, generator=generator) > zero).float()
            value /= 1 - zero
        return value / shape[0] if scale else value

    mean, sd = training["w_in"]
    external = normal((n_in, n_e), mean, sd, training["w_in_initial_zero_fraction"])
    if training.get("readout_w_init_mean") is not None:
        readout = normal(
            (n_e, n_out),
            training["readout_w_init_mean"],
            training["readout_w_init_std"],
            scale=False,
        )
    else:
        readout = normal((n_e, n_out), 5.1, 3.8) * training.get(
            "readout_w_out_scale", 225.0
        )
    normal((n_e, n_e), 0, 0)
    strength = training["ei_strength"] if ei_strength is None else ei_strength
    ei = normal((n_e, n_i), strength, strength * 0.1)
    ie_strength = strength * training["ei_ratio"]
    ie = normal((n_i, n_e), ie_strength, ie_strength * 0.1)
    normal((n_i, n_i), 0, 0)
    return generator, {
        "input_to_E.weight": external,
        "readout_projection.weight": readout,
        "E_to_I.weight": ei,
        "I_to_E.weight": ie,
    }


def weight_statistics(value):
    array = value.detach().cpu().numpy()
    return {
        "shape": list(array.shape),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
        "zero_count": int((array == 0).sum()),
        "zero_fraction": float((array == 0).mean()),
    }


def export_tensors(model, cfg):
    parameters = model.parameter_map()
    return {
        **{
            key: parameters[name].detach().cpu().clone()
            for name, key in PARAMETERS.items()
        },
        "W_ee.1": torch.zeros(cfg["n_hidden"], cfg["n_hidden"]),
        "W_ii.1": torch.zeros(cfg["n_inh"], cfg["n_inh"]),
    }


def reference_measurements(model, spikes, cfg):
    with torch.no_grad():
        result = model({"drive": spikes}, diagnostics=True)
    e = result.diagnostics["spk_e"].cpu().numpy()[:, 0]
    i = result.diagnostics["spk_i"].cpu().numpy()[:, 0]
    rhythm = rhythmicity_metrics(e, cfg["dt"])
    return {
        "rate_e": firing_rate(e, cfg["dt"]),
        "rate_i": firing_rate(i, cfg["dt"]),
        "cv": population_spike_count_cv(e, cfg["dt"]),
        "act": active_fraction(e),
        "contrast": rhythm["contrast"] if rhythm["contrast"] is not None else 0.0,
        "f0_hz": None,
        "lobe_lag_ms": rhythm["lobe_lag"],
        "trough_lag_ms": rhythm["trough_lag"],
        "iei_mode_lag_ms": rhythm["iei_mode_lag"],
        "lobe_to_trough": rhythm["lobe_to_trough"],
    }


def train_cell(cell, directory, samples, epochs):
    """Reset every presentation; select minimum draw-mean CE, then accuracy.

    Training retains the original device encoder stream and shuffled DataLoader;
    validation restarts its three CPU encoder/rate streams every epoch.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    cfg = recipe.training_settings(cell, samples, epochs)
    cfg["execution_digest"] = execution_digest(execution_identity())
    device = torch.device(resolve_device(os.environ.get("PINGLAB_DEVICE", "auto")))
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    x_train, x_validation, y_train, y_validation = load_dataset(
        "mnist", max_samples=samples, split=True
    )
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=cfg["batch_size"],
        shuffle=True,
    )
    validation_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_validation), torch.from_numpy(y_validation)),
        batch_size=cfg["batch_size"],
    )
    bundle = recipe.author_network(cfg, observables=("spikes",))
    trainable = ["input_to_E.weight", "readout_projection.weight"]
    if cfg["trainable_w_ei"]:
        trainable.append("E_to_I.weight")
    if cfg["trainable_w_ie"]:
        trainable.append("I_to_E.weight")
    model = GraphExecutor(
        plan_graph(bundle.graph),
        seed=cfg["seed"],
        trainable_parameters=trainable,
        surrogate_slope=cfg["surrogate_slope"],
    )
    _, initial = initial_draws(cfg)
    with torch.no_grad():
        for name, value in initial.items():
            model.parameter_map()[name].copy_(value)
    model.to(device)
    zero_ee = torch.zeros(cfg["n_hidden"], cfg["n_hidden"])
    zero_ii = torch.zeros(cfg["n_inh"], cfg["n_inh"])
    roles = {
        "W_in": initial["input_to_E.weight"],
        "W_out": initial["readout_projection.weight"],
        "W_EI_1": initial["E_to_I.weight"],
        "W_IE_1": initial["I_to_E.weight"],
        "W_EE_1": zero_ee,
        "W_II_1": zero_ii,
    }
    cfg["weight_initialization"] = {
        role: {
            "distribution": "lower_clamped_normal",
            "zeros_remain_trainable": True,
            "requested_initial_zero_fraction": cfg["w_in_initial_zero_fraction"]
            if role == "W_in"
            else 0.0,
            "statistics": weight_statistics(value),
        }
        for role, value in roles.items()
    }
    cfg["dataset_split"] = {
        "source_train_partition": "official_mnist_train",
        "source_test_partition": "official_mnist_test",
        "validation_fraction": 0.1,
        "split_seed": 42,
        "optimizer_train_samples": len(y_train),
        "validation_samples": len(y_validation),
        "official_test_samples": 10000,
        "checkpoint_selection_partition": "validation",
        "official_test_used_during_training": False,
    }
    cfg["validation_encoder_draws"] = {
        "count": 3,
        "encoder_seeds": list(recipe.VALIDATION_ENCODER_SEEDS),
        "input_rate_seeds": list(recipe.VALIDATION_RATE_SEEDS),
        "aggregation_unit": "validation_sample_then_encoder_draw",
        "checkpoint_selection": "minimum_mean_cross_entropy; tie maximum_mean_accuracy; tie earliest_epoch",
    }
    cfg.update(
        training_run_id=cell["training_run_id"],
        training_cell_name=cell["name"],
        mode="train",
        executor="snnlab.sim.GraphExecutor",
        graph_digest=bundle.manifest["graph_digest"],
        device=str(device),
    )
    write_json_atomic(directory / "config.json", cfg)
    with model_execution(
        model,
        bundle,
        directory / "training",
        kind="train",
        seed=cfg["seed"],
        settings=cfg,
        snapshot_stages=("end",),
    ):
        steps = duration_steps(cfg["t_ms"], cfg["dt"])
        seconds = steps * cfg["dt"] / 1000
        _, test_pixels, _, test_labels = load_dataset(
            "mnist", split=True, evaluation_only=True, evaluation_split="test"
        )
        index = int(np.flatnonzero(test_labels == 0)[0])
        torch.manual_seed(recipe.REFERENCE_ENCODER_SEED)
        reference = encode_images_poisson(
            torch.from_numpy(test_pixels[index : index + 1]).to(device),
            steps,
            cfg["dt"],
            cfg["input_rate"],
        )
        initial_metrics = reference_measurements(model, reference, cfg)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
        )
        rates = (
            torch.tensor(cfg["input_rates"], dtype=torch.float32)
            if cfg["input_rates"]
            else None
        )
        rate_generator = torch.Generator().manual_seed(
            cfg["seed"] + recipe.TRAINING_RATE_SEED_OFFSET
        )

        def sample_rates(batch_size, generator):
            return (
                cfg["input_rate"]
                if rates is None
                else rates[
                    torch.randint(len(rates), (batch_size,), generator=generator)
                ].to(device)
            )

        history = []
        best_loss = float("inf")
        best_accuracy = 0.0
        best_epoch = 0
        best_state = None
        started = perf_counter()
        for epoch in range(1, epochs + 1):
            trained = train_epoch(
                model,
                train_loader,
                optimizer,
                cfg,
                device,
                steps,
                seconds,
                sample_rates,
                rate_generator,
            )
            validated = validate_epoch(
                model, validation_loader, cfg, device, steps, seconds, sample_rates
            )
            validation_loss, accuracy = validated["test_loss"], validated["acc"]
            new_best = validation_loss < best_loss or (
                validation_loss == best_loss and accuracy > best_accuracy
            )
            if new_best:
                best_loss = validation_loss
                best_accuracy = accuracy
                best_epoch = epoch
                best_state = export_tensors(model, cfg)
            record = {
                "ep": epoch,
                "lr": cfg["lr"],
                "new_best": new_best,
                **trained,
                **validated,
                "weight_norms": {
                    PARAMETERS[name]: float(value.detach().norm())
                    for name, value in model.parameter_map().items()
                    if value.requires_grad
                },
                **reference_measurements(model, reference, cfg),
            }
            history.append(record)
            with (directory / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(record, allow_nan=False) + "\n")
            print(
                f"{cell['name']} epoch {epoch}/{epochs}: CE={validation_loss:g}, accuracy={accuracy:g}",
                flush=True,
            )
        if best_state is None:
            raise ValueError("training never produced a selectable checkpoint")
        final = export_tensors(model, cfg)
        torch.save(best_state, directory / "weights.pth")
        torch.save(final, directory / "weights_final.pth")
        final_roles = {
            role: final[key]
            for role, key in {
                "W_in": "W_ff.0",
                "W_out": "W_ff.1",
                "W_EE_1": "W_ee.1",
                "W_EI_1": "W_ei.1",
                "W_IE_1": "W_ie.1",
                "W_II_1": "W_ii.1",
            }.items()
        }
        write_json_atomic(
            directory / "metrics.json",
            {
                "config": cfg,
                "epochs": history,
                "init": initial_metrics,
                "end": reference_measurements(model, reference, cfg),
                "best_acc": best_accuracy,
                "best_epoch": best_epoch,
                "best_validation_loss": best_loss,
                "total_elapsed_s": perf_counter() - started,
                "weight_final": {
                    role: weight_statistics(value)
                    for role, value in final_roles.items()
                },
                "training_run_id": cell["training_run_id"],
                "training_cell_name": cell["name"],
                "checkpoints": {
                    role: {
                        "filename": filename,
                        "epoch": selected_epoch,
                        "sha256": file_sha256(directory / filename),
                    }
                    for role, filename, selected_epoch in (
                        ("best_validation", "weights.pth", best_epoch),
                        ("final_epoch", "weights_final.pth", epochs),
                    )
                },
            },
        )

    # Final capture above precedes loading the selected checkpoint's real tensors.
    with torch.no_grad():
        for name, value in model.parameter_map().items():
            value.copy_(best_state[PARAMETERS[name]].to(device))
    with model_execution(
        model,
        bundle,
        directory / "best-validation",
        kind="train",
        seed=cfg["seed"],
        settings={**cfg, "selected_epoch": best_epoch},
        snapshot_stages=("end",),
    ):
        pass


def execution_identity():
    """Read-only source, backend and raw dataset identity for training retries."""
    import os
    from importlib.metadata import version

    import snnlab

    source = Path(snnlab.__file__).parent
    raw = Path("/tmp/mnist/MNIST/raw")
    names = (
        "train-images-idx3-ubyte",
        "train-labels-idx1-ubyte",
        "t10k-images-idx3-ubyte",
        "t10k-labels-idx1-ubyte",
    )
    if any(not (raw / name).is_file() for name in names):
        raise ValueError(
            "prepopulate the explicit MNIST cache before freezing a training bank"
        )
    return {
        "dataset_files": {name: file_sha256(raw / name) for name in names},
        "versions": {
            name: version(name)
            for name in ("snnlab", "torch", "numpy", "torchvision", "scikit-learn")
        },
        "source": {
            str(p.relative_to(source)): file_sha256(p)
            for p in sorted(source.rglob("*"))
            if p.is_file() and p.suffix in (".py", ".json")
        },
        "helper_source": {
            module.__name__: file_sha256(Path(module.__file__)) for module in (recipe,)
        },
        "training_source": file_sha256(Path(__file__)),
        "runtime": {
            "requested_device": os.environ.get("PINGLAB_DEVICE", "auto"),
            "cuda": torch.version.cuda,
            "threads": torch.get_num_threads(),
            "default_dtype": str(torch.get_default_dtype()),
            "deterministic": torch.are_deterministic_algorithms_enabled(),
            "matmul_precision": torch.get_float32_matmul_precision(),
            "tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        },
    }


def execution_digest(identity):
    import hashlib

    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def train_epoch(
    model, loader, optimizer, cfg, device, steps, seconds, sample_rates, rate_generator
):
    model.train()
    loss_sum = 0.0
    batch_count = 0
    sample_count = 0
    grad_sum = 0.0
    grad_max = 0.0
    skipped = 0
    nan_batches = 0
    for pixels, labels in loader:
        pixels, labels = pixels.to(device), labels.to(device)
        drive = encode_images_poisson(
            pixels, steps, cfg["dt"], sample_rates(len(labels), rate_generator)
        )
        result = model({"drive": drive}, diagnostics=False)
        scores = result.outputs["class_scores"]
        if torch.isnan(scores).any():
            optimizer.zero_grad()
            nan_batches += 1
            continue
        loss = F.cross_entropy(scores, labels)
        if cfg["fr_reg_upper_strength"]:
            sample_rate = result.outputs["spk_e_count"].mean(1) / seconds
            loss = (
                loss
                + cfg["fr_reg_upper_strength"]
                * torch.relu(sample_rate - cfg["fr_reg_upper_target_hz"])
                .square()
                .mean()
            )
        optimizer.zero_grad()
        loss.backward()
        norm = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        )
        if np.isfinite(norm):
            optimizer.step()
            model.enforce_constraints()
            loss_sum += float(loss.detach())
            batch_count += 1
            grad_sum += norm
            grad_max = max(grad_max, norm)
        else:
            optimizer.zero_grad(set_to_none=True)
            skipped += 1
        sample_count += len(labels)
    return {
        "loss": loss_sum / max(batch_count, 1),
        "samples": sample_count,
        "grad_norm": grad_sum / max(batch_count, 1),
        "grad_norm_max": grad_max,
        "skipped_steps": skipped,
        "nan_forward_batches": nan_batches,
    }


def validate_epoch(model, loader, cfg, device, steps, seconds, sample_rates):
    model.eval()
    correct = 0
    total = 0
    ce_sum = 0.0
    e_sum = 0.0
    i_sum = 0.0
    margin = 0.0
    confidence = 0.0
    scale = 0.0
    draws = []
    output_spikes = 0.0
    output_silent = 0
    output_samples = 0
    class_spikes = torch.zeros(cfg["n_out"], dtype=torch.float64)
    by_rate = {}
    with torch.no_grad():
        for draw_index, (encoder_seed, rate_seed) in enumerate(
            zip(
                recipe.VALIDATION_ENCODER_SEEDS,
                recipe.VALIDATION_RATE_SEEDS,
                strict=True,
            )
        ):
            encoder = torch.Generator().manual_seed(encoder_seed)
            rate_rng = torch.Generator().manual_seed(rate_seed)
            draw_correct = 0
            draw_total = 0
            draw_loss = 0.0
            for pixels, labels in loader:
                pixels, labels = pixels.to(device), labels.to(device)
                sampled_rates = sample_rates(len(labels), rate_rng)
                drive = encode_images_poisson(
                    pixels, steps, cfg["dt"], sampled_rates, generator=encoder
                )
                result = model({"drive": drive}, diagnostics=False)
                scores = result.outputs["class_scores"]
                batch = len(labels)
                loss = float(F.cross_entropy(scores, labels)) * batch
                n_correct = int((scores.argmax(1) == labels).sum())
                ce_sum += loss
                draw_loss += loss
                correct += n_correct
                draw_correct += n_correct
                total += batch
                draw_total += batch
                e_sum += (
                    float(result.outputs["spk_e_count"].sum())
                    / cfg["n_hidden"]
                    / seconds
                )
                i_sum += (
                    float(result.outputs["spk_i_count"].sum()) / cfg["n_inh"] / seconds
                )
                if cfg["readout_mode"] == "spike-count":
                    out = scores.detach().cpu()
                    per_sample = out.sum(1)
                    output_spikes += float(per_sample.sum())
                    output_silent += int((per_sample == 0).sum())
                    output_samples += batch
                    class_spikes += out.sum(0).double()
                    realized = torch.as_tensor(sampled_rates).detach().cpu()
                    if realized.ndim == 0:
                        realized = realized.expand(batch)
                    for rate in torch.unique(realized).tolist():
                        mask = realized == rate
                        bucket = by_rate.setdefault(
                            float(rate),
                            {
                                "n_samples": 0,
                                "n_correct": 0,
                                "n_silent": 0,
                                "n_spikes": 0.0,
                            },
                        )
                        bucket["n_samples"] += int(mask.sum())
                        bucket["n_correct"] += int(
                            (out.argmax(1)[mask] == labels.detach().cpu()[mask]).sum()
                        )
                        bucket["n_silent"] += int((per_sample[mask] == 0).sum())
                        bucket["n_spikes"] += float(per_sample[mask].sum())
                z_true = scores.gather(1, labels[:, None]).squeeze(1)
                other = scores.clone()
                other.scatter_(1, labels[:, None], float("-inf"))
                margin += float((z_true - other.max(1).values).sum())
                confidence += float(scores.softmax(1).gather(1, labels[:, None]).sum())
                scale += float(scores.abs().mean(1).sum())
            draws.append(
                {
                    "draw": draw_index,
                    "encoder_seed": encoder_seed,
                    "input_rate_seed": rate_seed,
                    "n_correct": draw_correct,
                    "n_total": draw_total,
                    "acc": 100 * draw_correct / draw_total,
                    "test_loss": draw_loss / draw_total,
                }
            )
    validation_loss = ce_sum / total
    accuracy = 100 * correct / total
    return {
        "acc": accuracy,
        "test_loss": validation_loss,
        "test_rate_e": e_sum / total,
        "test_rate_i": i_sum / total,
        "test_margin": margin / total,
        "test_confidence": confidence / total,
        "test_logit_scale": scale / total,
        "validation_draws": draws,
        "test_output_spikes_per_sample": output_spikes / output_samples
        if output_samples
        else None,
        "test_output_silent_fraction": output_silent / output_samples
        if output_samples
        else None,
        "test_output_class_spike_fraction": (class_spikes / class_spikes.sum()).tolist()
        if class_spikes.sum()
        else [0.0] * cfg["n_out"]
        if output_samples
        else None,
        "test_output_by_input_rate": [
            {
                "rate_hz": rate,
                "n_samples": bucket["n_samples"],
                "accuracy_pct": 100 * bucket["n_correct"] / bucket["n_samples"],
                "spikes_per_sample": bucket["n_spikes"] / bucket["n_samples"],
                "silent_fraction": bucket["n_silent"] / bucket["n_samples"],
            }
            for rate, bucket in sorted(by_rate.items())
        ],
    }


if __name__ == "__main__":
    main()
