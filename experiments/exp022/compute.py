"""Exp022 compute: model-bank training and retained diagnostic simulations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import shutil
import socket
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "experiments"), str(REPO / "tools")]

from experiments.exp022 import recipe
from pingstore.contracts import PingstoreError, write_json_atomic
from pingstore.stages import reserve_stage, source_run, stage_run

from experiments.exp022.checkpoints import resolve_checkpoint

SCHEMA = "pinglab.exp022.bank"
SCHEMA_VERSION = 1
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
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo, check=True,
        capture_output=True, text=True,
    ).stdout.strip())
    return commit, dirty


def lock_identity(repo: Path) -> dict[str, Any]:
    path = repo / "uv.lock"
    return {
        "path": "uv.lock",
        "sha256": sha256_file(path) if path.exists() else None,
    }


def resolved_parameters(
    cell: dict[str, Any], args: list[str], max_samples: int, epochs: int,
    scientific_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Cold-readable scientific contract, including the exact CLI argument map."""
    values: dict[str, Any] = {}
    index = 1  # skip the ``train`` verb
    while index < len(args):
        token = args[index]
        if not token.startswith("--"):
            index += 1
            continue
        if index + 1 >= len(args) or args[index + 1].startswith("--"):
            values[token] = True
            index += 1
            continue
        following: list[str] = []
        index += 1
        while index < len(args) and not args[index].startswith("--"):
            following.append(args[index])
            index += 1
        values[token] = following[0] if len(following) == 1 else following
    values.pop("--wipe-dir", None)
    result = {
        "training_run_id": cell["training_run_id"],
        "family": cell["family"],
        "model_recipe": cell["model"],
        "seed": cell["seed"],
        "max_samples": max_samples,
        "epochs": epochs,
        "arguments": values,
    }
    if scientific_contract is not None:
        result["scientific_contract"] = scientific_contract
    return result


def create_manifest(
    *, repo: Path, bank_root: Path, bank_id: str,
    cells: list[dict[str, Any]], tier_for: Callable[[dict[str, Any]], str],
    samples_epochs: Callable[[dict[str, Any]], tuple[int, int]],
    build_args: Callable[[dict[str, Any], Path, int, int], list[str]],
    scientific_contract_for: Callable[
        [dict[str, Any], int, int], dict[str, Any]
    ] | None = None,
    plumbing: bool = False, selection_tier: str = "all",
) -> dict[str, Any]:
    root = bank_root.resolve()
    if root == repo.resolve():
        raise ValueError("bank working root may not be the repository root")
    commit, dirty = git_identity(repo)
    if dirty:
        raise ValueError("refusing to create a bank manifest from a dirty worktree")
    rows = []
    for cell in cells:
        max_samples, epochs = samples_epochs(cell)
        spec = ({k: v for k, v in cell.items() if k != "max_samples"}
                if plumbing else cell)
        out = root / "cells" / cell["name"]
        args = build_args(spec, out, max_samples, epochs)
        command = [python_executable(), str(repo / "tools" / "snnsim" / "tool.py"), *args]
        rows.append({
            "name": cell["name"],
            "training_run_id": cell["training_run_id"],
            "family": cell["family"],
            "resource_tier": tier_for(cell),
            "parameters": resolved_parameters(
                cell, args, max_samples, epochs,
                scientific_contract=(
                    scientific_contract_for(cell, max_samples, epochs)
                    if scientific_contract_for else None
                ),
            ),
            "command": command,
            "command_shell": shlex.join(command),
            "output_directory": str(out),
            "required_outputs": list(REQUIRED_CELL_FILES),
        })
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "bank_id": bank_id,
        "created_at_utc": utc_now(),
        "repository": {"commit": commit, "dirty": dirty},
        "environment": {
            "lockfile": lock_identity(repo),
            "python": platform.python_version(),
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
        raise ValueError(f"bank manifest hash mismatch: expected {supplied}, got {actual}")
    payload["manifest_sha256"] = supplied
    if payload.get("schema") != SCHEMA or payload.get("schema_version") != SCHEMA_VERSION:
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


ARG_TO_CONFIG = {
    "--refractory-e-ms": "refractory_e_ms",
    "--refractory-i-ms": "refractory_i_ms",
    "--refractory-policy": "refractory_policy",
    "--model": "model",
    "--dataset": "dataset",
    "--max-samples": "max_samples",
    "--epochs": "epochs",
    "--t-ms": "t_ms",
    "--dt": "dt",
    "--tau-gaba": "tau_gaba_ms",
    "--seed": "seed",
    "--ei-strength": "ei_strength",
    "--v-grad-dampen": "v_grad_dampen",
    "--w-in-initial-zero-fraction": "w_in_initial_zero_fraction",
    "--readout": "readout_mode",
    "--surrogate-slope": "surrogate_slope",
    "--readout-w-out-scale": "readout_w_out_scale",
    "--readout-w-init-mean": "readout_w_init_mean",
    "--readout-w-init-std": "readout_w_init_std",
    "--lr": "lr",
    "--batch-size": "batch_size",
    "--fr-reg-upper-target-hz": "fr_reg_upper_target_hz",
    "--fr-reg-upper-strength": "fr_reg_upper_strength",
    "--input-rates": "input_rates",
    "--input-rate": "input_rate",
    "--n-hidden": "hidden_sizes",
    "--weight-decay": "weight_decay",
    "--dales-law": "dales_law",
    "--w-in": "w_in",
    "--trainable-w-ei": "trainable_w_ei",
    "--trainable-w-ie": "trainable_w_ie",
}
OPERATIONAL_ARGUMENTS = {"--out-dir"}
FLOAT_CONFIG = {
    "refractory_e_ms",
    "refractory_i_ms",
    "dt",
    "t_ms",
    "tau_gaba_ms",
    "ei_strength",
    "v_grad_dampen",
    "w_in_initial_zero_fraction",
    "surrogate_slope",
    "readout_w_out_scale",
    "readout_w_init_mean",
    "readout_w_init_std",
    "lr",
    "fr_reg_upper_target_hz",
    "fr_reg_upper_strength",
    "input_rate",
    "weight_decay",
}
INT_CONFIG = {"max_samples", "epochs", "seed", "batch_size"}
BOOL_CONFIG = {"dales_law", "trainable_w_ei", "trainable_w_ie"}


def _expected_config(cell: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for flag, raw in cell["parameters"]["arguments"].items():
        key = ARG_TO_CONFIG.get(flag)
        if key is None:
            if flag in OPERATIONAL_ARGUMENTS:
                continue
            raise ValueError(
                f"manifest argument {flag!r} has no saved-config mapping or "
                "operational exemption"
            )
        if key in FLOAT_CONFIG:
            result[key] = float(raw)
        elif key in INT_CONFIG:
            result[key] = int(raw)
        elif key == "input_rates":
            result[key] = [float(value) for value in raw]
        elif key == "hidden_sizes":
            values = raw if isinstance(raw, list) else [raw]
            result[key] = [int(value) for value in values]
        elif key == "w_in":
            mean = float(raw)
            result[key] = [mean, mean * 0.1]
        elif key in BOOL_CONFIG:
            result[key] = bool(raw)
        else:
            result[key] = raw
    contract = cell["parameters"].get("scientific_contract")
    if contract is not None:
        result.update({
            "n_in": int(contract["input"]["channels"]),
            "n_hidden": int(contract["topology"]["excitatory_neurons"]),
            "n_inh": int(contract["topology"]["inhibitory_neurons"]),
            "n_out": int(contract["topology"]["output_neurons"]),
            "tau_ampa_ms": float(contract["dynamics"]["tau_ampa_ms"]),
            "grad_clip": float(contract["optimizer"]["gradient_clip_norm"]),
            "input_rate_sampling": contract["input"]["rate_sampling"],
        })
    return result


def validate_cell(cell: dict[str, Any], *, load_checkpoint: bool = True) -> dict[str, Any]:
    directory = Path(cell["output_directory"])
    missing = [name for name in REQUIRED_CELL_FILES if not (directory / name).is_file()]
    if missing:
        state = "missing" if len(missing) == len(REQUIRED_CELL_FILES) else "partial"
        return {"valid": False, "state": state, "reasons": [f"missing {name}" for name in missing]}
    reasons: list[str] = []
    try:
        config = _json(directory / "config.json")
        metrics = _json(directory / "metrics.json")
        history = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines() if line]
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
    if not isinstance(initialization, dict) or not required_roles <= set(initialization):
        reasons.append("config missing complete weight initialization provenance")
    elif initialization != metrics_initialization:
        reasons.append("config/metrics weight initialization provenance mismatch")
    else:
        for role, record in initialization.items():
            if record.get("zeros_remain_trainable") is not True:
                reasons.append(f"{role} does not declare trainable initialization zeros")
            if record.get("distribution") not in {
                "lower_clamped_normal", "signed_normal", "kaiming_uniform_signed",
                "uniform", "constant", "zeros",
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
    observed_samples = [row.get("samples") for row in history if row.get("samples") is not None]
    expected_train_samples = round(samples * 0.9)  # fixed MNIST validation split
    if len(observed_samples) < epochs or any(
        int(value) != expected_train_samples for value in observed_samples[:epochs]
    ):
        reasons.append(
            f"history does not record {expected_train_samples} training samples "
            f"for each of {epochs} epochs"
        )
    if load_checkpoint:
        import torch

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
                checkpoint = torch.load(path, map_location="cpu", weights_only=True)
                if not isinstance(checkpoint, dict) or not checkpoint:
                    reasons.append(f"{role} checkpoint is not a non-empty mapping")
                    continue
                n_in = int(config.get("n_in", 784))
                n_hidden = int(config.get("n_hidden", 1024))
                n_inh = int(config.get("n_inh", 256))
                expected_shapes = {
                    "W_ff.0": (n_in, n_hidden),
                    "W_ff.1": (n_hidden, 10),
                    "W_ei.1": (n_hidden, n_inh),
                    "W_ie.1": (n_inh, n_hidden),
                }
                for key, shape in expected_shapes.items():
                    value = checkpoint.get(key)
                    if value is None or tuple(value.shape) != shape:
                        reasons.append(f"{role} checkpoint {key} shape mismatch")
            except Exception as exc:  # noqa: BLE001
                reasons.append(
                    f"{role} checkpoint load failed: {type(exc).__name__}: {exc}"
                )
    return {"valid": not reasons, "state": "complete" if not reasons else "invalid", "reasons": reasons}


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
            cell["name"], cell.get("command"),
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
                ["squeue", "--noheader", "--states=all", "--jobs", str(job_id), "--format", "%A"],
                capture_output=True, text=True,
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
    manifest: dict[str, Any], cell: dict[str, Any], *, recover_stale: bool = False,
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
                    recovery, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600,
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
                    lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600,
                )
                with os.fdopen(descriptor, "w") as handle:
                    json.dump({"attempt_id": attempt_id, "cell_name": cell["name"]}, handle)
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


def summarize_status(manifest: dict[str, Any], *, load_checkpoint: bool = True) -> dict[str, Any]:
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
        rows.append({
            "name": cell["name"], "training_run_id": cell["training_run_id"],
            "resource_tier": cell["resource_tier"], "active": active,
            "stale": stale, **result,
        })
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
        "bank_id": manifest["bank_id"], "counts": counts,
        "by_tier": by_tier, "by_training_run_id": by_tr,
        "retry_cells": [
            row["name"] for row in rows
            if not row["valid"] and not row["active"] and not row["stale"]
        ],
        "recoverable_cells": [row["name"] for row in rows if row["stale"]],
        "cells": rows,
    }



def cell_dir(name: str) -> Path:
    """Shared per-cell artifact directory."""
    return recipe.training_root() / name


def load_cell(name: str) -> Path:
    """Return a trained cell's directory, or fail loudly if this notebook has
    not been run. Analysis notebooks call this instead of training."""
    d = cell_dir(name)
    if not (d / "weights.pth").exists() or not (d / "weights_final.pth").exists():
        raise SystemExit(
            f"missing trained cell '{name}' at {recipe._display_path(d)}; "
            "run exp022 (Training) first to produce the shared cells."
        )
    return d


def _train_one_cell(cell: dict, plumbing: bool) -> None:
    """Train one registered cell under the configured working root."""
    _writable_compute_root(recipe.training_root())
    ms, ep = recipe.cell_samples_epochs(cell)  # honours PINGLAB_NB022_PLUMBING
    spec = cell
    if plumbing:
        # build_train_args re-applies a canonical cell's own max_samples (60000),
        # which would defeat the tiny plumbing scale. Strip it so the plumbing
        # ms=100 takes and the retained parameters match what was trained.
        spec = {k: v for k, v in cell.items() if k != "max_samples"}
    args = recipe.build_train_args(spec, cell_dir(cell["name"]), ms, ep)
    print(
        f"[train-cell] {cell['training_run_id']} / {cell['name']} "
        f"(n={ms}, {ep} ep) → {cell_dir(cell['name'])}"
    )
    subprocess.run([sys.executable, str(recipe.SNN_TOOL), *args], cwd=REPO, check=True)
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
    parser.add_argument("--plumbing", action="store_true")
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
    tier = manifest.get("selection", {}).get("tier")
    try:
        selected_cells = (
            [cell for cell in recipe.CANONICAL_CELLS
             if cell["family"] == "dt" and cell["dt_ms"] != recipe.DT_MS]
            if tier == "refractory-replacement"
            else recipe.cells_in_resource_tier(tier)
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    manifest_names_list = [row.get("name") for row in manifest.get("cells", [])]
    if len(manifest_names_list) != len(set(manifest_names_list)):
        raise SystemExit("bank manifest contains duplicate cell names")
    expected_names_list = [cell["name"] for cell in selected_cells]
    if manifest_names_list != expected_names_list:
        raise SystemExit(
            "bank cell list does not exactly match its declared selection"
        )
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
            train_args = recipe.build_train_args(
                command_spec, root / "cells" / spec["name"], samples, epochs
            )
            resolved = resolved_parameters(
                spec,
                train_args,
                samples,
                epochs,
                scientific_contract=recipe.scientific_contract(spec, samples, epochs),
            )
            command = [python_executable(), str(recipe.SNN_TOOL), *train_args]
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
        command = manifest["_runtime_commands"][name]
        completed = subprocess.run(command, cwd=REPO)
        exit_code = completed.returncode
        if exit_code == 0:
            spec = _cell_by_name(name)
            assert spec is not None
            old_root = recipe.TRAINING_ROOT
            try:
                recipe.TRAINING_ROOT = Path(manifest["bank_root"]) / "cells"
                _stamp_training_run_identity(spec)
            finally:
                recipe.TRAINING_ROOT = old_root
            _stamp_bank_identity(directory, manifest, row)
        validation = validate_cell(row)
        try:
            metrics_payload = recipe.load_metrics(directory)
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
        if args.plumbing:
            os.environ["PINGLAB_NB022_PLUMBING"] = "1"
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
            build_args=recipe.build_train_args,
            scientific_contract_for=recipe.scientific_contract,
            plumbing=args.plumbing,
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
    status = summarize_status(
        manifest, load_checkpoint=bool(args.bank_finalize)
    )
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
    actual = {path.name for path in bank.iterdir() if path.is_dir()}
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
        args = [
            sys.executable,
            str(recipe.SNN_TOOL),
            "sim",
            *recipe.refractory_args(),
            "--infer",
            "--load-config",
            str(trained / "config.json"),
            "--load-weights",
            str(checkpoint["path"]),
            "--digit",
            "0",
            "--sample",
            "0",
            "--out-dir",
            str(destination),
        ]
        if cell["family"] == "variable_rate":
            args += ["--input-rate", "5"]
        print(f"[compute probe] {cell['name']}", flush=True)
        subprocess.run(args, cwd=REPO, check=True)
        write_json_atomic(
            destination / "probe-command.json",
            {
                "command": args,
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

    actual = {path.name for path in bank.iterdir() if path.is_dir()}
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
    return inventory


def _finalize_bank(manifest_path: Path, manifest: dict, status: dict) -> str:
    """Validate a complete parallel bank and atomically finish its compute run."""
    if len(manifest["cells"]) != len(recipe.CANONICAL_CELLS):
        raise SystemExit("bank finalization requires the complete 102-cell recipe")
    incomplete = [row["name"] for row in status["cells"] if not row["valid"]]
    if incomplete:
        raise SystemExit(f"bank finalization refused: {len(incomplete)} cells are invalid")
    attempts = {}
    for row in manifest["cells"]:
        name = row["name"]
        if lock_path(manifest, name).exists():
            raise SystemExit(f"bank finalization refused: {name} still has an attempt lock")
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
            raise SystemExit(f"bank finalization refused: invalid attempt record for {name}")
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
        configuration=recipe.SCALE,
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
    if _handle_bank_cli(sys.argv[1:]):
        return
    retired = {"--skip-training", "--plot-only", "--only-missing"} & set(sys.argv[1:])
    if retired:
        raise SystemExit(
            "combined lifecycle flags are retired: use analyse.py --source COMPUTE_RUN "
            "or present.py --source ANALYSIS_RUN"
        )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hpc",
        action="store_true",
        help="create a fresh 102-cell bank and submit its Slurm array",
    )
    parser.add_argument(
        "--hpc-root",
        type=Path,
        help="dedicated absolute working directory for --hpc (or EXP022_HPC_ROOT)",
    )
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
    if args.hpc or args.hpc_root is not None:
        parser.error(
            "--hpc/--hpc-root are retired; use experiments.exp022.hpc prepare/review "
            "so the "
            "cell allocation is frozen and reviewed before receipt-first submission"
        )
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
        configuration=recipe.SCALE,
    ) as run:
        if inputs:
            bank = inputs["bank"].export
        else:
            bank = run.export / "cells"
            previous = recipe.TRAINING_ROOT
            try:
                recipe.TRAINING_ROOT = bank
                for cell in recipe.CANONICAL_CELLS:
                    _train_one_cell(
                        cell, os.environ.get("PINGLAB_NB022_PLUMBING") == "1"
                    )
            finally:
                recipe.TRAINING_ROOT = previous
        generate_snapshots(bank, run.export / "snapshots")
        if not inputs:
            promote_cells(run.export)


if __name__ == "__main__":
    main()
