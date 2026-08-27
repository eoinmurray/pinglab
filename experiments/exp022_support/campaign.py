"""Safe campaign orchestration for the exp022 Wilkes3 checkpoint bank.

This module deliberately contains no scientific registry.  Callers pass the
registry owned by :mod:`experiments.exp022`, so manifests, validation, status,
and scheduler retries cannot drift onto a second cell list.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shlex
import socket
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

SCHEMA = "pinglab.exp022.campaign"
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


def git_dirty_paths(repo: Path) -> list[str]:
    output = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo, check=True,
        capture_output=True, text=True,
    ).stdout
    return [line[3:] for line in output.splitlines() if len(line) > 3]


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
    *, repo: Path, campaign_root: Path, campaign_id: str,
    cells: list[dict[str, Any]], tier_for: Callable[[dict[str, Any]], str],
    samples_epochs: Callable[[dict[str, Any]], tuple[int, int]],
    build_args: Callable[[dict[str, Any], Path, int, int], list[str]],
    scientific_contract_for: Callable[
        [dict[str, Any], int, int], dict[str, Any]
    ] | None = None,
    plumbing: bool = False, selection_tier: str = "all",
) -> dict[str, Any]:
    root = campaign_root.resolve()
    if root == repo.resolve():
        raise ValueError("campaign root may not be the repository root")
    commit, dirty = git_identity(repo)
    if dirty:
        raise ValueError("refusing to create a campaign manifest from a dirty worktree")
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
        "campaign_id": campaign_id,
        "created_at_utc": utc_now(),
        "repository": {"commit": commit, "dirty": dirty},
        "environment": {
            "lockfile": lock_identity(repo),
            "python": platform.python_version(),
        },
        "campaign_root": str(root),
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
        raise ValueError(f"campaign manifest hash mismatch: expected {supplied}, got {actual}")
    payload["manifest_sha256"] = supplied
    if payload.get("schema") != SCHEMA or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported exp022 campaign manifest schema")
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
    "--model": "model", "--dataset": "dataset", "--max-samples": "max_samples",
    "--epochs": "epochs", "--t-ms": "t_ms", "--dt": "dt",
    "--tau-gaba": "tau_gaba_ms", "--seed": "seed", "--ei-strength": "ei_strength",
    "--v-grad-dampen": "v_grad_dampen", "--w-in-initial-zero-fraction": "w_in_initial_zero_fraction",
    "--readout": "readout_mode", "--surrogate-slope": "surrogate_slope",
    "--readout-w-out-scale": "readout_w_out_scale",
    "--readout-w-init-mean": "readout_w_init_mean",
    "--readout-w-init-std": "readout_w_init_std", "--lr": "lr",
    "--batch-size": "batch_size",
    "--fr-reg-upper-target-hz": "fr_reg_upper_target_hz",
    "--fr-reg-upper-strength": "fr_reg_upper_strength",
    "--input-rates": "input_rates",
    "--input-rate": "input_rate", "--n-hidden": "hidden_sizes",
    "--weight-decay": "weight_decay", "--dales-law": "dales_law",
    "--w-in": "w_in", "--trainable-w-ei": "trainable_w_ei",
    "--trainable-w-ie": "trainable_w_ie",
}
OPERATIONAL_ARGUMENTS = {"--out-dir"}
FLOAT_CONFIG = {
    "dt", "t_ms", "tau_gaba_ms", "ei_strength", "v_grad_dampen",
    "w_in_initial_zero_fraction", "surrogate_slope", "readout_w_out_scale",
    "readout_w_init_mean", "readout_w_init_std", "lr",
    "fr_reg_upper_target_hz", "fr_reg_upper_strength",
    "input_rate", "weight_decay",
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
        if payload.get("campaign_resolved_parameters") != cell["parameters"]:
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
        "campaign_id": manifest["campaign_id"],
        "campaign_manifest_sha256": manifest["manifest_sha256"],
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
    return Path(manifest["campaign_root"]) / "status" / f"{cell_name}.json"


def lock_path(manifest: dict[str, Any], cell_name: str) -> Path:
    return Path(manifest["campaign_root"]) / "status" / f"{cell_name}.lock"


def attempt_is_active(record: dict[str, Any]) -> bool | None:
    """Return True/False when activity is provable, otherwise None."""
    if record.get("state") != "running":
        return False
    job_id = record.get("slurm_job_id")
    if job_id:
        try:
            query = subprocess.run(
                ["squeue", "--noheader", "--jobs", str(job_id), "--format", "%A"],
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
        "campaign_id": manifest["campaign_id"], "counts": counts,
        "by_tier": by_tier, "by_training_run_id": by_tr,
        "retry_cells": [
            row["name"] for row in rows
            if not row["valid"] and not row["active"] and not row["stale"]
        ],
        "recoverable_cells": [row["name"] for row in rows if row["stale"]],
        "cells": rows,
    }


def print_status(status: dict[str, Any]) -> None:
    print(f"campaign {status['campaign_id']}")
    print(f"{'cell':44} {'TR':5} {'tier':16} state")
    for row in status["cells"]:
        print(f"{row['name'][:44]:44} {row['training_run_id']:5} {row['resource_tier']:16} {row['state']}")
    print("counts " + " ".join(f"{key}={value}" for key, value in sorted(status["counts"].items())))
    print(f"retry ({len(status['retry_cells'])}): " + " ".join(status["retry_cells"]))
