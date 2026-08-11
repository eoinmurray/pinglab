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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


SCHEMA = "pinglab.exp022.campaign"
SCHEMA_VERSION = 1
REQUIRED_CELL_FILES = ("config.json", "metrics.json", "metrics.jsonl", "weights.pth")


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
    return {
        "training_run_id": cell["training_run_id"],
        "family": cell["family"],
        "model_recipe": cell["model"],
        "seed": cell["seed"],
        "max_samples": max_samples,
        "epochs": epochs,
        "arguments": values,
    }


def create_manifest(
    *, repo: Path, campaign_root: Path, campaign_id: str,
    cells: list[dict[str, Any]], tier_for: Callable[[dict[str, Any]], str],
    samples_epochs: Callable[[dict[str, Any]], tuple[int, int]],
    build_args: Callable[[dict[str, Any], Path, int, int], list[str]],
    plumbing: bool = False,
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
        command = [sys.executable, str(repo / "tools" / "snn" / "tool.py"), *args]
        rows.append({
            "name": cell["name"],
            "training_run_id": cell["training_run_id"],
            "family": cell["family"],
            "resource_tier": tier_for(cell),
            "parameters": resolved_parameters(cell, args, max_samples, epochs),
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
    return actual == expected


ARG_TO_CONFIG = {
    "--model": "model", "--dataset": "dataset", "--max-samples": "max_samples",
    "--epochs": "epochs", "--t-ms": "t_ms", "--dt": "dt",
    "--tau-gaba": "tau_gaba_ms", "--seed": "seed", "--ei-strength": "ei_strength",
    "--v-grad-dampen": "v_grad_dampen", "--w-in-sparsity": "w_in_sparsity",
    "--readout": "readout_mode", "--surrogate-slope": "surrogate_slope",
    "--readout-w-out-scale": "readout_w_out_scale", "--lr": "lr",
    "--batch-size": "batch_size", "--fr-reg-upper-theta": "fr_reg_upper_theta",
    "--fr-reg-upper-strength": "fr_reg_upper_strength",
    "--input-rates": "input_rates_hz",
}
FLOAT_CONFIG = {
    "dt", "t_ms", "tau_gaba_ms", "ei_strength", "v_grad_dampen",
    "w_in_sparsity", "surrogate_slope", "readout_w_out_scale", "lr",
    "fr_reg_upper_theta", "fr_reg_upper_strength",
}
INT_CONFIG = {"max_samples", "epochs", "seed", "batch_size"}


def _expected_config(cell: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for flag, raw in cell["parameters"]["arguments"].items():
        key = ARG_TO_CONFIG.get(flag)
        if key is None:
            continue
        if key in FLOAT_CONFIG:
            result[key] = float(raw)
        elif key in INT_CONFIG:
            result[key] = int(raw)
        elif key == "input_rates_hz":
            result[key] = [float(value) for value in raw]
        else:
            result[key] = raw
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
    nested = metrics.get("config", {})
    expected = _expected_config(cell)
    for key, wanted in expected.items():
        actual = config.get(key, nested.get(key))
        if not _same(actual, wanted):
            reasons.append(f"config {key} mismatch: {actual!r} != {wanted!r}")
    epochs = int(cell["parameters"]["epochs"])
    samples = int(cell["parameters"]["max_samples"])
    if len(history) < epochs or int(history[-1].get("ep", -1)) < epochs:
        reasons.append(f"history did not reach epoch {epochs}")
    observed_samples = [row.get("samples") for row in history if row.get("samples") is not None]
    if observed_samples and max(int(value) for value in observed_samples) < samples:
        reasons.append(f"history did not reach {samples} samples")
    if load_checkpoint:
        try:
            import torch
            checkpoint = torch.load(directory / "weights.pth", map_location="cpu", weights_only=True)
            if not isinstance(checkpoint, dict) or not checkpoint:
                reasons.append("checkpoint is not a non-empty mapping")
            elif not any("out" in str(key).lower() or "readout" in str(key).lower()
                         for key in checkpoint):
                reasons.append("checkpoint has no recognizable readout parameters")
        except Exception as exc:  # noqa: BLE001 - corrupt checkpoints must classify, not crash
            reasons.append(f"checkpoint load failed: {type(exc).__name__}: {exc}")
    return {"valid": not reasons, "state": "complete" if not reasons else "invalid", "reasons": reasons}


def preserve_partial(directory: Path) -> Path | None:
    if not directory.exists() or not any(directory.iterdir()):
        return None
    failed_root = directory.parents[1] / "failed" / directory.name
    destination = failed_root / utc_now().replace(":", "-")
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
        "command": cell["command"],
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "gpu": {"cuda_visible_devices": gpu} if gpu is not None else {},
        "started_at_utc": utc_now(),
        "state": "running",
    }


def summarize_status(manifest: dict[str, Any], *, load_checkpoint: bool = True) -> dict[str, Any]:
    rows = []
    for cell in manifest["cells"]:
        result = validate_cell(cell, load_checkpoint=load_checkpoint)
        record_path = Path(cell["output_directory"]) / "attempt.json"
        if not result["valid"] and record_path.exists():
            try:
                attempt = _json(record_path)
                if attempt.get("state") == "running":
                    result["state"] = "running"
                elif attempt.get("state") == "failed" and result["state"] != "invalid":
                    result["state"] = "failed"
            except Exception:  # noqa: BLE001
                pass
        rows.append({
            "name": cell["name"], "training_run_id": cell["training_run_id"],
            "resource_tier": cell["resource_tier"], **result,
        })
    counts: dict[str, int] = {}
    by_tier: dict[str, dict[str, int]] = {}
    by_tr: dict[str, dict[str, int]] = {}
    for row in rows:
        counts[row["state"]] = counts.get(row["state"], 0) + 1
        for grouping, key in ((by_tier, row["resource_tier"]), (by_tr, row["training_run_id"])):
            grouping.setdefault(key, {})[row["state"]] = grouping.setdefault(key, {}).get(row["state"], 0) + 1
    return {
        "campaign_id": manifest["campaign_id"], "counts": counts,
        "by_tier": by_tier, "by_training_run_id": by_tr,
        "retry_cells": [row["name"] for row in rows if not row["valid"]],
        "cells": rows,
    }


def print_status(status: dict[str, Any]) -> None:
    print(f"campaign {status['campaign_id']}")
    print(f"{'cell':44} {'TR':5} {'tier':16} state")
    for row in status["cells"]:
        print(f"{row['name'][:44]:44} {row['training_run_id']:5} {row['resource_tier']:16} {row['state']}")
    print("counts " + " ".join(f"{key}={value}" for key, value in sorted(status["counts"].items())))
    print(f"retry ({len(status['retry_cells'])}): " + " ".join(status["retry_cells"]))
