"""Create immutable flat run folders for local and collection execution."""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

from .contracts import (
    RUN_SCHEMA,
    PingstoreError,
    run_root,
    validate_run,
    write_json_atomic,
)
from .payload import inventory_payload
from .registry import memberships


def _safe(value: object, fallback: str) -> str:
    text = re.sub(r"[^a-z0-9.-]+", "-", str(value).lower()).strip("-.")
    return text or fallback


def execution_origin(host: str | None = None) -> str:
    slurm_job = os.environ.get("SLURM_JOB_ID")
    if slurm_job:
        cluster = os.environ.get("SLURM_CLUSTER_NAME") or host or platform.node()
        return f"slurm-{_safe(cluster, 'cluster')}-{_safe(slurm_job, 'job')}"
    raw = host or "local"
    return "local" if raw == "local" else _safe(raw, "local")


def make_run_id(experiment: str, identity: str, origin: str) -> str:
    return f"{experiment}-{_safe(identity, 'run')}-{_safe(origin, 'unknown')}"


def _manifest(staging: Path) -> dict:
    path = staging / "_manifest.json"
    if not path.is_file():
        raise PingstoreError(f"cannot finalize run: {path} is missing")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise PingstoreError(f"cannot finalize run: {path} is not an object")
    return value


def capture_local_run(
    repo: Path,
    experiment: str,
    staging: Path,
    *,
    state: Path | None = None,
    root: Path | None = None,
) -> dict:
    """Copy a successful result into a flat immutable run directory."""
    repo = repo.resolve()
    manifest = _manifest(staging)
    identity = manifest.get("run_id")
    if not isinstance(identity, str) or not identity:
        raise PingstoreError("cannot finalize run: manifest run_id is missing")
    collection = memberships(repo).get(experiment)
    if collection is None:
        raise PingstoreError(f"cannot finalize {experiment}: collection is missing")
    origin = execution_origin(manifest.get("host"))
    run_id = make_run_id(experiment, identity, origin)
    store = (root or repo / ".pingstore").resolve()
    destination = run_root(store, run_id)
    temporary = destination.with_name("." + destination.name + ".tmp")
    if destination.exists() or temporary.exists():
        raise PingstoreError(f"run already exists: {run_id}")
    files = temporary / "files"
    files.parent.mkdir(parents=True, exist_ok=False)
    try:
        shutil.copytree(staging, files)
        if state is not None and state.exists():
            shutil.copytree(state, files / "state")
        inventory = inventory_payload(files, run_id=run_id)
        completed = datetime.now(timezone.utc).isoformat(timespec="seconds")
        run = {
            "schema": RUN_SCHEMA,
            "run_id": run_id,
            "experiment": experiment,
            "collection": collection,
            "origin": origin,
            "created_at": manifest.get("run_at") or completed,
            "execution": {
                "command": ["uv", "run", "python", f"experiments/{experiment}.py"],
                "configuration": manifest.get("scale"),
                "completed_at": completed,
            },
            "provenance": {
                "git_commit": manifest.get("git_sha"),
                "dirty": manifest.get("dirty"),
                "code_dirty": manifest.get("code_dirty"),
                "dirty_patch": manifest.get("patch"),
            },
            "files_digest": "sha256:" + inventory["payload_digest"],
        }
        validate_run(run)
        write_json_atomic(temporary / "run.json", run)
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.rename(temporary, destination)
        return run
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def finalize_local_run(repo: Path, experiment: str, temporary: Path) -> dict:
    """Finalize a run whose files were written directly in its hidden root."""
    repo = repo.resolve()
    files = temporary / "files"
    manifest = _manifest(files)
    identity = manifest.get("run_id")
    if not isinstance(identity, str) or not identity:
        raise PingstoreError("cannot finalize run: manifest run_id is missing")
    collection = memberships(repo).get(experiment)
    if collection is None:
        raise PingstoreError(f"cannot finalize {experiment}: collection is missing")
    origin = execution_origin(manifest.get("host"))
    run_id = make_run_id(experiment, identity, origin)
    destination = run_root(repo / ".pingstore", run_id)
    expected = destination.with_name("." + destination.name + ".tmp")
    if temporary.resolve() != expected.resolve():
        raise PingstoreError(f"working run must be {expected}")
    if destination.exists():
        raise PingstoreError(f"run already exists: {run_id}")
    inventory = inventory_payload(files, run_id=run_id)
    completed = datetime.now(timezone.utc).isoformat(timespec="seconds")
    run = {
        "schema": RUN_SCHEMA,
        "run_id": run_id,
        "experiment": experiment,
        "collection": collection,
        "origin": origin,
        "created_at": manifest.get("run_at") or completed,
        "execution": {
            "command": ["uv", "run", "python", f"experiments/{experiment}.py"],
            "configuration": manifest.get("scale"),
            "completed_at": completed,
        },
        "provenance": {
            "git_commit": manifest.get("git_sha"),
            "dirty": manifest.get("dirty"),
            "code_dirty": manifest.get("code_dirty"),
            "dirty_patch": manifest.get("patch"),
        },
        "files_digest": "sha256:" + inventory["payload_digest"],
    }
    validate_run(run)
    write_json_atomic(temporary / "run.json", run)
    os.rename(temporary, destination)
    return run


def capture_failed_local_run(
    repo: Path,
    experiment: str,
    staging: Path,
    *,
    state: Path | None = None,
    root: Path | None = None,
) -> Path:
    """Retain an incomplete result as a hidden folder."""
    manifest = _manifest(staging)
    identity = str(manifest.get("run_id") or "unknown")
    origin = execution_origin(manifest.get("host"))
    suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = make_run_id(experiment, f"{identity}-failed-{suffix}", origin)
    store = (root or repo.resolve() / ".pingstore").resolve()
    destination = run_root(store, run_id).with_name("." + run_id + ".tmp")
    if destination.exists():
        raise PingstoreError(f"incomplete run already exists: {destination.name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(staging, destination / "files")
    if state is not None and state.exists():
        shutil.copytree(state, destination / "files" / "state")
    return destination


def capture_campaign_metadata(root: Path, plan: dict) -> dict:
    """Capture each campaign experiment using the flat convention."""
    store = root / ".pingstore"
    captured: list[str] = []
    campaign = str(plan["campaign_id"])
    origin = execution_origin(platform.node())
    for stage in plan["stages"]:
        for row in stage["experiments"]:
            experiment = row["slug"]
            source = Path(row["paths"]["derived"])
            run_id = make_run_id(experiment, campaign, origin)
            destination = run_root(store, run_id)
            if destination.exists():
                captured.append(run_id)
                continue
            temporary = destination.with_name("." + destination.name + ".tmp")
            files = temporary / "files"
            files.parent.mkdir(parents=True, exist_ok=False)
            shutil.copytree(source, files)
            state_value = row["paths"].get("state")
            state = Path(state_value) if state_value else None
            if state is not None and state.exists():
                shutil.copytree(state, files / "state")
            inventory = inventory_payload(files, run_id=run_id)
            run = {
                "schema": RUN_SCHEMA,
                "run_id": run_id,
                "experiment": experiment,
                "collection": plan["collection"],
                "origin": origin,
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "execution": {"command": row["command"], "campaign_id": campaign},
                "provenance": plan.get("source", {}),
                "files_digest": "sha256:" + inventory["payload_digest"],
            }
            validate_run(run)
            write_json_atomic(temporary / "run.json", run)
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.rename(temporary, destination)
            captured.append(run_id)
    return {"root": str(store), "runs": captured}
