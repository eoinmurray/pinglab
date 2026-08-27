"""Create immutable v2 run folders for local and collection execution."""

from __future__ import annotations

import os
import platform
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

from .contracts import (
    RUN_SCHEMA,
    PingstoreError,
    load_json,
    payload_digest,
    run_root,
    validate_run_directory,
    write_json_atomic,
)
from .layout import (
    copy_legacy_derived,
    display_manifest,
    initialize_layout,
    read_execution_manifest,
)
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


def _local_record(repo: Path, experiment: str, manifest: dict) -> dict:
    identity = manifest.get("run_id")
    if not isinstance(identity, str) or not identity:
        raise PingstoreError("cannot finalize run: manifest run_id is missing")
    collection = memberships(repo).get(experiment)
    if collection is None:
        raise PingstoreError(f"cannot finalize {experiment}: collection is missing")
    origin = execution_origin(manifest.get("host"))
    completed = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return {
        "schema": RUN_SCHEMA,
        "run_id": make_run_id(experiment, identity, origin),
        "experiment": experiment,
        "collection": collection,
        "origin": origin,
        "created_at": manifest.get("run_at") or completed,
        "execution": {
            "command": ["uv", "run", "python", f"experiments/{experiment}.py"],
            "configuration": manifest.get("scale"),
            "completed_at": completed,
            "declaration": manifest,
        },
        "provenance": {
            "git_commit": manifest.get("git_sha"),
            "dirty": manifest.get("dirty"),
            "code_dirty": manifest.get("code_dirty"),
            "dirty_patch": manifest.get("patch"),
        },
    }


def _complete(
    temporary: Path, destination: Path, run: dict, manifest: dict | None = None
) -> dict:
    if destination.exists():
        raise PingstoreError(f"run already exists: {run['run_id']}")
    initialize_layout(temporary, run["experiment"])
    if manifest is not None:
        display_manifest(temporary, manifest, run["run_id"])
    run["payload_digest"] = payload_digest(temporary)
    write_json_atomic(temporary / "run.json", run)
    validate_run_directory(temporary)
    os.rename(temporary, destination)
    return run


def capture_local_run(
    repo: Path,
    experiment: str,
    staging: Path,
    *,
    state: Path | None = None,
    root: Path | None = None,
) -> dict:
    """Adapt externally staged legacy output into the v2 contract."""
    repo = repo.resolve()
    manifest = load_json(staging / "_manifest.json")
    run = _local_record(repo, experiment, manifest)
    destination = run_root((root or repo / ".pingstore").resolve(), run["run_id"])
    temporary = destination.with_name("." + destination.name + ".tmp")
    if destination.exists() or temporary.exists():
        raise PingstoreError(f"run already exists: {run['run_id']}")
    initialize_layout(temporary, experiment)
    copy_legacy_derived(staging, temporary)
    if state is not None and state.exists():
        shutil.copytree(state, temporary / "export/state")
    return _complete(temporary, destination, run, manifest)


def finalize_local_run(repo: Path, experiment: str, temporary: Path) -> dict:
    """Finalize only a hidden v2 run; failed validation leaves it hidden."""
    manifest = read_execution_manifest(temporary)
    run = _local_record(repo.resolve(), experiment, manifest)
    destination = run_root(repo.resolve() / ".pingstore", run["run_id"])
    expected = destination.with_name("." + destination.name + ".tmp")
    if temporary.resolve() != expected.resolve():
        raise PingstoreError(f"working run must be {expected}")
    return _complete(temporary, destination, run, manifest)


def capture_failed_local_run(
    repo: Path,
    experiment: str,
    staging: Path,
    *,
    state: Path | None = None,
    root: Path | None = None,
) -> Path:
    manifest = load_json(staging / "_manifest.json")
    identity = str(manifest.get("run_id") or "unknown")
    suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = make_run_id(
        experiment,
        f"{identity}-failed-{suffix}",
        execution_origin(manifest.get("host")),
    )
    store = (root or repo.resolve() / ".pingstore").resolve()
    destination = run_root(store, run_id).with_name("." + run_id + ".tmp")
    if destination.exists():
        raise PingstoreError(f"incomplete run already exists: {destination.name}")
    initialize_layout(destination, experiment)
    copy_legacy_derived(staging, destination)
    if state is not None and state.exists():
        shutil.copytree(state, destination / "export/state")
    return destination


def capture_campaign_metadata(root: Path, plan: dict) -> dict:
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
                validate_run_directory(destination)
                captured.append(run_id)
                continue
            temporary = destination.with_name("." + destination.name + ".tmp")
            if temporary.exists():
                raise PingstoreError(f"incomplete run already exists: {temporary}")
            initialize_layout(temporary, experiment)
            copy_legacy_derived(source, temporary)
            state_value = row["paths"].get("state")
            if state_value and Path(state_value).exists():
                shutil.copytree(state_value, temporary / "export/state")
            run = {
                "schema": RUN_SCHEMA,
                "run_id": run_id,
                "experiment": experiment,
                "collection": plan["collection"],
                "origin": origin,
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "execution": {"command": row["command"], "campaign_id": campaign},
                "provenance": plan.get("source", {}),
            }
            manifest = (
                load_json(source / "_manifest.json")
                if (source / "_manifest.json").exists()
                else None
            )
            _complete(temporary, destination, run, manifest)
            captured.append(run_id)
    return {"root": str(store), "runs": captured}
