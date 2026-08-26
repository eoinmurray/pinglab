"""Explicit publication-view promotion with reverse provenance."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from uuid import uuid4

from .payload import (
    CONTRACT_VERSION,
    ContractError,
    load_json,
    validate_inventory,
    validate_run_manifest,
    verify_payload,
    write_json_atomic,
)

FIGURE_SUFFIXES = frozenset({".pdf", ".png", ".svg"})
GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _experiment_name(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or path.as_posix() != value
        or len(path.parts) != 1
        or value in {".", ".."}
    ):
        raise ContractError("experiment must be one safe path component")
    return value


def _source_rows(run_root: Path, source: Path, inventory: dict) -> list[dict]:
    indexed = {row["path"]: row for row in inventory["files"]}
    rows = []
    for item in sorted(
        source.rglob("*"), key=lambda path: path.relative_to(source).as_posix()
    ):
        if item.is_symlink():
            raise ContractError(f"promotion source contains a symlink: {item}")
        if not item.is_file():
            continue
        relative = item.relative_to(source).as_posix()
        source_relative = item.relative_to(run_root).as_posix()
        inventory_row = indexed.get(source_relative)
        if inventory_row is None:
            raise ContractError(
                f"promotion source is absent from inventory: {source_relative}"
            )
        rows.append(
            {
                "path": relative,
                "source_path": source_relative,
                "size_bytes": inventory_row["size_bytes"],
                "sha256": inventory_row["sha256"],
            }
        )
    return rows


def _composition_source(run_root: Path, experiment: str) -> dict | None:
    path = run_root / "composition.json"
    if not path.is_file():
        return None
    document = load_json(path)
    if document.get("run_id") != load_json(run_root / "run.json").get("run_id"):
        raise ContractError("composition ledger belongs to another run")
    row = document.get("experiments", {}).get(experiment)
    if not isinstance(row, dict):
        raise ContractError(f"composition ledger is missing {experiment}")
    return row


def _generating_git_commit(
    run: dict, source: Path, run_root: Path | None = None
) -> str:
    """Prefer experiment-level campaign provenance for repaired outputs."""
    if run_root is not None:
        composition = _composition_source(run_root, source.name)
        if composition is not None:
            commit = composition.get("source_git_commit")
            if not isinstance(commit, str) or not GIT_COMMIT.fullmatch(commit):
                raise ContractError(
                    f"composition source commit is invalid: {source.name}"
                )
            return commit
    numbers = source / "numbers.json"
    try:
        document = json.loads(numbers.read_text())
    except (OSError, json.JSONDecodeError):
        return run["source"]["git_commit"]
    provenance = document.get("collection_provenance")
    if run.get("kind") != "campaign" or not isinstance(provenance, dict):
        return run["source"]["git_commit"]
    commit = provenance.get("source_git_commit")
    if provenance.get("campaign_id") != run["run_id"] or not isinstance(commit, str):
        return run["source"]["git_commit"]
    return commit if GIT_COMMIT.fullmatch(commit) else run["source"]["git_commit"]


def promote_experiment(
    run_root: Path,
    experiment: str,
    *,
    artifacts_root: Path,
    promoted_at_utc: str | None = None,
) -> dict:
    """Promote one inventoried experiment directory into the active artifact view."""
    run_root = run_root.resolve()
    experiment = _experiment_name(experiment)
    run = validate_run_manifest(load_json(run_root / "run.json"))
    inventory = validate_inventory(load_json(run_root / "inventory.json"))
    if run["run_id"] != inventory["run_id"]:
        raise ContractError("run.json and inventory.json use different run IDs")
    if run["status"] not in {"complete", "legacy"}:
        raise ContractError("only complete or legacy runs can be promoted")
    verify_payload(run_root, inventory)

    source_relative = Path("derived") / "artifacts" / "data" / experiment
    source = run_root / source_relative
    if not source.is_dir():
        raise ContractError(f"promotion source is not a directory: {source}")
    if (source / "_provenance.json").exists():
        raise ContractError("promotion source already contains reverse provenance")
    if not (source / "numbers.json").is_file():
        raise ContractError("promotion source requires numbers.json")
    if not any(
        item.is_file() and item.suffix.lower() in FIGURE_SUFFIXES
        for item in source.rglob("*")
    ):
        raise ContractError(
            "promotion source requires at least one PDF, PNG, or SVG figure"
        )

    rows = _source_rows(run_root, source, inventory)
    timestamp = promoted_at_utc or datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    ).replace("+00:00", "Z")
    provenance = {
        "contract_version": CONTRACT_VERSION,
        "run_id": run["run_id"],
        "campaign_id": run["run_id"] if run["kind"] in {"campaign", "legacy"} else None,
        "generating_git_commit": _generating_git_commit(run, source, run_root),
        "campaign_source_git_commit": run["source"]["git_commit"],
        "executor": run["execution"].get("executor", "legacy"),
        "graph_digest": run["execution"].get("graph_digest"),
        "training_digest": run["execution"].get("training_digest"),
        "source_directory": source_relative.as_posix(),
        "source_inventory_payload_digest": inventory["payload_digest"],
        "archive": run["archive"],
        "promoted_at_utc": timestamp,
        "files": rows,
    }
    composition = _composition_source(run_root, experiment)
    if composition is not None:
        provenance["composition_source"] = composition

    artifacts_root = artifacts_root.resolve()
    artifacts_root.mkdir(parents=True, exist_ok=True)
    destination = artifacts_root / experiment
    staging = Path(
        tempfile.mkdtemp(dir=artifacts_root, prefix=f".{experiment}.staging-")
    )
    backup = artifacts_root / f".{experiment}.backup-{uuid4().hex}"
    moved_existing = False
    try:
        for item in source.rglob("*"):
            relative = item.relative_to(source)
            target = staging / relative
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            elif item.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, target)
        for row in rows:
            promoted = staging / row["path"]
            if (
                promoted.stat().st_size != row["size_bytes"]
                or _sha256(promoted) != row["sha256"]
            ):
                raise ContractError(f"promoted file differs from source: {row['path']}")
        write_json_atomic(staging / "_provenance.json", provenance)

        if destination.exists():
            destination.rename(backup)
            moved_existing = True
        staging.rename(destination)
        if moved_existing:
            shutil.rmtree(backup)
        return {
            "run_id": run["run_id"],
            "experiment": experiment,
            "destination": str(destination),
            "file_count": len(rows),
            "payload_digest": inventory["payload_digest"],
        }
    except Exception:
        if moved_existing and not destination.exists() and backup.exists():
            backup.rename(destination)
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
