"""Immutable archive, remote verification, and safe restoration operations."""

from __future__ import annotations

import copy
import json
import os
import re
import tempfile
from pathlib import Path

from .contract import (
    ContractError,
    load_json,
    validate_inventory,
    validate_run_manifest,
    verify_payload,
)
from .storage import Store, StoredObject

ARCHIVE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def validate_archive_id(archive_id: str) -> str:
    if not ARCHIVE_ID_RE.fullmatch(archive_id):
        raise ContractError(
            "archive ID must start with an alphanumeric character and contain "
            "only letters, digits, dot, underscore, or hyphen"
        )
    return archive_id


def _json_bytes(value: dict) -> bytes:
    return (json.dumps(value, indent=2) + "\n").encode()


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_local(root: Path) -> tuple[dict, dict]:
    root = root.resolve()
    run = validate_run_manifest(load_json(root / "run.json"))
    inventory = validate_inventory(load_json(root / "inventory.json"))
    if run["run_id"] != inventory["run_id"]:
        raise ContractError("run.json and inventory.json use different run IDs")
    verify_payload(root, inventory)
    return run, inventory


def archive_run(root: Path, archive_id: str, store: Store) -> dict:
    root = root.resolve()
    validate_archive_id(archive_id)
    run, inventory = _load_local(root)
    if run.get("archive") is not None:
        raise ContractError("run.json already records an archive identity")
    if store.exists(archive_id):
        raise ContractError(f"archive identity already exists: {archive_id}")

    archived_run = copy.deepcopy(run)
    archived_run["archive"] = {
        "archive_id": archive_id,
        "uri": store.logical_uri(archive_id),
    }
    validate_run_manifest(archived_run)
    store.put_archive(
        archive_id,
        root,
        _json_bytes(archived_run),
        _json_bytes(inventory),
    )
    verify_archive(store, archive_id)
    _atomic_json(root / "run.json", archived_run)
    return archived_run


def _expected_objects(inventory: dict) -> dict[str, int]:
    expected = {row["path"]: row["size_bytes"] for row in inventory["files"]}
    expected["run.json"] = -1
    expected["inventory.json"] = -1
    return expected


def _object_map(objects: list[StoredObject]) -> dict[str, int]:
    result: dict[str, int] = {}
    for item in objects:
        if item.path in result:
            raise ContractError(f"archive contains duplicate object: {item.path}")
        result[item.path] = item.size_bytes
    return result


def verify_archive(store: Store, archive_id: str) -> dict:
    validate_archive_id(archive_id)
    try:
        run = validate_run_manifest(
            json.loads(store.read_bytes(archive_id, "run.json"))
        )
        inventory = validate_inventory(
            json.loads(store.read_bytes(archive_id, "inventory.json"))
        )
    except json.JSONDecodeError as exc:
        raise ContractError("archive contains invalid JSON manifests") from exc
    if run["run_id"] != inventory["run_id"]:
        raise ContractError("archived manifests use different run IDs")
    archive = run.get("archive")
    if archive is None or archive.get("archive_id") != archive_id:
        raise ContractError("archived run.json does not identify this archive")

    actual = _object_map(store.objects(archive_id))
    expected = _expected_objects(inventory)
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))
        unexpected = sorted(set(actual) - set(expected))
        raise ContractError(
            f"archive object set differs; missing={missing}, unexpected={unexpected}"
        )
    for row in inventory["files"]:
        path = row["path"]
        if actual[path] != row["size_bytes"]:
            raise ContractError(f"archive size mismatch: {path}")
        if store.sha256(archive_id, path) != row["sha256"]:
            raise ContractError(f"archive SHA-256 mismatch: {path}")
    return {
        "archive_id": archive_id,
        "uri": archive["uri"],
        "run_id": run["run_id"],
        "file_count": inventory["file_count"],
        "total_size_bytes": inventory["total_size_bytes"],
        "payload_digest": inventory["payload_digest"],
    }


def restore_archive(store: Store, archive_id: str, destination: Path) -> dict:
    validate_archive_id(archive_id)
    destination = destination.resolve()
    if destination.exists():
        raise ContractError(f"restore destination already exists: {destination}")
    verify_archive(store, archive_id)
    destination.parent.mkdir(parents=True, exist_ok=True)
    store.restore(archive_id, destination)
    try:
        run, inventory = _load_local(destination)
        if run.get("archive", {}).get("archive_id") != archive_id:
            raise ContractError("restored run.json has the wrong archive identity")
    except Exception:
        # Preserve the partial restore for diagnosis; never silently remove evidence.
        raise
    return {
        "archive_id": archive_id,
        "destination": str(destination),
        "run_id": run["run_id"],
        "file_count": inventory["file_count"],
        "total_size_bytes": inventory["total_size_bytes"],
        "payload_digest": inventory["payload_digest"],
    }
