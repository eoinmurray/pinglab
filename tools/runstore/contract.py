"""Version-1 runstore contract validation and deterministic inventories."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

CONTRACT_VERSION = "runstore/v1"
MANIFEST_NAMES = frozenset({"run.json", "inventory.json"})
KINDS = frozenset({"adhoc", "campaign", "legacy"})
STATUSES = frozenset({"planned", "running", "complete", "failed", "legacy"})
ROLES = frozenset({"state", "derived", "log"})
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ContractError(ValueError):
    """Raised when a runstore document or payload violates the contract."""


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{label} must be an object")
    return value


def _require_string(value: Any, label: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise ContractError(f"{label} must be a non-empty string")
    return value


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise ContractError(f"{label} must be a lowercase SHA-256")
    return value


def _require_utc_timestamp(value: Any, label: str) -> str:
    text = _require_string(value, label)
    if not text.endswith("Z"):
        raise ContractError(f"{label} must be an RFC 3339 UTC timestamp ending in Z")
    try:
        datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError as exc:
        raise ContractError(f"{label} is not a valid timestamp: {text!r}") from exc
    return text


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise ContractError(f"missing contract file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ContractError(f"invalid JSON in {path}: {exc}") from exc
    return _require_mapping(value, str(path))


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    """Write one JSON document without exposing a partial destination file."""
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


def validate_run_manifest(value: dict[str, Any]) -> dict[str, Any]:
    if value.get("contract_version") != CONTRACT_VERSION:
        raise ContractError(f"run.json contract_version must be {CONTRACT_VERSION!r}")
    _require_string(value.get("run_id"), "run.json run_id")
    if value.get("kind") not in KINDS:
        raise ContractError(f"run.json kind must be one of {sorted(KINDS)}")
    if value.get("status") not in STATUSES:
        raise ContractError(f"run.json status must be one of {sorted(STATUSES)}")
    _require_utc_timestamp(value.get("created_at_utc"), "run.json created_at_utc")

    source = _require_mapping(value.get("source"), "run.json source")
    commit = source.get("git_commit")
    if commit is not None and (
        not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit)
    ):
        raise ContractError(
            "run.json source.git_commit must be 40 lowercase hex chars or null"
        )
    if source.get("git_clean") is not None and not isinstance(
        source.get("git_clean"), bool
    ):
        raise ContractError("run.json source.git_clean must be boolean or null")
    lockfile = source.get("lockfile")
    if lockfile is not None:
        lock = _require_mapping(lockfile, "run.json source.lockfile")
        validate_relative_path(lock.get("path"), "run.json source.lockfile.path")
        _require_sha256(lock.get("sha256"), "run.json source.lockfile.sha256")

    execution = _require_mapping(value.get("execution"), "run.json execution")
    experiment = execution.get("experiment")
    collection = execution.get("collection")
    for item, label in ((experiment, "experiment"), (collection, "collection")):
        if item is not None:
            _require_string(item, f"run.json execution.{label}")
    if experiment is None and collection is None:
        raise ContractError("run.json execution needs an experiment or collection")
    command = execution.get("command")
    if (
        not isinstance(command, list)
        or not command
        or not all(isinstance(part, str) and part for part in command)
    ):
        raise ContractError(
            "run.json execution.command must be a non-empty string array"
        )

    if not isinstance(value.get("upstream"), list):
        raise ContractError("run.json upstream must be an array")
    archive = value.get("archive")
    if archive is not None:
        archive_obj = _require_mapping(archive, "run.json archive")
        _require_string(archive_obj.get("archive_id"), "run.json archive.archive_id")
        uri = _require_string(archive_obj.get("uri"), "run.json archive.uri")
        if not uri.startswith(("r2://", "file://")):
            raise ContractError("run.json archive.uri must start with r2:// or file://")
    notes = value.get("provenance_notes")
    if not isinstance(notes, str):
        raise ContractError("run.json provenance_notes must be a string")
    return value


def validate_relative_path(value: Any, label: str = "path") -> str:
    text = _require_string(value, label)
    if "\\" in text:
        raise ContractError(f"{label} must use POSIX separators")
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != text:
        raise ContractError(f"{label} must be a normalized relative POSIX path")
    if text in MANIFEST_NAMES:
        raise ContractError(f"{label} cannot inventory {text}")
    return text


def canonical_payload_digest(files: list[dict[str, Any]]) -> str:
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def validate_inventory(value: dict[str, Any]) -> dict[str, Any]:
    if value.get("contract_version") != CONTRACT_VERSION:
        raise ContractError(
            f"inventory.json contract_version must be {CONTRACT_VERSION!r}"
        )
    _require_string(value.get("run_id"), "inventory.json run_id")
    _require_utc_timestamp(
        value.get("generated_at_utc"), "inventory.json generated_at_utc"
    )
    files = value.get("files")
    if not isinstance(files, list):
        raise ContractError("inventory.json files must be an array")

    paths: list[str] = []
    for index, raw in enumerate(files):
        row = _require_mapping(raw, f"inventory.json files[{index}]")
        path = validate_relative_path(row.get("path"), f"files[{index}].path")
        paths.append(path)
        size = row.get("size_bytes")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ContractError(
                f"files[{index}].size_bytes must be a non-negative integer"
            )
        _require_sha256(row.get("sha256"), f"files[{index}].sha256")
        if row.get("role") not in ROLES:
            raise ContractError(f"files[{index}].role must be one of {sorted(ROLES)}")
        if set(row) != {"path", "size_bytes", "sha256", "role"}:
            raise ContractError(
                f"files[{index}] must contain only path, size_bytes, sha256, role"
            )

    if paths != sorted(paths):
        raise ContractError("inventory.json files must be sorted by path")
    if len(paths) != len(set(paths)):
        raise ContractError("inventory.json contains duplicate paths")
    if value.get("file_count") != len(files):
        raise ContractError("inventory.json file_count does not match files")
    total = sum(row["size_bytes"] for row in files)
    if value.get("total_size_bytes") != total:
        raise ContractError("inventory.json total_size_bytes does not match files")
    _require_sha256(value.get("payload_digest"), "inventory.json payload_digest")
    if value["payload_digest"] != canonical_payload_digest(files):
        raise ContractError("inventory.json payload_digest does not match files")
    return value


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _role_for(relative: Path) -> str:
    first = relative.parts[0] if relative.parts else ""
    if first == "derived":
        return "derived"
    if first == "logs" or relative.suffix == ".log":
        return "log"
    return "state"


def inventory_payload(
    root: Path, *, run_id: str, generated_at_utc: str | None = None
) -> dict[str, Any]:
    root = root.resolve()
    if not root.is_dir():
        raise ContractError(f"run root is not a directory: {root}")
    files: list[dict[str, Any]] = []
    for item in sorted(
        root.rglob("*"), key=lambda path: path.relative_to(root).as_posix()
    ):
        if item.is_symlink():
            raise ContractError(f"payload symlinks are not supported in v1: {item}")
        if not item.is_file() or item.relative_to(root).as_posix() in MANIFEST_NAMES:
            continue
        relative = item.relative_to(root)
        files.append(
            {
                "path": relative.as_posix(),
                "size_bytes": item.stat().st_size,
                "sha256": _hash_file(item),
                "role": _role_for(relative),
            }
        )
    timestamp = generated_at_utc or datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    ).replace("+00:00", "Z")
    result = {
        "contract_version": CONTRACT_VERSION,
        "run_id": run_id,
        "generated_at_utc": timestamp,
        "file_count": len(files),
        "total_size_bytes": sum(row["size_bytes"] for row in files),
        "payload_digest": canonical_payload_digest(files),
        "files": files,
    }
    return validate_inventory(result)


def verify_payload(root: Path, inventory: dict[str, Any]) -> None:
    expected = validate_inventory(inventory)
    actual = inventory_payload(
        root,
        run_id=expected["run_id"],
        generated_at_utc=expected["generated_at_utc"],
    )
    for field in ("file_count", "total_size_bytes", "payload_digest", "files"):
        if actual[field] != expected[field]:
            raise ContractError(
                f"payload does not match inventory.json: {field} differs"
            )


def provenance_gaps(run: dict[str, Any] | None) -> list[str]:
    if run is None:
        return ["missing run.json"]
    source = run["source"]
    gaps = []
    if source.get("git_commit") is None:
        gaps.append("unknown generating Git commit")
    if source.get("git_clean") is None:
        gaps.append("unknown clean-tree state")
    if source.get("lockfile") is None:
        gaps.append("unknown lockfile identity")
    return gaps
