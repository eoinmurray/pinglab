"""Small contracts for Pingstore's flat filesystem convention."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

RUN_SCHEMA = "pingstore.run/v2"
EXPERIMENT_RE = re.compile(r"^exp[0-9]{3}$")
RUN_ID_RE = re.compile(r"^exp[0-9]{3}-[a-z0-9][a-z0-9.-]*$")
VIEW_RE = re.compile(r"^[a-z0-9][a-z0-9./-]*$")


class PingstoreError(ValueError):
    """Raised when the flat store violates its filesystem contract."""


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise PingstoreError(f"missing Pingstore document: {path}") from exc
    except json.JSONDecodeError as exc:
        raise PingstoreError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PingstoreError(f"{path} must contain a JSON object")
    return value


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_run(value: dict[str, Any]) -> dict[str, Any]:
    if value.get("schema") != RUN_SCHEMA:
        raise PingstoreError(f"run schema must be {RUN_SCHEMA}")
    run_id = value.get("run_id")
    experiment = value.get("experiment")
    if not isinstance(run_id, str) or not RUN_ID_RE.fullmatch(run_id):
        raise PingstoreError("run_id must encode expNNN and a safe run identity")
    if not isinstance(experiment, str) or not EXPERIMENT_RE.fullmatch(experiment):
        raise PingstoreError("experiment must be expNNN")
    if not run_id.startswith(experiment + "-"):
        raise PingstoreError("run_id must begin with experiment-")
    for key in ("collection", "origin", "created_at"):
        if not isinstance(value.get(key), str) or not value[key]:
            raise PingstoreError(f"{key} must be a non-empty string")
    if not isinstance(value.get("execution"), dict):
        raise PingstoreError("execution must be an object")
    if not isinstance(value.get("provenance"), dict):
        raise PingstoreError("provenance must be an object")
    digest = value.get("payload_digest")
    if not isinstance(digest, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
        raise PingstoreError("payload_digest must be a prefixed SHA-256")
    return value


def validate_collections(value: dict[str, Any]) -> dict[str, Any]:
    for name, run_ids in value.items():
        if not isinstance(name, str) or not VIEW_RE.fullmatch(name):
            raise PingstoreError(f"invalid collection view name: {name!r}")
        if not isinstance(run_ids, list) or not all(
            isinstance(run_id, str) and RUN_ID_RE.fullmatch(run_id)
            for run_id in run_ids
        ):
            raise PingstoreError(f"collection view {name!r} must be a run-ID array")
        if len(run_ids) != len(set(run_ids)):
            raise PingstoreError(f"collection view {name!r} contains duplicate runs")
    return value


def run_root(root: Path, run_id: str) -> Path:
    if not RUN_ID_RE.fullmatch(run_id):
        raise PingstoreError(f"invalid run ID: {run_id}")
    return root / "runs" / run_id


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def payload_inventory(directory: Path) -> list[dict[str, Any]]:
    """Inventory all payload bytes, including nested manifests; exclude run.json."""
    rows = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise PingstoreError(f"unsupported payload entry: {path}")
        relative = path.relative_to(directory).as_posix()
        if path.is_file() and relative != "run.json":
            rows.append(
                {
                    "path": relative,
                    "size_bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
            )
    return rows


def payload_digest(directory: Path) -> str:
    encoded = json.dumps(
        payload_inventory(directory), sort_keys=True, separators=(",", ":")
    ).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def validate_layout(directory: Path) -> None:
    if directory.is_symlink() or not directory.is_dir():
        raise PingstoreError(f"run must be a real directory: {directory}")
    expected = {"run.json", "README.md", "export", "presentation"}
    if {p.name for p in directory.iterdir()} != expected:
        raise PingstoreError(
            "run must contain exactly run.json, README.md, export/, presentation/"
        )
    for name in expected:
        path = directory / name
        if path.is_symlink() or (path.is_dir() != (name in {"export", "presentation"})):
            raise PingstoreError(f"invalid run entry: {path}")
        if name in {"run.json", "README.md"} and not path.is_file():
            raise PingstoreError(f"run entry must be a regular file: {path}")
    for path in (directory / "presentation").iterdir():
        if path.is_symlink() or not path.is_file():
            raise PingstoreError(f"presentation must be flat regular files: {path}")
    for path in (directory / "export").rglob("*"):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise PingstoreError(f"unsupported export entry: {path}")


def validate_run_directory(directory: Path) -> dict[str, Any]:
    """Validate structure, identity and checksums before reading or publishing."""
    validate_layout(directory)
    run = validate_run(load_json(directory / "run.json"))
    if directory.name not in {run["run_id"], f".{run['run_id']}.tmp"}:
        raise PingstoreError("run directory and run.json identity differ")
    if "data_root" in run:
        raise PingstoreError("data_root is obsolete; use export_root beneath export/")
    if "export_root" in run:
        relative = run["export_root"]
        if not isinstance(relative, str):
            raise PingstoreError("export_root must be a relative path beneath export/")
        path = Path(relative)
        if (path.is_absolute() or not path.parts or path.parts[0] != "export"
                or ".." in path.parts or not (directory / path).is_dir()):
            raise PingstoreError("export_root must name a directory beneath export/")
    if payload_digest(directory) != run["payload_digest"]:
        raise PingstoreError(f"payload checksum mismatch: {directory}")
    return run
