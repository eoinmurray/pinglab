"""Small contracts for Pingstore's flat filesystem convention."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

RUN_SCHEMA = "pingstore.run/v4"
EXPERIMENT_RE = re.compile(r"^exp[0-9]{3}$")
STAGE_ID_RE = re.compile(r"^(exp[0-9]{3})-r([0-9]{3,})-(compute|analyse|present)$")
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
        raise PingstoreError(f"operational run schema must be {RUN_SCHEMA}")
    run_id = value.get("run_id")
    experiment = value.get("experiment")
    if not isinstance(run_id, str) or not STAGE_ID_RE.fullmatch(run_id):
        raise PingstoreError("run_id must encode experiment, counter and stage")
    if not isinstance(experiment, str) or not EXPERIMENT_RE.fullmatch(experiment):
        raise PingstoreError("experiment must be expNNN")
    if not run_id.startswith(experiment + "-"):
        raise PingstoreError("run_id must begin with experiment-")
    stage = value.get("stage")
    match = STAGE_ID_RE.fullmatch(run_id)
    if stage not in ("compute", "analyse", "present"):
        raise PingstoreError("stage must be compute, analyse or present")
    if match is None or match.group(1) != experiment or match.group(3) != stage:
        raise PingstoreError("run_id must encode experiment, counter and stage")
    if not isinstance(value.get("inputs"), dict):
        raise PingstoreError("runs require explicit inputs (empty for new compute)")
    for role, reference in value["inputs"].items():
        if not isinstance(role, str) or not role or not isinstance(reference, dict):
            raise PingstoreError("invalid input role/reference")
        if set(reference) != {"run_id", "payload_digest"}:
            raise PingstoreError("input pins contain only run_id and payload_digest")
        if not isinstance(reference.get("run_id"), str) or not STAGE_ID_RE.fullmatch(reference["run_id"]):
            raise PingstoreError("input must name a completed v4 run")
        if reference["run_id"] == run_id:
            raise PingstoreError("run cannot be its own input")
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", str(reference.get("payload_digest", ""))):
            raise PingstoreError("input requires a payload checksum")
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
            isinstance(run_id, str) and STAGE_ID_RE.fullmatch(run_id)
            for run_id in run_ids
        ):
            raise PingstoreError(f"collection view {name!r} must be a run-ID array")
        if len(run_ids) != len(set(run_ids)):
            raise PingstoreError(f"collection view {name!r} contains duplicate runs")
    return value


def run_root(root: Path, run_id: str) -> Path:
    if not STAGE_ID_RE.fullmatch(run_id):
        raise PingstoreError(f"invalid run ID: {run_id}")
    return root / "runs" / run_id


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def payload_inventory(directory: Path) -> list[dict[str, Any]]:
    """Inventory immutable scientific bytes beneath export/."""
    root = directory / "export"
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise PingstoreError(f"unsupported payload entry: {path}")
        relative = path.relative_to(directory).as_posix()
        if path.is_file():
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
    manifest = directory / "run.json"
    if manifest.is_symlink() or not manifest.is_file():
        raise PingstoreError(f"run.json must be a regular file: {manifest}")
    run = validate_run(load_json(manifest))
    names = {p.name for p in directory.iterdir()}
    if names != {"run.json", "README.md", "export"}:
        raise PingstoreError(
            "v4 run must contain exactly run.json, README.md and export/"
        )
    directories = {"export"}
    flat = directory / "export" if run["stage"] == "present" else None
    for name in names:
        path = directory / name
        if path.is_symlink() or (path.is_dir() != (name in directories)):
            raise PingstoreError(f"invalid run entry: {path}")
        if name in {"run.json", "README.md"} and not path.is_file():
            raise PingstoreError(f"run entry must be a regular file: {path}")
    if flat is not None:
        for path in flat.iterdir():
            if path.is_symlink() or not path.is_file():
                raise PingstoreError(f"presentation must be flat regular files: {path}")
    else:
        export = directory / "export"
        from .layout import canonical_role_name

        for path in export.rglob("*"):
            relative = path.relative_to(export)
            if len(relative.parts) > 2:
                raise PingstoreError(
                    f"scientific export exceeds one unit-directory level: {path}"
                )
            if path.is_dir() and len(relative.parts) != 1:
                raise PingstoreError(f"invalid scientific unit directory: {path}")
            if path.is_file() and canonical_role_name(path.name) != path.name:
                raise PingstoreError(f"noncanonical scientific role filename: {path}")
        for unit in (path for path in export.iterdir() if path.is_dir()):
            files = [path for path in unit.iterdir() if path.is_file()]
            if len(files) < 2:
                raise PingstoreError(
                    f"scientific unit directories require at least two files: {unit}"
                )
    for name in directories & names:
        for path in (directory / name).rglob("*"):
            if path.is_symlink() or not (path.is_file() or path.is_dir()):
                raise PingstoreError(f"unsupported payload entry: {path}")


def validate_run_directory(directory: Path) -> dict[str, Any]:
    """Validate structure, identity and checksums before reading or publishing."""
    validate_layout(directory)
    run = validate_run(load_json(directory / "run.json"))
    if directory.name not in {run["run_id"], f".{run['run_id']}.tmp"}:
        raise PingstoreError("run directory and run.json identity differ")
    if "data_root" in run:
        raise PingstoreError("data_root is obsolete; use export_root beneath export/")
    if "export_root" in run:
        raise PingstoreError("export_root is obsolete; unit directories live directly under export/")
    if payload_digest(directory) != run["payload_digest"]:
        raise PingstoreError(f"payload checksum mismatch: {directory}")
    return run


def validate_operational_run_directory(directory: Path) -> dict[str, Any]:
    """Validate one operational v4 run before consuming evidence."""
    if any(path.is_symlink() for path in (directory, *directory.parents)):
        raise PingstoreError("operational input paths must not use symlinks")
    return validate_run_directory(directory)
