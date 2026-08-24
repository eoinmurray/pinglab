"""Versioned Pingstore contracts and deterministic JSON helpers."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

COLLECTION_DATASET_SCHEMA = "pingstore.collection-dataset/v1"
EXPERIMENT_RUN_SCHEMA = "pingstore.experiment-run/v1"
DATASET_STATUSES = frozenset({"working", "frozen", "verified", "published"})
RUN_STATUSES = frozenset({"planned", "running", "complete", "finalized", "failed"})
RUN_DISPOSITIONS = frozenset({"temporary", "candidate", "retained"})
SHA256_RE = re.compile(r"^(?:sha256:)?[0-9a-f]{64}$")
SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")


class PingstoreError(ValueError):
    """Raised when Pingstore state violates its contract."""


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


def canonical_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(encoded.encode()).hexdigest()


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PingstoreError(f"{label} must be an object")
    return value


def _strings(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise PingstoreError(f"{label} must be an array of non-empty strings")
    if len(value) != len(set(value)):
        raise PingstoreError(f"{label} must not contain duplicates")
    return value


def _slug(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SLUG_RE.fullmatch(value):
        raise PingstoreError(f"{label} must be a lowercase slug")
    return value


def _relative(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise PingstoreError(f"{label} must be a relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise PingstoreError(f"{label} must be a normalized relative POSIX path")
    return value


def validate_collection_dataset(value: dict[str, Any]) -> dict[str, Any]:
    if value.get("schema") != COLLECTION_DATASET_SCHEMA:
        raise PingstoreError(f"dataset schema must be {COLLECTION_DATASET_SCHEMA}")
    dataset_id = value.get("dataset_id")
    if not isinstance(dataset_id, str) or "/" not in dataset_id:
        raise PingstoreError("dataset_id must be collection-qualified")
    collection = _slug(value.get("collection"), "collection")
    if not dataset_id.startswith(collection + "/"):
        raise PingstoreError("dataset_id must begin with collection/")
    if value.get("status") not in DATASET_STATUSES:
        raise PingstoreError(f"dataset status must be one of {sorted(DATASET_STATUSES)}")
    experiments = _strings(value.get("experiments"), "experiments")
    for experiment in experiments:
        _slug(experiment, "experiment")
    runs = _mapping(value.get("runs"), "runs")
    official = _mapping(value.get("official_runs"), "official_runs")
    preview = _mapping(value.get("preview_overrides"), "preview_overrides")
    experiment_set = set(experiments)
    if set(runs) != experiment_set:
        raise PingstoreError("runs must contain exactly the registered experiments")
    if not set(official).issubset(experiment_set):
        raise PingstoreError("official_runs names an unregistered experiment")
    if not set(preview).issubset(experiment_set):
        raise PingstoreError("preview_overrides names an unregistered experiment")
    for experiment, run_ids in runs.items():
        values = _strings(run_ids, f"runs.{experiment}")
        for selected_name, selected in (
            ("official_runs", official.get(experiment)),
            ("preview_overrides", preview.get(experiment)),
        ):
            if selected is not None and selected not in values:
                raise PingstoreError(
                    f"{selected_name}.{experiment} must select a retained run"
                )
    _strings(value.get("collection_assets"), "collection_assets")
    _strings(value.get("upstream_datasets"), "upstream_datasets")
    digest = value.get("digest")
    if value["status"] == "working" and digest is not None:
        raise PingstoreError("working datasets must not carry an immutable digest")
    if value["status"] != "working" and not (
        isinstance(digest, str) and SHA256_RE.fullmatch(digest)
    ):
        raise PingstoreError("frozen or later datasets require a SHA-256 digest")
    migration = value.get("migration")
    if migration is not None:
        _mapping(migration, "migration")
    return value


def validate_experiment_run(value: dict[str, Any]) -> dict[str, Any]:
    if value.get("schema") != EXPERIMENT_RUN_SCHEMA:
        raise PingstoreError(f"run schema must be {EXPERIMENT_RUN_SCHEMA}")
    run_id = value.get("run_id")
    if not isinstance(run_id, str) or "/" not in run_id:
        raise PingstoreError("run_id must be globally collection-qualified")
    _slug(value.get("collection"), "collection")
    _slug(value.get("experiment"), "experiment")
    if value.get("status") not in RUN_STATUSES:
        raise PingstoreError(f"run status must be one of {sorted(RUN_STATUSES)}")
    if value.get("disposition") not in RUN_DISPOSITIONS:
        raise PingstoreError(
            f"run disposition must be one of {sorted(RUN_DISPOSITIONS)}"
        )
    _mapping(value.get("source"), "source")
    execution = _mapping(value.get("execution"), "execution")
    command = execution.get("command")
    if not isinstance(command, list) or not all(isinstance(x, str) for x in command):
        raise PingstoreError("execution.command must be a string array")
    _strings(value.get("upstream_runs"), "upstream_runs")
    _strings(value.get("upstream_datasets"), "upstream_datasets")
    payload = _mapping(value.get("payload"), "payload")
    location = payload.get("location")
    if not isinstance(location, str) or not location:
        raise PingstoreError("payload.location must be non-empty")
    digest = payload.get("inventory_digest")
    if value["status"] == "finalized" and not (
        isinstance(digest, str) and SHA256_RE.fullmatch(digest)
    ):
        raise PingstoreError("finalized runs require an inventory digest")
    archive = value.get("archive")
    if archive is not None:
        _mapping(archive, "archive")
    legacy = value.get("legacy_identity")
    if legacy is not None:
        _mapping(legacy, "legacy_identity")
    return value


def relative_payload_path(root: Path, path: Path) -> str:
    return _relative(path.resolve().relative_to(root.resolve()).as_posix(), "path")
