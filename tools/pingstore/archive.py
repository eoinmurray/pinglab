"""Portable local archives for frozen CollectionDatasets."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from .catalogue import Catalogue
from .contracts import (
    PingstoreError,
    canonical_digest,
    load_json,
    validate_collection_dataset,
    validate_experiment_run,
    write_json_atomic,
)
from .payload import inventory_payload, verify_payload


def archive_dataset(
    catalogue: Catalogue, dataset_id: str, destination: Path
) -> dict[str, Any]:
    """Create an immutable, portable archive without changing source payloads."""
    dataset_file = catalogue.frozen_path(dataset_id)
    dataset = validate_collection_dataset(load_json(dataset_file))
    if dataset["status"] not in {"frozen", "verified", "published"}:
        raise PingstoreError("only frozen or later datasets can be archived")
    if destination.exists():
        raise PingstoreError(f"archive destination already exists: {destination}")
    staging = destination.with_name(destination.name + ".staging")
    if staging.exists():
        raise PingstoreError(f"archive staging path already exists: {staging}")
    staging.mkdir(parents=True)
    archived_runs: list[str] = []
    try:
        write_json_atomic(staging / "collection-dataset.json", dataset)
        for experiment, run_ids in sorted(dataset["runs"].items()):
            for run_id in run_ids:
                source_root = catalogue.run_path(
                    dataset["collection"], experiment, run_id
                )
                run = validate_experiment_run(load_json(source_root / "run.json"))
                payload_source = Path(run["payload"]["location"])
                if not payload_source.is_dir():
                    raise PingstoreError(
                        f"run payload is unavailable: {payload_source}"
                    )
                safe = run_id.replace("/", "--")
                target_root = staging / "experiment-runs" / experiment / safe
                payload_target = target_root / "payload"
                payload_target.parent.mkdir(parents=True)
                shutil.copytree(payload_source, payload_target)
                inventory = inventory_payload(payload_source, run_id=run_id)
                expected = run["payload"].get("inventory_digest")
                actual = "sha256:" + inventory["payload_digest"]
                if expected != actual:
                    raise PingstoreError(
                        f"run payload inventory drift for {run_id}: {expected} != {actual}"
                    )
                write_json_atomic(target_root / "inventory.json", inventory)
                authored_source = source_root / "authored-sources"
                if authored_source.is_dir():
                    shutil.copytree(authored_source, target_root / "authored-sources")
                portable = dict(run)
                portable["payload"] = dict(run["payload"])
                portable["payload"]["location"] = (
                    f"experiment-runs/{experiment}/{safe}/payload"
                )
                portable["payload"]["location_base"] = "archive"
                write_json_atomic(target_root / "run.json", portable)
                archived_runs.append(run_id)
        manifest = {
            "schema": "pingstore.dataset-archive/v1",
            "dataset_id": dataset_id,
            "dataset_digest": dataset["digest"],
            "runs": sorted(archived_runs),
        }
        manifest["digest"] = canonical_digest(manifest)
        write_json_atomic(staging / "archive.json", manifest)
        os.rename(staging, destination)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def restore_dataset(source: Path, destination_root: Path) -> dict[str, Any]:
    """Restore a native archive into a clean Pingstore root."""
    manifest = load_json(source / "archive.json")
    expected = manifest.get("digest")
    unsigned = dict(manifest)
    unsigned.pop("digest", None)
    if expected != canonical_digest(unsigned):
        raise PingstoreError("dataset archive manifest digest does not match")
    dataset = validate_collection_dataset(load_json(source / "collection-dataset.json"))
    if dataset["dataset_id"] != manifest.get("dataset_id"):
        raise PingstoreError("dataset archive identity does not match")
    catalogue = Catalogue(destination_root)
    frozen_target = catalogue.frozen_path(dataset["dataset_id"])
    if frozen_target.exists():
        raise PingstoreError(f"frozen dataset already exists: {frozen_target}")
    for experiment, run_ids in sorted(dataset["runs"].items()):
        for run_id in run_ids:
            safe = run_id.replace("/", "--")
            archived_root = source / "experiment-runs" / experiment / safe
            run = validate_experiment_run(load_json(archived_root / "run.json"))
            inventory = load_json(archived_root / "inventory.json")
            verify_payload(archived_root / "payload", inventory)
            if (
                run["payload"]["inventory_digest"]
                != "sha256:" + inventory["payload_digest"]
            ):
                raise PingstoreError(f"archived payload identity drift for {run_id}")
            target_root = catalogue.run_path(dataset["collection"], experiment, run_id)
            if target_root.exists():
                raise PingstoreError(f"run already exists: {run_id}")
            shutil.copytree(archived_root, target_root)
            restored = dict(run)
            restored["payload"] = dict(run["payload"])
            restored["payload"]["location"] = str(target_root / "payload")
            restored["payload"].pop("location_base", None)
            write_json_atomic(target_root / "run.json", restored)
            verify_payload(target_root / "payload", inventory)
    frozen_target.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(frozen_target, dataset)
    return manifest
