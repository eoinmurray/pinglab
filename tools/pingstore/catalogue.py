"""Mutable working catalogues and immutable dataset snapshots."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from .contracts import (
    COLLECTION_DATASET_SCHEMA,
    PingstoreError,
    canonical_digest,
    load_json,
    validate_collection_dataset,
    validate_experiment_run,
    write_json_atomic,
)


class Catalogue:
    def __init__(self, root: Path = Path("runs/pingstore")) -> None:
        self.root = root

    def dataset_path(self, collection: str) -> Path:
        return self.root / "collections" / collection / "collection-dataset.json"

    def frozen_path(self, dataset_id: str) -> Path:
        safe = dataset_id.replace("/", "--")
        return self.root / "frozen" / safe / "collection-dataset.json"

    def run_path(self, collection: str, experiment: str, run_id: str) -> Path:
        safe = run_id.replace("/", "--")
        return self.root / "experiment-runs" / collection / experiment / safe

    def load_dataset(self, collection: str) -> dict[str, Any]:
        return validate_collection_dataset(load_json(self.dataset_path(collection)))

    def save_dataset(self, dataset: dict[str, Any]) -> Path:
        validate_collection_dataset(dataset)
        path = self.dataset_path(dataset["collection"])
        write_json_atomic(path, dataset)
        return path

    def create_dataset(
        self, collection: str, experiments: list[str], *, migration: dict | None = None
    ) -> dict[str, Any]:
        path = self.dataset_path(collection)
        if path.exists():
            raise PingstoreError(f"working dataset already exists: {path}")
        dataset = {
            "schema": COLLECTION_DATASET_SCHEMA,
            "dataset_id": f"{collection}/working",
            "collection": collection,
            "status": "working",
            "experiments": sorted(experiments),
            "runs": {experiment: [] for experiment in sorted(experiments)},
            "official_runs": {},
            "preview_overrides": {},
            "collection_assets": [],
            "upstream_datasets": [],
            "migration": migration,
            "digest": None,
        }
        self.save_dataset(dataset)
        return dataset

    def register_run(self, run: dict[str, Any]) -> None:
        validate_experiment_run(run)
        dataset = self.load_dataset(run["collection"])
        experiment = run["experiment"]
        if experiment not in dataset["runs"]:
            raise PingstoreError(f"{experiment} is not registered in the collection")
        run_ids = dataset["runs"][experiment]
        if run["run_id"] not in run_ids:
            run_ids.append(run["run_id"])
            run_ids.sort()
            self.save_dataset(dataset)

    def register_experiment(self, collection: str, experiment: str) -> None:
        dataset = self.load_dataset(collection)
        if experiment in dataset["experiments"]:
            return
        dataset["experiments"].append(experiment)
        dataset["experiments"].sort()
        dataset["runs"][experiment] = []
        self.save_dataset(dataset)

    def select(self, experiment: str, run_id: str, *, preview: bool = False) -> None:
        matches: list[dict[str, Any]] = []
        for path in sorted((self.root / "collections").glob("*/collection-dataset.json")):
            dataset = validate_collection_dataset(load_json(path))
            if experiment in dataset["runs"]:
                matches.append(dataset)
        if len(matches) != 1:
            raise PingstoreError(
                f"expected one collection containing {experiment}; found {len(matches)}"
            )
        dataset = matches[0]
        if run_id not in dataset["runs"][experiment]:
            raise PingstoreError(f"{run_id} is not retained for {experiment}")
        key = "preview_overrides" if preview else "official_runs"
        dataset[key][experiment] = run_id
        self.save_dataset(dataset)

    def freeze(self, collection: str, snapshot: str) -> dict[str, Any]:
        dataset = self.load_dataset(collection)
        missing = sorted(set(dataset["experiments"]) - set(dataset["official_runs"]))
        if missing:
            raise PingstoreError(
                "cannot freeze without official runs for: " + ", ".join(missing)
            )
        frozen = copy.deepcopy(dataset)
        frozen["dataset_id"] = f"{collection}/{snapshot}"
        frozen["status"] = "frozen"
        frozen["preview_overrides"] = {}
        frozen["digest"] = None
        frozen["digest"] = canonical_digest(frozen)
        validate_collection_dataset(frozen)
        path = self.frozen_path(frozen["dataset_id"])
        if path.exists():
            existing = validate_collection_dataset(load_json(path))
            if existing != frozen:
                raise PingstoreError(f"frozen dataset already exists with drift: {path}")
            return existing
        write_json_atomic(path, frozen)
        return frozen
