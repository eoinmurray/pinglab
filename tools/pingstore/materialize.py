"""Materialize a selected dataset into an isolated Demolab artifact view."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from .catalogue import Catalogue
from .contracts import PingstoreError, canonical_digest, load_json, write_json_atomic

RAW_DATA_SUFFIXES = {".h5", ".hdf5", ".npy", ".npz", ".pt", ".pth"}


def _publication_ignore(_directory: str, names: list[str]) -> set[str]:
    """Keep raw numerical state in ExperimentRuns, outside PublicationView."""
    return {name for name in names if Path(name).suffix.lower() in RAW_DATA_SUFFIXES}


def _copy_publication_tree(source: Path, destination: Path) -> None:
    shutil.copytree(source, destination, ignore=_publication_ignore)


def _official_run(catalogue: Catalogue, experiment: str) -> tuple[dict, dict]:
    matches: list[tuple[dict, str]] = []
    for path in sorted(
        (catalogue.root / "collections").glob("*/collection-dataset.json")
    ):
        dataset = catalogue.load_dataset(path.parent.name)
        selected = dataset["official_runs"].get(experiment)
        if selected:
            matches.append((dataset, selected))
    if len(matches) != 1:
        raise PingstoreError(
            f"expected one official run for {experiment}; found {len(matches)}"
        )
    dataset, run_id = matches[0]
    run_root = catalogue.run_path(dataset["collection"], experiment, run_id)
    return dataset, load_json(run_root / "run.json")


def materialize_experiment(
    catalogue: Catalogue, experiment: str, artifacts_root: Path
) -> dict[str, Any]:
    """Atomically refresh one experiment from its official immutable run."""
    dataset, run = _official_run(catalogue, experiment)
    source = Path(run["payload"]["location"])
    if not source.is_dir():
        raise PingstoreError(f"official payload is unavailable: {source}")
    destination = artifacts_root / experiment
    staging = artifacts_root / f".{experiment}.pingstore-staging"
    previous = artifacts_root / f".{experiment}.pingstore-previous"
    if staging.exists() or previous.exists():
        raise PingstoreError(f"unfinished materialization exists for {experiment}")
    artifacts_root.mkdir(parents=True, exist_ok=True)
    _copy_publication_tree(source, staging)
    try:
        if destination.exists():
            os.rename(destination, previous)
        os.rename(staging, destination)
        if previous.exists():
            shutil.rmtree(previous)
    except BaseException:
        if destination.exists() and previous.exists():
            shutil.rmtree(destination)
        if previous.exists():
            os.rename(previous, destination)
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return {
        "schema": "pingstore.materialized-experiment/v1",
        "dataset_id": dataset["dataset_id"],
        "experiment": experiment,
        "run_id": run["run_id"],
        "inventory_digest": run["payload"]["inventory_digest"],
    }


def materialize_publication_view(
    catalogue: Catalogue, destination: Path, *, activate: bool = False
) -> dict[str, Any]:
    """Materialize every official selection into one Demolab publication view."""
    selections: dict[str, dict[str, str]] = {}
    historical_records: dict[str, dict[str, str]] = {}
    for path in sorted(
        (catalogue.root / "collections").glob("*/collection-dataset.json")
    ):
        dataset = catalogue.load_dataset(path.parent.name)
        for experiment, run_id in sorted(dataset["official_runs"].items()):
            if experiment in selections:
                raise PingstoreError(f"multiple collections select {experiment}")
            selections[experiment] = {
                "collection": dataset["collection"],
                "dataset_id": dataset["dataset_id"],
                "run_id": run_id,
            }
    staging = destination.with_name(destination.name + ".pingstore-staging")
    if staging.exists():
        raise PingstoreError(f"unfinished publication staging exists: {staging}")
    staging.mkdir(parents=True)
    for experiment, selected in sorted(selections.items()):
        run_root = catalogue.run_path(
            selected["collection"], experiment, selected["run_id"]
        )
        run = load_json(run_root / "run.json")
        source = Path(run["payload"]["location"])
        if not source.is_dir():
            raise PingstoreError(f"selected payload is unavailable: {source}")
        _copy_publication_tree(source, staging / experiment)
    for path in sorted((catalogue.root / "historical").glob("*/record.json")):
        record = load_json(path)
        experiment = record["experiment"]
        if experiment in selections:
            raise PingstoreError(
                f"historical and active evidence overlap for {experiment}"
            )
        source = Path(record["payload"]["location"])
        if not source.is_dir():
            raise PingstoreError(f"historical payload is unavailable: {source}")
        _copy_publication_tree(source, staging / experiment)
        historical_records[experiment] = {
            "collection": record.get("collection"),
            "inventory_digest": record["payload"]["inventory_digest"],
        }
    # Preserve non-experiment publication assets; experiment payload ownership
    # remains exclusively with the selected datasets.
    if destination.is_dir():
        for child in destination.iterdir():
            if (
                not child.name.startswith("exp")
                and child.name != ".pingstore-view.json"
            ):
                target = staging / child.name
                if child.is_dir():
                    shutil.copytree(child, target)
                else:
                    shutil.copy2(child, target)
    view = {
        "schema": "pingstore.materialized-view/v1",
        "selections": selections,
        "historical_records": historical_records,
    }
    view["digest"] = canonical_digest(view)
    write_json_atomic(staging / ".pingstore-view.json", view)
    if not activate:
        return {**view, "staging": str(staging), "active": False}
    previous = destination.with_name(destination.name + ".pingstore-previous")
    if previous.exists():
        raise PingstoreError(f"unfinished publication backup exists: {previous}")
    try:
        if destination.exists():
            os.rename(destination, previous)
        os.rename(staging, destination)
        if previous.exists():
            shutil.rmtree(previous)
    except BaseException:
        if destination.exists() and previous.exists():
            shutil.rmtree(destination)
        if previous.exists():
            os.rename(previous, destination)
        raise
    return {**view, "destination": str(destination), "active": True}


def materialize_shadow(
    catalogue: Catalogue,
    collection: str,
    destination: Path,
    *,
    use_preview: bool = True,
    use_proposal: bool = False,
) -> dict[str, Any]:
    dataset = catalogue.load_dataset(collection)
    selections = dict(dataset["official_runs"])
    if use_proposal:
        selections.update(dataset.get("selection_proposal", {}))
    if use_preview:
        selections.update(dataset["preview_overrides"])
    staging = destination.with_name(destination.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    files: list[str] = []
    for experiment, run_id in sorted(selections.items()):
        run_root = catalogue.run_path(collection, experiment, run_id)
        run = load_json(run_root / "run.json")
        source = Path(run["payload"]["location"])
        if not source.is_dir():
            raise PingstoreError(f"selected payload is unavailable: {source}")
        target = staging / experiment
        shutil.copytree(source, target)
        files.extend(
            path.relative_to(staging).as_posix()
            for path in target.rglob("*")
            if path.is_file()
        )
    view = {
        "schema": "pingstore.materialized-view/v1",
        "dataset_id": dataset["dataset_id"],
        "collection": collection,
        "selections": selections,
        "proposal": use_proposal,
        "files": sorted(files),
    }
    view["digest"] = canonical_digest(view)
    write_json_atomic(staging / ".pingstore-view.json", view)
    if destination.exists():
        shutil.rmtree(staging)
        raise PingstoreError(f"shadow destination already exists: {destination}")
    os.rename(staging, destination)
    return view


def cutover(*_args: object, **_kwargs: object) -> None:
    raise PingstoreError(
        "cutover requires a separately reviewed implementation invocation; "
        "this PR provides shadow materialization only"
    )
