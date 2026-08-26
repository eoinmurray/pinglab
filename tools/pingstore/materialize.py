"""Materialize flat runs or manually named views into publication artifacts."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from .contracts import (
    PingstoreError,
    load_json,
    run_root,
    validate_collections,
    validate_run,
)

RAW_DATA_SUFFIXES = {".h5", ".hdf5", ".npy", ".npz", ".pt", ".pth"}


def _ignore(directory: str, names: list[str]) -> set[str]:
    ignored = {name for name in names if Path(name).suffix.lower() in RAW_DATA_SUFFIXES}
    if Path(directory).name == "files":
        ignored.add("state")
    return ignored


def _replace_tree(source: Path, destination: Path) -> None:
    staging = destination.with_name("." + destination.name + ".pingstore-staging")
    previous = destination.with_name("." + destination.name + ".pingstore-previous")
    if staging.exists() or previous.exists():
        raise PingstoreError(
            f"unfinished materialization exists for {destination.name}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, staging, ignore=_ignore)
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
        shutil.rmtree(staging, ignore_errors=True)
        raise


def materialize_run(root: Path, run_id: str, artifacts_root: Path) -> dict:
    directory = run_root(root, run_id)
    run = validate_run(load_json(directory / "run.json"))
    files = directory / "files"
    if not files.is_dir():
        raise PingstoreError(f"run files are missing: {run_id}")
    _replace_tree(files, artifacts_root / run["experiment"])
    return {"run_id": run_id, "experiment": run["experiment"]}


def materialize_view(root: Path, name: str, destination: Path) -> dict:
    collections = validate_collections(load_json(root / "collections.json"))
    try:
        selected = collections[name]
    except KeyError as exc:
        raise PingstoreError(f"unknown collection view: {name}") from exc
    staging = destination.with_name(destination.name + ".pingstore-staging")
    if staging.exists() or destination.exists():
        raise PingstoreError(f"materialization destination must be new: {destination}")
    staging.mkdir(parents=True)
    experiments: set[str] = set()
    try:
        for run_id in selected:
            directory = run_root(root, run_id)
            run = validate_run(load_json(directory / "run.json"))
            experiment = run["experiment"]
            if experiment in experiments:
                raise PingstoreError(f"view selects multiple runs for {experiment}")
            experiments.add(experiment)
            shutil.copytree(directory / "files", staging / experiment, ignore=_ignore)
        os.rename(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {"view": name, "runs": list(selected), "destination": str(destination)}
