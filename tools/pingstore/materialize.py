"""Materialize a selected dataset into an isolated Demolab artifact view."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from .catalogue import Catalogue
from .contracts import PingstoreError, canonical_digest, load_json, write_json_atomic


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
