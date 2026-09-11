"""Experiment collection membership derived from writing metadata."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .contracts import PingstoreError

HISTORY_SCHEMA = "pinglab.experiment-history/v1"
EXPERIMENT_RE = re.compile(r"^exp[0-9]{3}$")
COLLECTION_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
WRITING_COLLECTION_RE = re.compile(r'collection:\s*"([a-z0-9-]+)"')


def history_path(repo: Path) -> Path:
    return repo / "experiments/history.json"


def load_history(repo: Path) -> dict[str, dict[str, str]]:
    path = history_path(repo)
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise PingstoreError(f"experiment history is missing: {path}") from exc
    if value.get("schema") != HISTORY_SCHEMA:
        raise PingstoreError(f"experiment history schema must be {HISTORY_SCHEMA}")
    historical = value.get("historical")
    if not isinstance(historical, dict):
        raise PingstoreError("experiment history must contain an object")
    for experiment, row in historical.items():
        if not EXPERIMENT_RE.fullmatch(experiment):
            raise PingstoreError(f"invalid historical experiment: {experiment}")
        if not isinstance(row, dict):
            raise PingstoreError(f"history for {experiment} must be an object")
        if not isinstance(row.get("disposition"), str) or not isinstance(
            row.get("evidence"), str
        ):
            raise PingstoreError(
                f"history for {experiment} requires disposition and evidence"
            )
        high_watermark = row.get("highest_allocated_counter")
        if high_watermark is not None and (
            not isinstance(high_watermark, int)
            or isinstance(high_watermark, bool)
            or high_watermark < 1
        ):
            raise PingstoreError(
                f"invalid historical counter for {experiment}: {high_watermark}"
            )
        collection = row.get("collection")
        if collection is not None and (
            not isinstance(collection, str)
            or not COLLECTION_RE.fullmatch(collection)
        ):
            raise PingstoreError(
                f"invalid historical collection for {experiment}: {collection}"
            )
    return historical


def memberships(repo: Path) -> dict[str, str]:
    """Return active experiment memberships declared by their writings."""
    result: dict[str, str] = {}
    for path in sorted((repo / "writings").glob("exp[0-9][0-9][0-9].typ")):
        matches = WRITING_COLLECTION_RE.findall(path.read_text(errors="replace"))
        if not matches:
            continue
        unique = set(matches)
        if len(unique) != 1:
            raise PingstoreError(
                f"writing declares conflicting collections: {path}"
            )
        experiment = path.stem
        collection = matches[0]
        if not EXPERIMENT_RE.fullmatch(experiment):
            raise PingstoreError(f"invalid experiment writing: {experiment}")
        if not COLLECTION_RE.fullmatch(collection):
            raise PingstoreError(
                f"invalid collection for {experiment}: {collection}"
            )
        result[experiment] = collection
    return result


def membership(repo: Path, experiment: str) -> str:
    """Return one active membership or fail with a useful contract error."""
    try:
        return memberships(repo)[experiment]
    except KeyError as exc:
        raise PingstoreError(
            f"active writing for {experiment} must declare collection metadata"
        ) from exc


def _runnable_experiments(repo: Path) -> set[str]:
    experiments = repo / "experiments"
    legacy = {
        path.stem for path in experiments.glob("exp[0-9][0-9][0-9].py")
    }
    staged = {
        path.name
        for path in experiments.glob("exp[0-9][0-9][0-9]")
        if path.is_dir()
        and any(
            (path / f"{stage}.py").is_file()
            for stage in ("compute", "analyse", "present")
        )
    }
    return legacy | staged


def coverage(repo: Path) -> dict[str, Any]:
    declared = memberships(repo)
    registered = set(declared)
    runnable = _runnable_experiments(repo)
    historical = load_history(repo)
    overlap = registered & set(historical)
    if overlap:
        raise PingstoreError(
            f"active and historical experiments must be disjoint: {sorted(overlap)}"
        )
    capture_routes: dict[str, str] = {}
    experiments = repo / "experiments"
    for experiment in sorted(runnable):
        stage_dir = experiments / experiment
        if stage_dir.is_dir() and any(
            (stage_dir / f"{stage}.py").is_file()
            for stage in ("compute", "analyse", "present")
        ):
            capture_routes[experiment] = "independent-stages"
            continue
        text = (experiments / f"{experiment}.py").read_text()
        if "published_run(" in text:
            capture_routes[experiment] = "atomic-published-run"
        elif "finalize_prepared_run(" in text:
            capture_routes[experiment] = "legacy-success-finalizer"
    return {
        "runnable": sorted(runnable),
        "registered": sorted(registered),
        "missing_membership": sorted(runnable - registered),
        "stale_membership": sorted(registered - runnable),
        "capture_routes": capture_routes,
        "missing_capture": sorted(runnable - set(capture_routes)),
        "historical": historical,
        "passed": runnable == registered and runnable == set(capture_routes),
    }
