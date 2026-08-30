"""Authoritative experiment-to-collection membership registry."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .contracts import PingstoreError

REGISTRY_SCHEMA = "pingstore.experiment-registry/v1"
EXPERIMENT_RE = re.compile(r"^exp[0-9]{3}$")
COLLECTION_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
WRITING_COLLECTION_RE = re.compile(r'collection:\s*"([a-z0-9-]+)"')


def registry_path(repo: Path) -> Path:
    return repo / "experiments/collections/registry.json"


def load_registry(repo: Path) -> dict[str, Any]:
    path = registry_path(repo)
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise PingstoreError(f"experiment registry is missing: {path}") from exc
    if value.get("schema") != REGISTRY_SCHEMA:
        raise PingstoreError(f"experiment registry schema must be {REGISTRY_SCHEMA}")
    experiments = value.get("experiments")
    if not isinstance(experiments, dict):
        raise PingstoreError("experiment registry experiments must be an object")
    for experiment, collection in experiments.items():
        if not EXPERIMENT_RE.fullmatch(experiment):
            raise PingstoreError(f"invalid registered experiment: {experiment}")
        if not isinstance(collection, str) or not COLLECTION_RE.fullmatch(collection):
            raise PingstoreError(f"invalid collection for {experiment}: {collection}")
    historical = value.get("historical", {})
    if not isinstance(historical, dict):
        raise PingstoreError("experiment registry historical must be an object")
    if set(experiments) & set(historical):
        raise PingstoreError(
            "runnable and historical registry entries must be disjoint"
        )
    return value


def memberships(repo: Path) -> dict[str, str]:
    registry = load_registry(repo)
    result = dict(registry["experiments"])
    result.update(
        {
            experiment: row["collection"]
            for experiment, row in registry["historical"].items()
            if isinstance(row, dict) and isinstance(row.get("collection"), str)
        }
    )
    return result


def coverage(repo: Path) -> dict[str, Any]:
    registry = load_registry(repo)
    registered = set(registry["experiments"])
    experiments = repo / "experiments"
    legacy_runnable = {
        path.stem for path in (repo / "experiments").glob("exp[0-9][0-9][0-9].py")
    }
    staged_runnable = {
        path.name
        for path in experiments.glob("exp[0-9][0-9][0-9]")
        if path.is_dir()
        and any(
            (path / f"{stage}.py").is_file()
            for stage in ("compute", "analyse", "present")
        )
    }
    runnable = legacy_runnable | staged_runnable
    capture_routes: dict[str, str] = {}
    for experiment in sorted(runnable):
        stage_dir = experiments / experiment
        # An audit may reuse another experiment's compute run without owning
        # a compute stage of its own.
        if all(
            (stage_dir / f"{stage}.py").is_file() for stage in ("analyse", "present")
        ):
            capture_routes[experiment] = "independent-stages"
            continue
        text = (experiments / f"{experiment}.py").read_text()
        if "published_run(" in text:
            capture_routes[experiment] = "atomic-published-run"
        elif "finalize_prepared_run(" in text:
            capture_routes[experiment] = "legacy-success-finalizer"
    missing_capture = sorted(runnable - set(capture_routes))
    writing_mismatches: dict[str, dict[str, str]] = {}
    for experiment, registered_collection in registry["experiments"].items():
        writing = repo / "writings" / f"{experiment}.typ"
        if not writing.is_file():
            continue
        match = WRITING_COLLECTION_RE.search(writing.read_text(errors="replace")[:2000])
        if match and match.group(1) != registered_collection:
            writing_mismatches[experiment] = {
                "registry": registered_collection,
                "writing": match.group(1),
            }
    return {
        "runnable": sorted(runnable),
        "registered": sorted(registered),
        "missing_membership": sorted(runnable - registered),
        "stale_membership": sorted(registered - runnable),
        "capture_routes": capture_routes,
        "missing_capture": missing_capture,
        "writing_mismatches": writing_mismatches,
        "historical": registry["historical"],
        "passed": (
            runnable == registered and not missing_capture and not writing_mismatches
        ),
    }
