"""Experiment collection membership derived from writing metadata."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .contracts import PingstoreError

EXPERIMENT_RE = re.compile(r"^exp[0-9]{3}$")
COLLECTION_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
WRITING_COLLECTION_RE = re.compile(r'collection:\s*"([a-z0-9-]+)"')


def memberships(repo: Path) -> dict[str, str]:
    """Return active experiment memberships declared by their writings."""
    result: dict[str, str] = {}
    for path in sorted((repo / "writings").glob("exp[0-9][0-9][0-9].typ")):
        matches = WRITING_COLLECTION_RE.findall(path.read_text(errors="replace"))
        if not matches:
            continue
        unique = set(matches)
        if len(unique) != 1:
            raise PingstoreError(f"writing declares conflicting collections: {path}")
        experiment = path.stem
        collection = matches[0]
        if not EXPERIMENT_RE.fullmatch(experiment):
            raise PingstoreError(f"invalid experiment writing: {experiment}")
        if not COLLECTION_RE.fullmatch(collection):
            raise PingstoreError(f"invalid collection for {experiment}: {collection}")
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
    legacy = {path.stem for path in experiments.glob("exp[0-9][0-9][0-9].py")}
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
        "article_only": sorted(registered - runnable),
        "capture_routes": capture_routes,
        "missing_capture": sorted(runnable - set(capture_routes)),
        "passed": runnable <= registered and runnable == set(capture_routes),
    }
