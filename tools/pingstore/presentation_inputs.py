"""Pinglab-owned presentation projection. Never writes to Pingstore or runs experiments.

Demolab's generic prepare hook calls this before ordinary and URL-driven renders.
The generated JSON belongs to the presentation runtime, not the storage convention.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from pingstore.contracts import (
    PingstoreError,
    load_json,
    write_json_atomic,
)
from pingstore.discovery import discover_runs
from pingstore.layout import export_directory, presentation_directory


def article_inputs(root: Path) -> dict[str, list[str]]:
    result = {}
    for source in sorted((root / "writings").glob("exp*.typ")):
        text = source.read_text(encoding="utf-8")
        declaration = re.search(r"^#let inputs = \((.*?)\)", text, re.M | re.S)
        if declaration:
            result[source.stem] = re.findall(r'"([^"]+)"', declaration[1])
    return result


def experiment_dependencies(articles: dict, declared: dict | None = None) -> dict:
    """Merge declared computation and article inputs, independently of stored runs."""
    upstream = {article: set() for article in articles}
    for article, parents in (declared or {}).items():
        upstream.setdefault(article, set()).update(parents)
    for article, keys in articles.items():
        upstream[article].update(key.split(".")[-1] for key in keys)
    for article, parents in list(upstream.items()):
        parents.discard(article)
        for parent in parents:
            upstream.setdefault(parent, set())
    downstream = {article: set() for article in upstream}
    for article, parents in upstream.items():
        for parent in parents:
            downstream[parent].add(article)
    return {
        article: {"upstream": sorted(parents), "downstream": sorted(downstream[article])}
        for article, parents in sorted(upstream.items())
    }


def execution_duration(record: dict) -> float | None:
    """Elapsed time of this recorded operation, never inherited scientific work."""
    execution = record["execution"]
    values = [execution.get(key) for key in ("started_at", "completed_at")]
    if any(value is None for value in values):
        return None
    try:
        timestamps = [datetime.fromisoformat(value.replace("Z", "+00:00")) for value in values]
        if any(value.utcoffset() is None for value in timestamps):
            raise ValueError("execution timestamps must include a timezone")
        seconds = (timestamps[1] - timestamps[0]).total_seconds()
        if seconds < 0:
            raise ValueError("completed_at precedes started_at")
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise PingstoreError(f"{record['run_id']}: invalid execution duration: {exc}") from exc
    return seconds


def scientific_timing(directory: Path, record: dict) -> dict | None:
    """Project compact historical timing stored authoritatively in run.json."""
    declaration = record.get("scientific_execution")
    if declaration is None:
        return None
    if not isinstance(declaration, dict):
        raise PingstoreError(f"{record['run_id']}: invalid scientific_execution")
    required = {"duration_seconds", "started_at", "completed_at", "origin"}
    if not required <= declaration.keys():
        raise PingstoreError(f"{record['run_id']}: incomplete scientific_execution")
    seconds = declaration["duration_seconds"]
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0:
        raise PingstoreError(f"{record['run_id']}: invalid scientific duration")
    elapsed = execution_duration(
        {
            "run_id": record["run_id"],
            "execution": {
                "started_at": declaration["started_at"],
                "completed_at": declaration["completed_at"],
            },
        }
    )
    if elapsed != seconds:
        raise PingstoreError(f"{record['run_id']}: inconsistent scientific duration")
    result = {
        "duration_seconds": seconds,
        "started_at": declaration["started_at"],
        "completed_at": declaration["completed_at"],
        "origin": declaration["origin"],
        "job_seconds": declaration.get("job_seconds"),
        "jobs": declaration.get("jobs"),
    }
    return result


def historical_producer(record: dict) -> tuple[dict, dict]:
    historical = record.get("historical_import")
    if not isinstance(historical, dict):
        return {}, {}
    producer = historical.get("producer")
    return historical, producer if isinstance(producer, dict) else {}


def display_origin(record: dict) -> dict[str, str]:
    """Choose the origin of the displayed scientific work, not a local import."""
    scientific = record.get("scientific_execution")
    if isinstance(scientific, dict):
        origin = scientific.get("origin")
        if isinstance(origin, str) and origin:
            return {"value": origin, "basis": "scientific-execution"}

    historical, producer = historical_producer(record)
    origin = producer.get("origin", historical.get("producer_origin"))
    if isinstance(origin, str) and origin:
        return {"value": origin, "basis": "historical-producer"}
    if "slurm_job_id" in producer or "slurm_array" in historical:
        return {"value": "slurm", "basis": "historical-producer"}

    operation = record["execution"].get("operation")
    if operation in ("import", "historical-import"):
        return {"value": "unknown", "basis": "unrecorded-import-source"}
    return {"value": record["origin"], "basis": "recorded-operation"}


def display_timing(record: dict, scientific: dict | None) -> dict:
    """Choose wall-clock timing for the displayed execution provenance."""
    imported = record["execution"].get("operation") in ("import", "historical-import")
    operation_seconds = execution_duration(record)
    if scientific is not None:
        return {
            **scientific,
            "basis": "scientific-execution",
            "import_seconds": operation_seconds if imported else None,
        }

    if imported:
        _, producer = historical_producer(record)
        for key in ("status", "study_status", "execution_window"):
            window = producer.get(key)
            if not isinstance(window, dict):
                continue
            started = window.get("started_at_utc")
            completed = window.get("ended_at_utc")
            if started is None and completed is None:
                continue
            seconds = execution_duration({
                "run_id": record["run_id"],
                "execution": {"started_at": started, "completed_at": completed},
            })
            return {
                "duration_seconds": seconds,
                "started_at": started,
                "completed_at": completed,
                "basis": "historical-producer",
                "import_seconds": operation_seconds,
            }
        return {
            "duration_seconds": None,
            "basis": "unrecorded-import-source",
            "import_seconds": operation_seconds,
        }

    return {
        "duration_seconds": operation_seconds,
        "basis": "recorded-operation",
        "import_seconds": None,
    }


def projection(
    root: Path, *, overrides: dict | None = None, article: str = "",
    declared_dependencies: dict | None = None,
) -> dict:
    source = root / ".pingstore/runs"
    # The authoritative discovery adapter validates ALL visible v4 payloads first.
    discovered = discover_runs(source) if source.exists() or source.is_symlink() else []
    records = {}
    if source.exists():
        for directory in sorted(source.iterdir()):
            if (
                directory.name.startswith(".")
                or directory.is_symlink()
                or not directory.is_dir()
            ):
                continue
            records[directory.name] = load_json(directory / "run.json")
    for key, record in records.items():
        for reference in record["inputs"].values():
            parent = records.get(reference["run_id"])
            if (
                parent is None
                or parent["payload_digest"] != reference["payload_digest"]
            ):
                raise PingstoreError(
                    f"{key}: missing or changed upstream input {reference['run_id']}"
                )

    def ancestors(key: str, trail: tuple = ()) -> set[str]:
        if key in trail:
            raise PingstoreError("cyclic run provenance: " + key)
        found = set()
        for reference in records[key]["inputs"].values():
            parent = reference["run_id"]
            found.add(parent)
            found.update(ancestors(parent, (*trail, key)))
        return found

    sizes = {}
    for key in records:
        directory = source / key
        files = [
            p
            for p in directory.rglob("*")
            if p.is_file() and p != directory / "run.json"
        ]
        sizes[key] = sum(p.stat().st_size for p in files)

    memberships = {}
    views = root / ".pingstore/collections.json"
    if views.exists():
        from pingstore.contracts import validate_collections

        for name, ids in validate_collections(load_json(views)).items():
            for key in ids:
                memberships.setdefault(key, []).append(name)
    runs = []
    for entry in discovered:
        record = records[entry["id"]]
        timing = scientific_timing(source / entry["id"], record)
        directory = presentation_directory(source / entry["id"], record)
        files = sorted(directory.iterdir())
        parents = ancestors(entry["id"])
        runs.append(
            {
                "id": entry["id"],
                "experiment": record["experiment"],
                "stage": record["stage"],
                "created_at": entry["created_at"],
                "collection": record["collection"],
                "views": memberships.get(entry["id"], []),
                "origin": record["origin"],
                "display_origin": display_origin(record),
                "duration_seconds": execution_duration(record),
                "display_timing": display_timing(record, timing),
                "execution_operation": record["execution"].get("operation"),
                "scientific_timing": timing,
                "basepath": "/" + directory.relative_to(root).as_posix(),
                "export_bytes": sum(p.stat().st_size for p in files),
                "export_files": len(files),
                "payload_bytes": sizes[entry["id"]],
                "upstream_payload_bytes": sum(sizes[key] for key in parents),
                "upstream_runs": sorted(parents),
                "files": [p.name for p in files],
            }
        )
    runs.sort(key=lambda row: (row["created_at"], row["id"]), reverse=True)
    # Display-only stage rows never enter the selectable presentation inventory.
    display_runs = list(runs)
    for key, record in records.items():
        if record["stage"] == "present":
            continue
        directory = export_directory(source / key, record)
        timing = scientific_timing(source / key, record)
        display_runs.append({
            "id": record["run_id"],
            "experiment": record["experiment"],
            "stage": record["stage"],
            "created_at": datetime.fromisoformat(
                record["created_at"].replace("Z", "+00:00")
            ).astimezone(timezone.utc).isoformat(),
            "origin": record["origin"],
            "display_origin": display_origin(record),
            "duration_seconds": execution_duration(record),
            "display_timing": display_timing(record, timing),
            "execution_operation": record["execution"].get("operation"),
            "scientific_timing": timing,
            "export_bytes": sum(
                path.stat().st_size for path in directory.rglob("*") if path.is_file()
            ),
        })
    display_runs.sort(key=lambda row: (row["created_at"], row["id"]), reverse=True)
    attachments = article_inputs(root)
    defaults_path = root / "writings/run-defaults.json"
    defaults = load_json(defaults_path) if defaults_path.exists() else {}
    by_id = {run["id"]: run for run in runs}
    for article_id, pins in defaults.items():
        if article_id not in attachments or not isinstance(pins, dict):
            raise PingstoreError("unknown article or invalid defaults: " + article_id)
        for key, identity in pins.items():
            run = by_id.get(identity)
            if (
                key not in attachments[article_id]
                or run is None
                or run["experiment"] != key.split(".")[-1]
            ):
                raise PingstoreError(
                    f"{article_id}/{key}: unavailable default {identity}"
                )
    for name, basepath in (overrides or {}).items():
        key = name.removeprefix("source.")
        if not name.startswith("source.") or key not in attachments.get(article, []):
            raise PingstoreError(f"{article}: input is not declared: {name}")
        if not any(
            run["basepath"] == basepath and run["experiment"] == key.split(".")[-1]
            for run in runs
        ):
            raise PingstoreError(
                f"{key}: URL does not select a validated presentation export"
            )
    return {
        "schema": "pinglab.presentation-inputs/v1",
        "articles": attachments,
        "experiment_dependencies": experiment_dependencies(attachments, declared_dependencies),
        "defaults": defaults,
        "runs": runs,
        "display_runs": display_runs,
    }


def prepare(root: Path, *, declared_dependencies: dict | None = None) -> int:
    """Write validated presentation JSON for the supplied lab, never the package."""
    try:
        root = root.resolve()
        overrides = json.loads(os.environ.get("DEMOLAB_INPUTS", "{}"))
        if not isinstance(overrides, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in overrides.items()
        ):
            raise PingstoreError("DEMOLAB_INPUTS must be a JSON object of strings")
        data = projection(
            root,
            overrides=overrides,
            article=os.environ.get("DEMOLAB_ARTICLE", ""),
            declared_dependencies=declared_dependencies,
        )
        target = root / ".demolab/pinglab-inputs.json"
        serialized = json.dumps(data, indent=2, sort_keys=True) + "\n"
        if not target.is_file() or target.read_text() != serialized:
            write_json_atomic(target, data)
        print(
            f"Validated presentation inputs: {len(data['runs'])} present runs",
            file=sys.stderr,
        )
        return 0
    except (OSError, ValueError) as exc:
        print(f"pingstore presentation-inputs: {exc}", file=sys.stderr)
        return 1
