"""Explicit, non-destructive conversion. Never run automatically during reads."""

from __future__ import annotations

import os
import shutil
from datetime import datetime
from pathlib import Path

from .core import (
    Store,
    inventory,
    load_json,
    operation_lock,
    safe_file,
    utc_now,
    validate_run,
    write_json_atomic,
)


def dependencies(record):
    """Keep current stage links and external scientific inputs; detach replaced history."""
    links = dict(record.get("runset_inputs", record.get("inputs", {})))
    links.update(
        {
            k: v
            for k, v in record.get("inputs", {}).items()
            if not v["run_id"].startswith(record["experiment"] + "-")
        }
    )
    links.update(record.get("archived_inputs", {}))
    if record.get("consolidation", {}).get("sources"):
        links = {k: v for k, v in links.items() if not k.startswith("original_")}
    return list(dict.fromkeys(v["run_id"] for v in links.values()))


def convert(record):
    execution = record.get("execution", {})
    scientific = (
        record.get("scientific_execution")
        or record.get("completed_hpc_import", {}).get("source_execution")
        or execution
    )
    metadata = {
        "origin": record.get("scientific_execution", {}).get("origin")
        or record.get("completed_hpc_import", {}).get("source_origin")
        or record.get("origin", "unknown")
    }
    if scientific.get("elapsed_seconds") is not None:
        metadata["duration_seconds"] = scientific["elapsed_seconds"]
    elif scientific.get("started_at") and scientific.get("completed_at"):
        metadata["duration_seconds"] = (
            datetime.fromisoformat(scientific["completed_at"].replace("Z", "+00:00"))
            - datetime.fromisoformat(scientific["started_at"].replace("Z", "+00:00"))
        ).total_seconds()
    if record.get("phase"):
        metadata["phase"] = record["phase"]
    result = dict(
        id=record["run_id"],
        experiment=record["experiment"],
        stage=record["stage"],
        created_at=execution.get("completed_at") or record["created_at"],
        description=execution.get("operation", ""),
        depends_on=dependencies(record),
        metadata=metadata,
    )
    details = {
        k: v
        for k, v in record.items()
        if k not in {"schema", "payload_digest", "runset_inputs", "archived_inputs"}
    }
    details["inputs"] = {
        role: {"run_id": pin["run_id"]}
        for role, pin in record.get("inputs", {}).items()
    }
    return validate_run(result), details


def migrate(source, destination, *, apply=False, link_payloads=False):
    """Copy old metadata and files into a separate store; preserve originals verbatim.

    link_payloads is for local validation copies only. Both stores must remain
    immutable while sharing linked payloads. Metadata is always written separately.
    """
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if (
        source == destination
        or destination.is_relative_to(source)
        or source.is_relative_to(destination)
    ):
        raise ValueError(
            "migration requires a separate destination outside the source store"
        )
    if not (source / "runs").is_dir():
        raise ValueError("migration source must contain a runs directory")
    records = {
        p.parent.name: load_json(p)
        for p in sorted((source / "runs").glob("*/run.json"))
        if not p.parent.name.startswith(".")
    }
    converted = {}
    for identity, original in records.items():
        record, details = convert(original)
        if record["id"] != identity:
            raise ValueError(f"source identity mismatch: {identity}")
        converted[identity] = (record, details, source / "runs" / identity, None)
    # Materialise old logical source views as ordinary independent runs. There is
    # no alias resolver or consolidated ownership policy in the replacement store.
    for owner, original in records.items():
        consolidation = original.get("consolidation", {})
        if not consolidation.get("resolve_sources"):
            continue
        for identity, archived in consolidation["sources"].items():
            if identity in converted:
                continue
            record, details = convert(archived["record"])
            converted[identity] = (
                record,
                details,
                source / "runs" / owner,
                archived["export_paths"],
            )
    available = set(converted)
    plan = {
        "source": str(source),
        "destination": str(destination),
        "runs": sorted(converted),
        "unresolved_dependencies": {
            i: [d for d in r[0]["depends_on"] if d not in available]
            for i, r in converted.items()
            if set(r[0]["depends_on"]) - available
        },
        "unfinished_runs": sorted(
            path.name for path in (source / "runs").glob(".*.tmp")
        ),
        "applied": False,
    }
    if not apply:
        return plan
    store = Store(destination)
    with operation_lock(store.root):
        for identity, (record, details, directory, mapping) in converted.items():
            target = store.runs / identity
            if target.exists():
                if (
                    store.get(identity).record != record
                    or load_json(target / "details.json") != details
                ):
                    raise ValueError(f"migration destination differs: {identity}")
                continue
            temporary = store.runs / f".{identity}.tmp"
            temporary.mkdir(parents=True)
            original = records.get(identity)
            if original is None:
                original = next(
                    r["consolidation"]["sources"][identity]["record"]
                    for r in records.values()
                    if identity in r.get("consolidation", {}).get("sources", {})
                )
            write_json_atomic(
                store.root / "migration" / "originals" / f"{identity}.json", original
            )
            _copy_payload(directory, temporary, mapping, link_payloads)
            # Unlink possible copied metadata before writing, even in hard-link mode.
            (temporary / "details.json").unlink(missing_ok=True)
            write_json_atomic(temporary / "details.json", details)
            store._commit(temporary, record)
        plan.update(applied=True, completed_at=utc_now())
        write_json_atomic(store.root / "migration" / "report.json", plan)
    return plan


def _copy_payload(directory, temporary, mapping, link_payloads):
    copy = os.link if link_payloads else shutil.copy2
    inventory(directory)
    if mapping is None:
        for item in directory.iterdir():
            if item.name in {"run.json", ".protected"}:
                continue
            if item.is_dir():
                shutil.copytree(item, temporary / item.name, copy_function=copy)
            else:
                copy(item, temporary / item.name)
    else:
        for logical, retained in mapping.items():
            target_file = safe_file(temporary / "export", logical)
            target_file.parent.mkdir(parents=True, exist_ok=True)
            copy(safe_file(directory / "export", retained), target_file)
