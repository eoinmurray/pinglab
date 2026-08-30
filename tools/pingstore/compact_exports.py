"""Recoverable migration to flat singletons and standard recording roles."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from .contracts import (
    RUN_SCHEMA,
    PingstoreError,
    load_json,
    payload_digest,
    validate_run,
    write_json_atomic,
)
from .layout import canonical_role_name, normalize_export_layout
from .prune_evidence import atomic_text, validate_graph, visible_runs


def validate_source_graph(runs: Path) -> dict[str, dict]:
    """Validate the checksums and dependency graph of the authorized v4.2 store."""
    records = {}
    for directory in visible_runs(runs):
        if directory.is_symlink() or {p.name for p in directory.iterdir()} != {
            "run.json",
            "README.md",
            "export",
        }:
            raise PingstoreError(f"invalid source run root: {directory}")
        record = validate_run(load_json(directory / "run.json"))
        if (
            record["schema"] != RUN_SCHEMA
            or payload_digest(directory) != record["payload_digest"]
        ):
            raise PingstoreError(f"invalid source run payload: {directory}")
        if any(path.is_symlink() for path in directory.rglob("*")):
            raise PingstoreError(f"source run contains a symlink: {directory}")
        records[directory.name] = record
    for child, record in records.items():
        for reference in record["inputs"].values():
            parent = records.get(reference["run_id"])
            if parent is None or parent["payload_digest"] != reference["payload_digest"]:
                raise PingstoreError(
                    f"{child}: missing or changed input {reference['run_id']}"
                )
    return records


def stage(staged: Path, archive_ref: str, changed_at: str) -> tuple[set[str], dict]:
    records = {path.name: load_json(path / "run.json") for path in visible_runs(staged)}
    original = json.loads(json.dumps(records))
    mappings = {}
    for run_id, record in records.items():
        mappings[run_id] = normalize_export_layout(staged / run_id, record)
        record["payload_digest"] = payload_digest(staged / run_id)
    digests = {run_id: record["payload_digest"] for run_id, record in records.items()}
    for record in records.values():
        for reference in record["inputs"].values():
            reference["payload_digest"] = digests[reference["run_id"]]

    changed = {run_id for run_id in records if records[run_id] != original[run_id]}
    for run_id in sorted(changed):
        directory = staged / run_id
        write_json_atomic(directory / "run.json", records[run_id])
        flattened = sum(
            "/" in old and "/" not in new for old, new in mappings[run_id].items()
        )
        renamed = sum(
            canonical_role_name(Path(old).name) != Path(old).name
            for old in mappings[run_id]
        )
        history = (directory / "README.md").read_text().rstrip()
        history += (
            f"\n\n- {changed_at}: compact-export migration flattened {flattened} "
            f"single-file units and standardized {renamed} role filenames while "
            "updating payload pins. No experiment stage was executed. Recoverable "
            f"originals: `{archive_ref}`.\n"
        )
        atomic_text(directory / "README.md", history)

    validated = validate_graph(staged)
    return changed, validated


def apply(repo: Path) -> Path:
    store = repo / ".pingstore"
    runs = store / "runs"
    validate_source_graph(runs)
    migration_id = "compact-exports-" + datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    archive = store / "migrations" / migration_id
    staged = store / f".{migration_id}.tmp"
    if archive.exists() or staged.exists():
        raise PingstoreError("migration target already exists")
    archive.mkdir(parents=True)
    shutil.copytree(runs, staged, copy_function=os.link)
    changed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    changed, validated = stage(
        staged, f"../../migrations/{migration_id}/original-runs", changed_at
    )
    report = {
        "schema": "pingstore.compact-exports/v1",
        "migration_id": migration_id,
        "created_at": changed_at,
        "status": "prepared",
        "changed_runs": sorted(changed),
        "validated_runs": len(validated),
    }
    write_json_atomic(archive / "migration.json", report)
    original = archive / "original-runs"
    os.rename(runs, original)
    try:
        os.rename(staged, runs)
        validate_graph(runs)
    except BaseException:
        if runs.exists():
            os.rename(runs, staged)
        os.rename(original, runs)
        raise
    report["status"] = "complete"
    write_json_atomic(archive / "migration.json", report)
    return archive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    print(apply(parser.parse_args().repo.resolve()))


if __name__ == "__main__":
    main()
