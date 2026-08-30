"""One-off recoverable migration to scientific-data-only v4 exports."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from .contracts import PingstoreError, load_json, payload_digest, write_json_atomic
from .prune_evidence import atomic_text, prune_references, validate_graph, visible_runs

CARRIED_EXP048 = (
    "headline_stream.pdf",
    "headline_stream.png",
    "varying_headline_stream.pdf",
    "varying_headline_stream.png",
)
LOG_RUNS = {"exp074-r001-compute", "exp075-r001-compute", "exp099-r001-compute"}


def _timing(imported: dict) -> dict:
    execution = imported["execution"]
    started = datetime.fromisoformat(execution["started_at"].replace("Z", "+00:00"))
    completed = datetime.fromisoformat(execution["completed_at"].replace("Z", "+00:00"))
    attempts = [cell["attempt"] for cell in execution["cells"]]
    return {
        "cells": len(attempts),
        "inherited_cells": 60,
        "retrained_cells": 42,
        "scheduler": "slurm",
        "origin": "slurm",
        "started_at": execution["started_at"],
        "completed_at": execution["completed_at"],
        "duration_seconds": (completed - started).total_seconds(),
        "jobs": len(attempts),
        "job_seconds": sum(item["elapsed_seconds"] for item in attempts),
        "note": "Compact timing projected from the archived historical import record.",
    }


def _remove_evidence_references(record: dict, evidence: Path) -> dict:
    files = {
        path.relative_to(evidence).as_posix()
        for path in evidence.rglob("*")
        if path.is_file()
    }
    return prune_references(record, all_files=files, deleted=files)


def stage(staged: Path, archive_ref: str, changed_at: str) -> tuple[set[str], dict]:
    records = {p.name: load_json(p / "run.json") for p in visible_runs(staged)}
    original = json.loads(json.dumps(records))

    exp022_record = load_json(
        staged / "exp022-r001-compute/export/evidence/imported-run.json"
    )
    commands = load_json(staged / "exp076-r001-compute/export/evidence/commands.json")

    for run_id, record in list(records.items()):
        directory = staged / run_id
        export = directory / "export"
        evidence = export / "evidence"
        if evidence.is_dir():
            if run_id == "exp048-r001-analyse":
                for name in CARRIED_EXP048:
                    shutil.copy2(evidence / "archive/payload" / name, export / name)
            if run_id == "exp082-r001-compute":
                shutil.copy2(
                    evidence / "archive/derived/artifacts/data/exp082/numbers.json",
                    export / "historical-summary.json",
                )
            record = _remove_evidence_references(record, evidence)
            shutil.rmtree(evidence)
        manifest = export / "_manifest.json"
        manifest.unlink(missing_ok=True)
        if run_id in LOG_RUNS:
            for name in ("output.log", "run.jsonl"):
                for path in export.rglob(name):
                    path.unlink()
        records[run_id] = record

    records["exp022-r001-compute"]["scientific_execution"] = _timing(exp022_record)
    records["exp022-r001-compute"]["historical_evidence"].pop("record", None)
    records["exp076-r001-compute"]["execution"]["commands"] = commands
    imported = records["exp082-r001-compute"]["historical_import"]
    imported.update(
        source_files=199,
        source_bytes=6079619,
        scientific_files=135,
        archived_metadata_files=64,
    )
    records["exp048-r001-analyse"].pop("source_file_mapping", None)

    for run_id, record in records.items():
        record["payload_digest"] = payload_digest(staged / run_id)
    digests = {run_id: record["payload_digest"] for run_id, record in records.items()}
    for record in records.values():
        for reference in record["inputs"].values():
            reference["payload_digest"] = digests[reference["run_id"]]

    changed = {run_id for run_id in records if records[run_id] != original[run_id]}
    for run_id in sorted(changed):
        directory = staged / run_id
        write_json_atomic(directory / "run.json", records[run_id])
        history = (directory / "README.md").read_text().rstrip()
        history += (
            f"\n\n- {changed_at}: export-layout migration removed retained execution metadata "
            "and compatibility manifests from scientific exports, flattened retained data "
            "where required, and updated payload pins. No experiment stage was executed. "
            f"Recoverable originals: `{archive_ref}`.\n"
        )
        atomic_text(directory / "README.md", history)

    validated = validate_graph(staged)
    forbidden = []
    for directory in visible_runs(staged):
        export = directory / "export"
        forbidden.extend(export.rglob("evidence"))
        forbidden.extend(export.rglob("_manifest.json"))
    if forbidden:
        raise PingstoreError("forbidden export metadata remains: " + str(forbidden[0]))
    return changed, validated


def apply(repo: Path) -> Path:
    store = repo / ".pingstore"
    runs = store / "runs"
    validate_graph(runs)
    migration_id = "export-layout-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = store / "migrations" / migration_id
    staged = store / f".{migration_id}.tmp"
    if archive.exists() or staged.exists():
        raise PingstoreError("migration target already exists")
    archive.mkdir(parents=True)
    shutil.copytree(runs, staged, copy_function=os.link)
    changed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    changed, validated = stage(staged, f"../../migrations/{migration_id}/original-runs", changed_at)
    report = {
        "schema": "pingstore.export-layout/v1",
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
    args = parser.parse_args()
    print(apply(args.repo.resolve()))


if __name__ == "__main__":
    main()
