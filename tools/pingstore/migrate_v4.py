"""Recoverable migration of completed v3 runs to the simpler v4 contract."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from .contracts import (
    PREVIOUS_RUN_SCHEMA,
    RUN_SCHEMA,
    PingstoreError,
    load_json,
    payload_digest,
    validate_operational_run_directory,
    validate_run_directory,
    write_json_atomic,
)

BOILERPLATE = {"command.json", "reservation.json", "run.sh", "source.patch", "writer.lock"}
HISTORY_MARKERS = ("migration", "correction", "ancestor-repair", "id-order")
ARCHIVED_METADATA = {
    "ancestry_repair",
    "format_migration",
    "hpc_identity_migration",
    "id_order_migration",
    "local_origin_correction",
    "source_neutral_id_migration",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def completed_v3_runs(runs: Path) -> list[Path]:
    result = []
    for directory in sorted(runs.iterdir()):
        if directory.name.startswith(".") or directory.is_symlink() or not directory.is_dir():
            continue
        record = load_json(directory / "run.json")
        if record.get("schema") != PREVIOUS_RUN_SCHEMA:
            raise PingstoreError(f"{directory.name}: migration requires a completed v3 run")
        validate_run_directory(directory)
        result.append(directory)
    return result


def retain_as_evidence(path: Path, stage: str) -> bool:
    if stage == "present" or path.name in BOILERPLATE:
        return False
    lowered = path.name.lower()
    return not any(marker in lowered for marker in HISTORY_MARKERS)


def rewrite_paths(value, *, retained: set[str], archive_prefix: str):
    if isinstance(value, dict):
        return {
            key: rewrite_paths(item, retained=retained, archive_prefix=archive_prefix)
            for key, item in value.items()
            if key != "run_json_sha256"
        }
    if isinstance(value, list):
        return [rewrite_paths(item, retained=retained, archive_prefix=archive_prefix)
                for item in value]
    if isinstance(value, str) and value.startswith("provenance/"):
        rest = value.removeprefix("provenance/")
        top = rest.split("/", 1)[0]
        if top in retained:
            return "export/evidence/" + rest
        return archive_prefix + "/provenance/" + rest
    return value


def readme(record: dict, existing: str, *, migrated_at: str, archive_prefix: str) -> str:
    text = existing.rstrip()
    if not text:
        text = (
            f"# {record['run_id']}\n\n"
            f"{record['stage'].capitalize()} run for `{record['experiment']}`. "
            "Machine-readable details are in `run.json`."
        )
    inputs = "\n".join(
        f"- `{role}`: `{reference['run_id']}` (`{reference['payload_digest']}`)"
        for role, reference in record["inputs"].items()
    ) or "- None"
    return (
        text
        + "\n\n## V4 history\n\n"
        + f"- Created: `{record['created_at']}`; stage: `{record['stage']}`; "
          f"origin: `{record['origin']}`.\n"
        + f"- Inputs:\n{inputs}\n"
        + f"- {migrated_at}: migrated from `pingstore.run/v3` to "
          f"`pingstore.run/v4`. Exported scientific bytes were preserved; "
          f"machine-consumed supporting records were reclassified under "
          f"`export/evidence/`; duplicate and historical provenance records are "
          f"recoverable at `{archive_prefix}`. No training, simulation, analysis, "
          f"plotting, materialization, or publication was performed.\n"
    )


def apply_migration(store: Path) -> Path:
    runs = store / "runs"
    sources = completed_v3_runs(runs)
    migration_id = "v3-to-v4-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = store / "migrations" / migration_id
    if archive.exists():
        raise PingstoreError(f"migration archive already exists: {archive}")
    archive.mkdir(parents=True)
    migrated_at = utc_now()
    plan = {
        "schema": "pingstore.migration/v4",
        "migration_id": migration_id,
        "created_at": migrated_at,
        "status": "preparing",
        "runs": [],
        "hidden_runs_unchanged": sorted(
            path.name for path in runs.iterdir() if path.name.startswith(".") and path.is_dir()
        ),
    }
    write_json_atomic(archive / "migration.json", plan)

    records: dict[str, dict] = {}
    digests: dict[str, str] = {}
    try:
        for directory in sources:
            original = load_json(directory / "run.json")
            run_archive = archive / directory.name
            run_archive.mkdir()
            shutil.copy2(directory / "run.json", run_archive / "run.json")
            original_readme = directory / "README.md"
            had_readme = original_readme.is_file()
            if had_readme:
                shutil.copy2(original_readme, run_archive / "README.md")
            provenance = directory / "provenance"
            if provenance.is_dir():
                shutil.copytree(provenance, run_archive / "provenance")
            replay_scripts = list((directory / "export").rglob("run.sh"))
            for script in replay_scripts:
                relative = script.relative_to(directory / "export")
                archived = run_archive / "export_replay_scripts" / relative
                archived.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(script, archived)
                script.unlink()

            evidence = directory / "export/evidence"
            if evidence.exists():
                raise PingstoreError(f"{directory.name}: export/evidence already exists")
            retained = set()
            if provenance.is_dir():
                selected = [path for path in provenance.iterdir()
                            if retain_as_evidence(path, original["stage"])]
                if selected:
                    evidence.mkdir()
                    for path in selected:
                        destination = evidence / path.name
                        if path.is_dir():
                            shutil.copytree(
                                path, destination, ignore=shutil.ignore_patterns("run.sh")
                            )
                        else:
                            shutil.copy2(path, destination)
                        retained.add(path.name)

            archive_prefix = f"../../migrations/{migration_id}/{directory.name}"
            record = rewrite_paths(original, retained=retained, archive_prefix=archive_prefix)
            for key in ARCHIVED_METADATA:
                record.pop(key, None)
            record["schema"] = RUN_SCHEMA
            record["inputs"] = {
                role: {"run_id": reference["run_id"],
                       "payload_digest": reference["payload_digest"]}
                for role, reference in original["inputs"].items()
            }
            compact = dict(record.get("provenance", {}))
            compact.pop("patch", None)
            compact.pop("dirty_patch", None)
            record["provenance"] = compact
            record["history_archive"] = archive_prefix
            write_json_atomic(directory / "run.json", record)
            record["payload_digest"] = payload_digest(directory)
            write_json_atomic(directory / "run.json", record)
            digests[directory.name] = record["payload_digest"]
            records[directory.name] = record
            existing = original_readme.read_text() if had_readme else ""
            (directory / "README.md").write_text(
                readme(record, existing, migrated_at=migrated_at, archive_prefix=archive_prefix)
            )
            if provenance.exists():
                shutil.rmtree(provenance)
            plan["runs"].append({
                "run_id": directory.name,
                "original_readme": had_readme,
                "retained_evidence": sorted(retained),
                "archived_export_replay_scripts": [
                    path.relative_to(directory / "export").as_posix()
                    for path in replay_scripts
                ],
                "v4_payload_digest": record["payload_digest"],
            })

        for directory in sources:
            record = records[directory.name]
            for reference in record["inputs"].values():
                try:
                    reference["payload_digest"] = digests[reference["run_id"]]
                except KeyError as exc:
                    raise PingstoreError(
                        f"{directory.name}: missing migrated input {reference['run_id']}"
                    ) from exc
            write_json_atomic(directory / "run.json", record)

        for directory in sources:
            record = validate_operational_run_directory(directory)
            for reference in record["inputs"].values():
                parent = validate_operational_run_directory(runs / reference["run_id"])
                if parent["payload_digest"] != reference["payload_digest"]:
                    raise PingstoreError(
                        f"{directory.name}: changed input {reference['run_id']}"
                    )
        plan["status"] = "complete"
        plan["completed_at"] = utc_now()
        write_json_atomic(archive / "migration.json", plan)
        return archive
    except BaseException:
        plan["status"] = "incomplete"
        plan["failed_at"] = utc_now()
        write_json_atomic(archive / "migration.json", plan)
        raise


def clean_migrated_records(store: Path) -> int:
    """Compact migrated metadata and remove archived replay scripts."""
    runs = store / "runs"
    changed_at = utc_now()
    migrations = store / "migrations"
    complete = sorted(migrations.glob("v3-to-v4-*/migration.json")) \
        if migrations.is_dir() else []
    report_path = complete[-1] if complete else None
    report = load_json(report_path) if report_path else None
    archive = report_path.parent if report_path else None
    by_id = {item["run_id"]: item for item in report["runs"]} if report else {}
    directories = [directory for directory in sorted(runs.iterdir())
                   if not directory.name.startswith(".") and directory.is_dir()]
    records = {}
    changed_runs = set()
    for directory in directories:
        record = validate_operational_run_directory(directory)
        compact = rewrite_paths(record, retained=set(), archive_prefix="")
        changed = compact != record
        for key in ARCHIVED_METADATA:
            changed = compact.pop(key, None) is not None or changed
        replay_scripts = list((directory / "export").rglob("run.sh"))
        for script in replay_scripts:
            if archive is None:
                raise PingstoreError("cannot remove replay script without a migration archive")
            relative = script.relative_to(directory / "export")
            archived = archive / directory.name / "export_replay_scripts" / relative
            archived.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(script, archived)
            script.unlink()
            changed = True
            item = by_id.get(directory.name)
            if item is not None:
                paths = item.setdefault("archived_export_replay_scripts", [])
                if relative.as_posix() not in paths:
                    paths.append(relative.as_posix())
        records[directory.name] = compact
        if changed:
            changed_runs.add(directory.name)
        write_json_atomic(directory / "run.json", compact)

    digests = {}
    for directory in directories:
        record = records[directory.name]
        digest = payload_digest(directory)
        if record["payload_digest"] != digest:
            record["payload_digest"] = digest
            changed_runs.add(directory.name)
        digests[directory.name] = digest

    for directory in directories:
        record = records[directory.name]
        for reference in record["inputs"].values():
            digest = digests[reference["run_id"]]
            if reference["payload_digest"] != digest:
                reference["payload_digest"] = digest
                changed_runs.add(directory.name)
        write_json_atomic(directory / "run.json", record)
        if directory.name not in changed_runs:
            continue
        with (directory / "README.md").open("a") as handle:
            handle.write(
                f"- {changed_at}: removed obsolete manifest-byte pins and v3 migration "
                "envelopes plus archived replay scripts from the operational run; "
                "originals remain in the migration archive.\n"
            )
        validate_operational_run_directory(directory)

    if report_path is not None:
        for run_id, digest in digests.items():
            if run_id in by_id:
                by_id[run_id]["v4_payload_digest"] = digest
        report["compacted_at"] = changed_at
        write_json_atomic(report_path, report)
    return len(changed_runs)


def rollback(store: Path, archive: Path) -> None:
    report = load_json(archive / "migration.json")
    if report.get("schema") != "pingstore.migration/v4":
        raise PingstoreError("not a v4 migration archive")
    runs = store / "runs"
    for item in report["runs"]:
        directory = runs / item["run_id"]
        current = validate_operational_run_directory(directory)
        if current["payload_digest"] != item["v4_payload_digest"]:
            raise PingstoreError(f"{directory.name}: v4 export changed; refusing rollback")
    for item in report["runs"]:
        directory = runs / item["run_id"]
        run_archive = archive / directory.name
        evidence = directory / "export/evidence"
        if evidence.exists():
            shutil.rmtree(evidence)
        replay_archive = run_archive / "export_replay_scripts"
        if replay_archive.is_dir():
            for archived in replay_archive.rglob("*"):
                if archived.is_file():
                    destination = directory / "export" / archived.relative_to(replay_archive)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(archived, destination)
        shutil.copy2(run_archive / "run.json", directory / "run.json")
        readme_path = run_archive / "README.md"
        if readme_path.is_file():
            shutil.copy2(readme_path, directory / "README.md")
        else:
            (directory / "README.md").unlink(missing_ok=True)
        provenance = run_archive / "provenance"
        if provenance.is_dir():
            shutil.copytree(provenance, directory / "provenance")
        validate_run_directory(directory)
    report["status"] = "rolled_back"
    report["rolled_back_at"] = utc_now()
    write_json_atomic(archive / "migration.json", report)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", type=Path, default=Path(".pingstore"))
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--apply", action="store_true")
    action.add_argument("--rollback", type=Path)
    action.add_argument("--clean-v4", action="store_true")
    args = parser.parse_args(argv)
    store = args.store.resolve()
    if args.rollback:
        rollback(store, args.rollback.resolve())
        print(f"Rolled back {args.rollback}")
    elif args.apply:
        print(apply_migration(store))
    elif args.clean_v4:
        print(f"Cleaned {clean_migrated_records(store)} v4 manifests")
    else:
        sources = completed_v3_runs(store / "runs")
        hidden = [path for path in (store / "runs").iterdir()
                  if path.name.startswith(".") and path.is_dir()]
        print(json.dumps({
            "completed_v3_runs": len(sources),
            "hidden_runs_unchanged": sorted(path.name for path in hidden),
            "target_schema": RUN_SCHEMA,
        }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
