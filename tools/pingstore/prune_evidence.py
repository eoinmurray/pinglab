"""One-off, recoverable pruning of audited unused v4 evidence files."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .contracts import (
    PingstoreError,
    load_json,
    payload_digest,
    validate_operational_run_directory,
    write_json_atomic,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_text(path: Path, text: str) -> None:
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def visible_runs(runs: Path) -> list[Path]:
    return [
        path
        for path in sorted(runs.iterdir())
        if path.is_dir() and not path.name.startswith(".") and not path.is_symlink()
    ]


def validate_graph(runs: Path) -> dict[str, dict]:
    records = {
        directory.name: validate_operational_run_directory(directory)
        for directory in visible_runs(runs)
    }
    for child, record in records.items():
        for reference in record["inputs"].values():
            parent = records.get(reference["run_id"])
            if parent is None or parent["payload_digest"] != reference["payload_digest"]:
                raise PingstoreError(f"{child}: missing or changed input {reference['run_id']}")
    return records


def load_manifest(repo: Path, manifest: Path) -> dict[str, dict[str, int]]:
    selected: dict[str, dict[str, int]] = defaultdict(dict)
    for number, line in enumerate(manifest.read_text().splitlines(), 1):
        try:
            raw_path, raw_size = line.rsplit("\t", 1)
            size = int(raw_size)
        except ValueError as exc:
            raise PingstoreError(f"invalid manifest row {number}") from exc
        relative = Path(raw_path)
        parts = relative.parts
        if (
            relative.is_absolute()
            or len(parts) < 6
            or parts[:2] != (".pingstore", "runs")
            or parts[3:5] != ("export", "evidence")
            or any(part in ("", ".", "..") for part in parts)
        ):
            raise PingstoreError(f"unsafe evidence path on row {number}: {raw_path}")
        run_id = parts[2]
        evidence_relative = Path(*parts[5:]).as_posix()
        if evidence_relative in selected[run_id]:
            raise PingstoreError(f"duplicate manifest path: {raw_path}")
        path = repo / relative
        if path.is_symlink() or not path.is_file() or path.stat().st_size != size:
            raise PingstoreError(f"manifest file changed or missing: {raw_path}")
        selected[run_id][evidence_relative] = size
    if not selected:
        raise PingstoreError("empty evidence-pruning manifest")
    return dict(selected)


def reference_removed(value: str, all_files: set[str], deleted: set[str]) -> bool:
    prefix = "export/evidence/"
    if not value.startswith(prefix):
        return False
    relative = value.removeprefix(prefix).rstrip("/")
    covered = {path for path in all_files if path == relative or path.startswith(relative + "/")}
    return bool(covered) and covered <= deleted


def prune_references(value, *, all_files: set[str], deleted: set[str]):
    """Remove path declarations whose complete target is being deleted."""
    marker = object()

    def visit(item):
        if isinstance(item, str):
            return marker if reference_removed(item, all_files, deleted) else item
        if isinstance(item, dict):
            changed = False
            result = {}
            for key, child in item.items():
                revised = visit(child)
                if revised is marker:
                    changed = True
                    continue
                changed = changed or revised is not child
                result[key] = revised
            return marker if changed and not result else result
        if isinstance(item, list):
            revised = [child for child in (visit(child) for child in item) if child is not marker]
            return marker if len(revised) != len(item) and not revised else revised
        return item

    result = visit(value)
    if result is marker or not isinstance(result, dict):
        raise PingstoreError("pruning removed the run record root")
    return result


def remove_empty_directories(root: Path) -> None:
    if not root.exists():
        return
    for path in sorted((p for p in root.rglob("*") if p.is_dir()), reverse=True):
        if not any(path.iterdir()):
            path.rmdir()
    if not any(root.iterdir()):
        root.rmdir()


def archive_originals(
    archive: Path,
    runs: Path,
    selected: dict[str, dict[str, int]],
    changed_runs: set[str],
) -> None:
    for run_id in sorted(changed_runs):
        destination = archive / "records" / run_id
        destination.mkdir(parents=True)
        shutil.copy2(runs / run_id / "run.json", destination / "run.json")
        shutil.copy2(runs / run_id / "README.md", destination / "README.md")
    for run_id, paths in sorted(selected.items()):
        for relative in sorted(paths):
            source = runs / run_id / "export" / "evidence" / relative
            destination = archive / "deleted" / run_id / "export" / "evidence" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)


def stage_pruned_store(
    staged: Path,
    selected: dict[str, dict[str, int]],
    archive_relative: str,
    changed_at: str,
) -> tuple[dict[str, dict], set[str]]:
    records = {directory.name: load_json(directory / "run.json") for directory in visible_runs(staged)}
    original_records = json.loads(json.dumps(records))
    original_digests = {run_id: record["payload_digest"] for run_id, record in records.items()}

    for run_id, paths in selected.items():
        directory = staged / run_id
        evidence = directory / "export" / "evidence"
        all_files = {
            path.relative_to(evidence).as_posix()
            for path in evidence.rglob("*")
            if path.is_file()
        }
        deleted = set(paths)
        if not deleted <= all_files:
            raise PingstoreError(f"{run_id}: staged evidence differs from manifest")
        records[run_id] = prune_references(
            records[run_id], all_files=all_files, deleted=deleted
        )
        for relative, expected_size in paths.items():
            target = evidence / relative
            if target.stat().st_size != expected_size:
                raise PingstoreError(f"{run_id}: staged evidence size changed: {relative}")
            target.unlink()
        remove_empty_directories(evidence)
        records[run_id]["payload_digest"] = payload_digest(directory)

    digests = {run_id: record["payload_digest"] for run_id, record in records.items()}
    for record in records.values():
        for reference in record["inputs"].values():
            reference["payload_digest"] = digests[reference["run_id"]]

    changed_runs = {
        run_id for run_id, record in records.items() if record != original_records[run_id]
    }
    for run_id in sorted(changed_runs):
        directory = staged / run_id
        write_json_atomic(directory / "run.json", records[run_id])
        deleted_count = len(selected.get(run_id, {}))
        deleted_bytes = sum(selected.get(run_id, {}).values())
        digest_changed = original_digests[run_id] != records[run_id]["payload_digest"]
        details = []
        if deleted_count:
            details.append(f"removed {deleted_count} downstream-unused evidence files ({deleted_bytes} bytes)")
            details.append("removed their path declarations from run.json")
        if not deleted_count or any(
            original_records[run_id]["inputs"].get(role) != reference
            for role, reference in records[run_id]["inputs"].items()
        ):
            details.append("updated explicit input payload pins")
        if digest_changed:
            details.append("recomputed the export payload digest")
        history = (directory / "README.md").read_text().rstrip()
        history += (
            "\n\n- "
            + changed_at
            + ": evidence-pruning migration "
            + "; ".join(details)
            + f". Recoverable originals are at `{archive_relative}`. "
            + "No experiment stage was executed and no presentation was published.\n"
        )
        atomic_text(directory / "README.md", history)

    validated = validate_graph(staged)
    for run_id, paths in selected.items():
        for relative in paths:
            if (staged / run_id / "export" / "evidence" / relative).exists():
                raise PingstoreError(f"staged deletion failed: {run_id}/{relative}")
    return validated, changed_runs


def apply(repo: Path, manifest: Path) -> Path:
    store = repo / ".pingstore"
    runs = store / "runs"
    original_records = validate_graph(runs)
    selected = load_manifest(repo, manifest)
    missing = set(selected) - set(original_records)
    if missing:
        raise PingstoreError("manifest names unknown runs: " + ", ".join(sorted(missing)))
    migration_id = "evidence-prune-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = store / "migrations" / migration_id
    staged = store / f".{migration_id}-runs.tmp"
    old_runs = store / f".{migration_id}-runs.old"
    if any(path.exists() for path in (archive, staged, old_runs)):
        raise PingstoreError("evidence-pruning migration target already exists")

    changed_at = utc_now()
    owner_ids = set(selected)
    changed_runs = owner_ids | {
        child
        for child, record in original_records.items()
        if any(reference["run_id"] in owner_ids for reference in record["inputs"].values())
    }
    archive.mkdir(parents=True)
    report = {
        "schema": "pingstore.evidence-prune/v1",
        "migration_id": migration_id,
        "created_at": changed_at,
        "status": "preparing",
        "manifest": manifest.relative_to(repo).as_posix(),
        "deleted_files": sum(len(paths) for paths in selected.values()),
        "deleted_bytes": sum(sum(paths.values()) for paths in selected.values()),
        "owner_runs": sorted(owner_ids),
        "changed_runs": sorted(changed_runs),
        "original_payload_digests": {
            run_id: original_records[run_id]["payload_digest"] for run_id in sorted(owner_ids)
        },
    }
    write_json_atomic(archive / "migration.json", report)
    try:
        archive_originals(archive, runs, selected, changed_runs)
        shutil.copy2(manifest, archive / "downstream-unused.tsv")
        shutil.copytree(runs, staged, copy_function=os.link)
        validated, actual_changed = stage_pruned_store(
            staged,
            selected,
            f"../../migrations/{migration_id}",
            changed_at,
        )
        if actual_changed != changed_runs:
            raise PingstoreError(
                "unexpected changed-run set: " + ", ".join(sorted(actual_changed ^ changed_runs))
            )
        report["new_payload_digests"] = {
            run_id: validated[run_id]["payload_digest"] for run_id in sorted(owner_ids)
        }
        report["status"] = "validated"
        write_json_atomic(archive / "migration.json", report)

        os.replace(runs, old_runs)
        try:
            os.replace(staged, runs)
            validate_graph(runs)
        except BaseException:
            if runs.exists():
                os.replace(runs, staged)
            os.replace(old_runs, runs)
            raise
        shutil.rmtree(old_runs)
        report["status"] = "complete"
        report["completed_at"] = utc_now()
        write_json_atomic(archive / "migration.json", report)
        return archive
    except BaseException:
        report["status"] = "incomplete"
        report["failed_at"] = utc_now()
        write_json_atomic(archive / "migration.json", report)
        if staged.exists():
            shutil.rmtree(staged)
        raise


def rollback(repo: Path, archive: Path) -> None:
    report = load_json(archive / "migration.json")
    if report.get("schema") != "pingstore.evidence-prune/v1" or report.get("status") != "complete":
        raise PingstoreError("not a completed evidence-pruning archive")
    runs = repo / ".pingstore" / "runs"
    validate_graph(runs)
    staged = runs.parent / f".{report['migration_id']}-rollback.tmp"
    old_runs = runs.parent / f".{report['migration_id']}-rollback.old"
    if staged.exists() or old_runs.exists():
        raise PingstoreError("rollback staging path already exists")
    shutil.copytree(runs, staged, copy_function=os.link)
    try:
        for run_id in report["changed_runs"]:
            record_archive = archive / "records" / run_id
            shutil.copy2(record_archive / "run.json", staged / run_id / "run.json")
            shutil.copy2(record_archive / "README.md", staged / run_id / "README.md")
        deleted = archive / "deleted"
        for path in deleted.rglob("*"):
            if path.is_file():
                destination = staged / path.relative_to(deleted)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
        validate_graph(staged)
        os.replace(runs, old_runs)
        try:
            os.replace(staged, runs)
            validate_graph(runs)
        except BaseException:
            if runs.exists():
                os.replace(runs, staged)
            os.replace(old_runs, runs)
            raise
        shutil.rmtree(old_runs)
        report["status"] = "rolled_back"
        report["rolled_back_at"] = utc_now()
        write_json_atomic(archive / "migration.json", report)
    finally:
        if staged.exists():
            shutil.rmtree(staged)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--apply", type=Path, metavar="MANIFEST")
    action.add_argument("--rollback", type=Path, metavar="ARCHIVE")
    args = parser.parse_args(argv)
    repo = args.repo.resolve()
    if args.apply:
        print(apply(repo, args.apply.resolve()))
    else:
        rollback(repo, args.rollback.resolve())
        print(f"Rolled back {args.rollback.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
