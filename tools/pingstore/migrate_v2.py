"""One-off, copy-first v1-to-v2 migration; never uploads or deletes the rollback.

Run as `python -m pingstore.migrate_v2 prepare|activate|recover STORE WORKDIR`.
WORKDIR must be new for preparation and outside STORE on the same filesystem.
Stop writers/readers during activation. Recovery restores the original only if
activation was interrupted between its two directory renames.
"""

from __future__ import annotations

import argparse
import copy
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from .contracts import (
    LEGACY_RUN_SCHEMA,
    PingstoreError,
    file_sha256,
    load_json,
    payload_digest,
    payload_inventory,
    validate_run_directory,
    write_json_atomic,
)
from .layout import display_manifest, initialize_layout, legacy_mapping
from .payload import canonical_payload_digest, inventory_payload


def tree_inventory(root: Path) -> list[dict]:
    rows = payload_inventory(root)
    if (root / "run.json").is_file():
        rows.append(
            {
                "path": "run.json",
                "size_bytes": (root / "run.json").stat().st_size,
                "sha256": file_sha256(root / "run.json"),
            }
        )
    return sorted(rows, key=lambda row: row["path"])


def verify_v1(source: Path) -> tuple[dict, str]:
    if source.is_symlink() or not source.is_dir():
        raise PingstoreError("legacy run must be a real directory")
    names = {p.name for p in source.iterdir()}
    if names not in ({"run.json", "files"}, {"run.json", "README.md", "files"}):
        raise PingstoreError(f"unexpected legacy root entries: {source}")
    if any(p.is_symlink() for p in source.iterdir()):
        raise PingstoreError("legacy root symlinks are not supported")
    run = load_json(source / "run.json")
    if run.get("schema") != "pingstore.run/v1" or run.get("run_id") != source.name:
        raise PingstoreError(
            "migration requires a visible v1 run with matching identity"
        )
    inventory = inventory_payload(source / "files", run_id=run["run_id"])
    if "sha256:" + inventory["payload_digest"] == run.get("files_digest"):
        return run, "files/ matches stored v1 digest"
    # The existing import moved README from files/ to the root without changing
    # its digest. Accept only an EXACT reconstruction, not an unchecked mismatch.
    readme = source / "README.md"
    if readme.is_file() and not (source / "files/README.md").exists():
        rows = inventory["files"] + [
            {
                "path": "README.md",
                "role": "state",
                "size_bytes": readme.stat().st_size,
                "sha256": file_sha256(readme),
            }
        ]
        reconstructed = "sha256:" + canonical_payload_digest(
            sorted(rows, key=lambda r: r["path"])
        )
        if reconstructed == run.get("files_digest"):
            return (
                run,
                "stored v1 digest exactly reconstructed with root README.md at files/README.md",
            )
    raise PingstoreError(f"legacy checksum mismatch: {source}")


def migrate_run(source: Path, destination: Path) -> dict:
    run, verification = verify_v1(source)
    baseline = tree_inventory(source)
    mapping = {
        "files/" + old: new for old, new in legacy_mapping(source / "files").items()
    }
    mapping["run.json"] = "export/provenance/format-v1/run.json"
    if (source / "README.md").exists():
        mapping["README.md"] = "export/provenance/format-v1/README.md"
    reserved = "export/provenance/format-v1/"
    if any(new.startswith(reserved) for old, new in mapping.items() if old.startswith("files/")):
        raise PingstoreError("legacy output collides with reserved migration evidence")
    if destination.exists():
        raise PingstoreError(f"migration destination already exists: {destination}")
    temporary = destination.with_name("." + destination.name + ".tmp")
    if temporary.exists():
        raise PingstoreError(f"migration staging already exists: {temporary}")
    initialize_layout(temporary, run["experiment"], schema=LEGACY_RUN_SCHEMA)
    for row in baseline:
        target = temporary / mapping[row["path"]]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / row["path"], target)
        if (
            target.stat().st_size != row["size_bytes"]
            or file_sha256(target) != row["sha256"]
        ):
            raise PingstoreError(f"migration copy mismatch: {row['path']}")
    old_readme = source / "README.md"
    if old_readme.exists():
        notes = old_readme.read_text()
        notes = notes.replace("[../run.json](../run.json)", "[run.json](run.json)")
        notes = notes.replace("`pingstore.run/v1`", "`pingstore.run/v2`")
        notes = notes.replace("`files_digest`", "`payload_digest`")
        notes = notes.replace("[state/](state/)", "[export/state/](export/state/)")
        notes = notes.replace("under `files/state/`", "under `export/state/`")
        notes = notes.replace("](provenance/", "](export/provenance/")
        notes = notes.replace(
            "Root figures, `numbers.json`, `_manifest.json`, and other original outputs:",
            "Figures and `numbers.json` in `presentation/`, with original execution records in `export/provenance/legacy/`:",
        )
        notes += (
            "\n## Format migration\n\nThe original README and run manifest are preserved in `export/provenance/format-v1/`.\n"
            "The complete byte-preserving path mapping is in `export/provenance/format-v1/mapping.json`.\n"
            "`export/` retains scientific and execution records. `presentation/` is flat;\n"
            "nested raster names now use the `rasters__` prefix. Historical absolute paths\n"
            "and original source provenance have not been rewritten.\n"
        )
        (temporary / "README.md").write_text(notes)
    converted = copy.deepcopy(run)
    converted["schema"] = LEGACY_RUN_SCHEMA
    converted.pop("files_digest")
    local_map = converted.get("provenance", {}).get("import_map")
    if local_map in mapping:
        converted["provenance"]["import_map"] = mapping[local_map]
    importer = converted.get("provenance", {}).get("migration", {})
    if importer.get("script") in mapping:
        importer["script"] = mapping[importer["script"]]
    converted["format_migration"] = {
        "from_schema": "pingstore.run/v1",
        "at": datetime.now(timezone.utc).isoformat(),
        "source_files_digest": run["files_digest"],
        "source_verification": verification,
        "original_manifest": mapping["run.json"],
        "mapping": "export/provenance/format-v1/mapping.json",
    }
    manifest_path = source / "files/_manifest.json"
    if manifest_path.exists():
        display_manifest(temporary, load_json(manifest_path), run["run_id"],
                         schema=LEGACY_RUN_SCHEMA)
    records = [{**row, "destination": mapping[row["path"]]} for row in baseline]
    write_json_atomic(
        temporary / converted["format_migration"]["mapping"], {"files": records}
    )
    write_json_atomic(temporary / "run.json", {**converted, "payload_digest": "sha256:" + "0" * 64})
    converted["payload_digest"] = payload_digest(temporary)
    write_json_atomic(temporary / "run.json", converted)
    validate_run_directory(temporary)
    if tree_inventory(source) != baseline:
        raise PingstoreError("source changed during migration")
    os.rename(temporary, destination)
    return {
        "run_id": run["run_id"],
        "source_verification": verification,
        "source_file_count": len(baseline),
        "mapped_bytes": sum(r["size_bytes"] for r in baseline),
    }


def prepare_store(source: Path, work: Path) -> dict:
    source, work = source.absolute(), work.absolute()
    if (
        source.is_symlink()
        or work.is_symlink()
        or source in work.parents
        or work in source.parents
        or work == source
    ):
        raise PingstoreError("work directory must be outside the real source store")
    if work.exists():
        raise PingstoreError("migration work directory must be new")
    if {p.name for p in source.iterdir()} - {"runs", "collections.json"}:
        raise PingstoreError("unknown store-level entries require an explicit mapping")
    runs = sorted((source / "runs").iterdir())
    if any(p.name.startswith(".") or not p.is_dir() for p in runs):
        raise PingstoreError(
            "incomplete or unexpected run entries; stop writers before migration"
        )
    baseline = tree_inventory(source)
    work.mkdir(parents=True)
    if source.stat().st_dev != work.stat().st_dev:
        raise PingstoreError("migration and rollback must use the source filesystem")
    prepared = work / "prepared"
    (prepared / "runs").mkdir(parents=True)
    report = {
        "source": str(source),
        "work": str(work),
        "phase": "preparing",
        "source_inventory": baseline,
    }
    write_json_atomic(work / "migration.json", report)
    migrated = [migrate_run(run, prepared / "runs" / run.name) for run in runs]
    if (source / "collections.json").exists():
        shutil.copy2(source / "collections.json", prepared / "collections.json")
    if tree_inventory(source) != baseline:
        raise PingstoreError("store changed during preparation")
    report.update(
        phase="prepared", runs=migrated, prepared_inventory=tree_inventory(prepared)
    )
    write_json_atomic(work / "migration.json", report)
    return report


def activate_store(source: Path, work: Path) -> None:
    source, work = source.absolute(), work.absolute()
    report = load_json(work / "migration.json")
    if (
        report["source"] != str(source)
        or report["work"] != str(work)
        or report["phase"] != "prepared"
    ):
        raise PingstoreError("migration is not prepared for these paths")
    prepared, rollback = work / "prepared", work / "rollback"
    if rollback.exists():
        raise PingstoreError(
            "rollback already exists; inspect/recover interrupted migration"
        )
    if (
        tree_inventory(source) != report["source_inventory"]
        or tree_inventory(prepared) != report["prepared_inventory"]
    ):
        raise PingstoreError("source or prepared store changed since verification")
    for run in (prepared / "runs").iterdir():
        validate_run_directory(run)
    report["phase"] = "activating"
    write_json_atomic(work / "migration.json", report)
    os.rename(source, rollback)
    try:
        os.rename(prepared, source)
    except BaseException:
        os.rename(rollback, source)
        report["phase"] = "prepared"
        write_json_atomic(work / "migration.json", report)
        raise
    report["phase"] = "active"
    write_json_atomic(work / "migration.json", report)


def recover_store(source: Path, work: Path) -> None:
    source, work = source.absolute(), work.absolute()
    report = load_json(work / "migration.json")
    if (
        report["source"] != str(source)
        or report["work"] != str(work)
        or report["phase"] != "activating"
    ):
        raise PingstoreError("no interrupted activation for these paths")
    rollback = work / "rollback"
    if not source.exists() and rollback.is_dir():
        if tree_inventory(rollback) != report["source_inventory"]:
            raise PingstoreError("rollback checksum mismatch")
        os.rename(rollback, source)
        report["phase"] = "prepared"
    elif source.is_dir() and rollback.is_dir() and not (work / "prepared").exists():
        if tree_inventory(source) != report["prepared_inventory"]:
            raise PingstoreError("activated store checksum mismatch")
        report["phase"] = "active"
    elif (
        source.is_dir()
        and not rollback.exists()
        and tree_inventory(source) == report["source_inventory"]
    ):
        report["phase"] = "prepared"
    else:
        raise PingstoreError("ambiguous recovery state; no files changed")
    write_json_atomic(work / "migration.json", report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "activate", "recover"))
    parser.add_argument("store", type=Path)
    parser.add_argument("work", type=Path)
    args = parser.parse_args()
    operation = {
        "prepare": prepare_store,
        "activate": activate_store,
        "recover": recover_store,
    }[args.operation]
    operation(args.store, args.work)
    print(f"{args.operation} complete; journal: {args.work / 'migration.json'}")


if __name__ == "__main__":
    main()
