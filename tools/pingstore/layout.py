"""Versioned run layout and explicit adapters for legacy output producers."""

from __future__ import annotations

import shutil
from pathlib import Path

from .contracts import (
    LEGACY_RUN_SCHEMA, PREVIOUS_RUN_SCHEMA, RUN_SCHEMA, PingstoreError, load_json,
    write_json_atomic,
)

FIGURE_SUFFIXES = {".svg", ".png", ".jpg", ".jpeg", ".pdf", ".gif", ".webp", ".mp4"}
DATA_SUFFIXES = {".h5", ".hdf5", ".npy", ".npz", ".pt", ".pth"}
RECORD_NAMES = {
    "_manifest.json",
    "_run.txt",
    "_provenance.json",
    "run.sh",
    "_dirty.patch",
    "reproducer.json",
}


def initialize_layout(root: Path, experiment: str, *, schema: str = RUN_SCHEMA) -> None:
    if schema not in (LEGACY_RUN_SCHEMA, PREVIOUS_RUN_SCHEMA, RUN_SCHEMA):
        raise PingstoreError(f"unsupported layout schema: {schema}")
    (root / "export").mkdir(parents=True, exist_ok=True)
    if schema == RUN_SCHEMA:
        readme = root / "README.md"
        if not readme.exists():
            readme.write_text(f"# {experiment} run\n\n## History\n\n")
        return
    if schema == PREVIOUS_RUN_SCHEMA:
        return
    (root / "presentation").mkdir(exist_ok=True)
    readme = root / "README.md"
    if not readme.exists():
        readme.write_text(
            f"# {experiment}\n\nExecution provenance is in `run.json`. Scientific results and execution\nrecords are in `export/`; copyable publication inputs are in `presentation/`.\n"
        )


def export_directory(root: Path, run: dict) -> Path:
    """Resolve scientific output from an already validated v2/v3 record."""
    default = "export" if run["schema"] in (PREVIOUS_RUN_SCHEMA, RUN_SCHEMA) else "export/state"
    return root / run.get("export_root", default)


def presentation_directory(root: Path, run: dict) -> Path | None:
    """Resolve publishable output; stage is authoritative, not the folder name.

    Callers must validate the complete run before consuming this directory.
    Untyped v2 evidence retains its original presentation behaviour.
    """
    if run["schema"] in (PREVIOUS_RUN_SCHEMA, RUN_SCHEMA):
        return root / "export" if run["stage"] == "present" else None
    if run["schema"] == LEGACY_RUN_SCHEMA:
        return root / "presentation" if run.get("stage") in (None, "present") else None
    raise PingstoreError(f"unsupported layout schema: {run['schema']}")


def has_presentation_content(directory: Path) -> bool:
    return any(path.name not in RECORD_NAMES and path.stat().st_size > 0
               for path in directory.iterdir())


def legacy_target(relative: Path) -> Path:
    """Classify known legacy outputs; refuse unknown files rather than lose them."""
    first = relative.parts[0]
    if first in {"state", "provenance"}:
        return Path("export") / relative
    if relative.name in RECORD_NAMES or relative.name.endswith("_command.json"):
        return Path("export/provenance/legacy") / relative
    if relative.suffix.lower() in FIGURE_SUFFIXES and not any(
        p.endswith(".bundle") for p in relative.parts
    ):
        return Path("presentation") / "__".join(relative.parts)
    if relative.suffix.lower() in DATA_SUFFIXES or any(
        p.endswith(".bundle") for p in relative.parts
    ):
        return Path("export/derived") / relative
    if len(relative.parts) == 1 and relative.suffix in {".json", ".csv", ".tsv"}:
        return Path("presentation") / relative.name
    raise PingstoreError(f"unclassified legacy output: {relative}")


def legacy_mapping(source: Path) -> dict[str, str]:
    mapping = {}
    targets: set[str] = set()
    for path in sorted(source.rglob("*")):
        if path.is_symlink() or not (path.is_dir() or path.is_file()):
            raise PingstoreError(f"unsupported legacy entry: {path}")
        if path.is_file():
            relative = path.relative_to(source)
            target = legacy_target(relative).as_posix()
            if target.casefold() in targets:
                raise PingstoreError(f"flattening collision: {target}")
            targets.add(target.casefold())
            mapping[relative.as_posix()] = target
    return mapping


def copy_legacy_derived(source: Path, destination: Path) -> None:
    mapping = legacy_mapping(source)
    for old, new in mapping.items():
        target = destination / new
        if target.exists():
            raise PingstoreError(f"capture destination already exists: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / old, target)


def display_manifest(root: Path, manifest: dict, run_id: str, *,
                     schema: str = RUN_SCHEMA) -> None:
    """Compatibility projection for the publishing engine, not authoritative provenance."""
    projected = dict(manifest)
    projected["pingstore_run_id"] = run_id
    if schema == RUN_SCHEMA:
        if manifest.get("stage") != "present":
            raise PingstoreError("only present runs can write publication metadata")
        destination = root / "export"
    elif schema == LEGACY_RUN_SCHEMA:
        destination = root / "presentation"
    else:
        raise PingstoreError(f"unsupported layout schema: {schema}")
    write_json_atomic(destination / "_manifest.json", projected)


def read_execution_manifest(root: Path) -> dict:
    return load_json(root / "export/provenance/_manifest.json")
