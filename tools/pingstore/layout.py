"""Explicit v2 layout helpers and adapters for legacy derived-output producers."""

from __future__ import annotations

import shutil
from pathlib import Path

from .contracts import PingstoreError, load_json, write_json_atomic

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


def initialize_layout(root: Path, experiment: str) -> None:
    (root / "export").mkdir(parents=True, exist_ok=True)
    (root / "presentation").mkdir(exist_ok=True)
    readme = root / "README.md"
    if not readme.exists():
        readme.write_text(
            f"# {experiment}\n\nExecution provenance is in `run.json`. Scientific results and execution\nrecords are in `export/`; copyable publication inputs are in `presentation/`.\n"
        )


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


def display_manifest(root: Path, manifest: dict, run_id: str) -> None:
    """Compatibility projection for the publishing engine, not authoritative provenance."""
    projected = dict(manifest)
    projected["pingstore_run_id"] = run_id
    write_json_atomic(root / "presentation/_manifest.json", projected)


def read_execution_manifest(root: Path) -> dict:
    return load_json(root / "export/provenance/_manifest.json")
