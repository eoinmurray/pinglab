"""Versioned run layout and explicit adapters for legacy output producers."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from .contracts import (
    LEGACY_RUN_SCHEMA,
    PREVIOUS_RUN_SCHEMA,
    RUN_SCHEMA,
    PingstoreError,
    load_json,
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
ROLE_ALIASES = {
    "snapshot.npz": "recording.npz",
    "recordings.npz": "recording.npz",
}


def canonical_role_name(name: str) -> str:
    """Return the standard scientific role filename, preserving extensions."""
    if name in ROLE_ALIASES:
        return ROLE_ALIASES[name]
    if name.endswith("--snapshot.npz"):
        return name.removesuffix("--snapshot.npz") + "--recording.npz"
    if name.endswith("-recordings.npz"):
        return name.removesuffix("-recordings.npz") + "-recording.npz"
    return name


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


def canonical_export_relative(relative: Path, *, export_root: str = "export") -> Path:
    """Map a scientific file to a root file or candidate unit/role path."""
    parts = relative.parts
    prefix = Path(export_root).parts
    if prefix and prefix[0] == "export" and tuple(parts[: len(prefix) - 1]) == prefix[1:]:
        parts = parts[len(prefix) - 1 :]
    if len(parts) <= 1:
        return Path(canonical_role_name(parts[0]))
    if len(parts) == 2:
        return Path(parts[0]) / canonical_role_name(parts[1])
    directories, filename = list(parts[:-1]), parts[-1]
    bundle = next(
        (index for index, name in enumerate(directories) if name.endswith(".bundle")),
        None,
    )
    if bundle is None:
        return Path("--".join(directories)) / canonical_role_name(filename)
    unit = "--".join(directories[: bundle + 1])
    remainder = directories[bundle + 1 :]
    role = "--".join([*remainder, filename]) if remainder else filename
    role = canonical_role_name(role)
    return Path(unit) / role


def canonical_export_mapping(
    relatives: list[Path], *, export_root: str = "export"
) -> dict[str, str]:
    """Map files, flattening units that contain only one scientific role."""
    candidates = {
        relative.as_posix(): canonical_export_relative(
            relative, export_root=export_root
        )
        for relative in relatives
    }
    counts: dict[Path, int] = {}
    for target in candidates.values():
        if target.parent != Path("."):
            counts[target.parent] = counts.get(target.parent, 0) + 1
    mapping = {}
    for source, target in candidates.items():
        if target.parent != Path(".") and counts[target.parent] == 1:
            target = Path(f"{target.parent.name}--{target.name}")
        mapping[source] = target.as_posix()
    return mapping


def canonical_export_unit(root: Path, *parts: str | Path) -> Path:
    values = []
    for part in parts:
        values.extend(Path(part).parts)
    direct = root.joinpath(*values)
    return direct if direct.exists() else root / "--".join(values)


def canonical_export_file(root: Path, *parts: str | Path) -> Path:
    relative = Path()
    for part in parts:
        relative /= Path(part)
    direct = root / relative
    if direct.exists():
        return direct
    bundled = canonical_export_relative(relative)
    candidate = root / bundled
    if candidate.exists() or bundled.parent == Path("."):
        return candidate
    return root / f"{bundled.parent.name}--{bundled.name}"


def _rewrite_paths(value, mapping: dict[str, str]):
    if isinstance(value, dict):
        return {key: _rewrite_paths(item, mapping) for key, item in value.items()}
    if isinstance(value, list):
        return [_rewrite_paths(item, mapping) for item in value]
    if isinstance(value, str):
        if value in mapping:
            return mapping[value]
        if value.startswith("export/") and value[7:] in mapping:
            return "export/" + mapping[value[7:]]
    return value


def normalize_export_layout(directory: Path, record: dict) -> dict[str, str]:
    """Normalize a hidden or staged run without mutating a visible source run."""
    export = directory / "export"
    export_root = record.get("export_root", "export")
    files = [path for path in sorted(export.rglob("*")) if path.is_file()]
    mapping = canonical_export_mapping(
        [path.relative_to(export) for path in files], export_root=export_root
    )
    if len(set(mapping.values())) != len(mapping):
        raise PingstoreError(f"{directory.name}: canonical export paths collide")
    temporary = directory / ".normalized-export.tmp"
    if temporary.exists():
        raise PingstoreError(f"{directory.name}: stale export normalization directory")
    temporary.mkdir()
    try:
        for source in files:
            target = temporary / mapping[source.relative_to(export).as_posix()]
            target.parent.mkdir(parents=True, exist_ok=True)
            os.link(source, target)
        for target in temporary.rglob("*.json"):
            try:
                value = json.loads(target.read_text())
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            revised = _rewrite_paths(value, mapping)
            if revised != value:
                write_json_atomic(target, revised)
        shutil.rmtree(export)
        os.replace(temporary, export)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    record.pop("export_root", None)
    revised = _rewrite_paths(record, mapping)
    record.clear()
    record.update(revised)
    return mapping


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
