"""Canonical layout helpers for operational v4 runs."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from .contracts import (
    PingstoreError,
    write_json_atomic,
)

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


def initialize_layout(root: Path, experiment: str) -> None:
    (root / "export").mkdir(parents=True, exist_ok=True)
    readme = root / "README.md"
    if not readme.exists():
        readme.write_text(f"# {experiment} run\n\n## History\n\n")


def export_directory(root: Path, run: dict) -> Path:
    """Resolve scientific output from an already validated v4 record."""
    return root / "export"


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
    """Resolve publishable output; stage is authoritative, not the folder name."""
    return root / "export" if run["stage"] == "present" else None


def has_presentation_content(directory: Path) -> bool:
    return any(path.name not in RECORD_NAMES and path.stat().st_size > 0
               for path in directory.iterdir())
