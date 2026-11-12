"""Path helpers for scientific files, including existing flattened exports.

These helpers never rewrite or validate a run's payload on completion.
"""

from pathlib import Path

from .core import PingstoreError, safe_file

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


def export_directory(root: Path, run: dict) -> Path:
    """Resolve the conventional scientific output directory."""
    return root / "export"


def canonical_export_relative(relative: Path, *, export_root: str = "export") -> Path:
    """Map a scientific file to a root file or candidate unit/role path."""
    parts = relative.parts
    prefix = Path(export_root).parts
    if (
        prefix
        and prefix[0] == "export"
        and tuple(parts[: len(prefix) - 1]) == prefix[1:]
    ):
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


def restore_native(export, mapping, prefix, destination):
    """Reconstruct original names without modifying retained scientific bytes."""
    prefix = Path(prefix)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    selected = {
        source: target
        for source, target in mapping.items()
        if Path(source).is_relative_to(prefix)
    }
    if not selected:
        raise PingstoreError(f"no retained native artifact: {prefix}")
    import shutil

    for source, target in selected.items():
        relative = Path(source).relative_to(prefix)
        if (
            ".." in relative.parts
            or Path(target).is_absolute()
            or ".." in Path(target).parts
        ):
            raise PingstoreError("invalid native artifact path")
        output = destination / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(safe_file(export, target), output)
    return destination
