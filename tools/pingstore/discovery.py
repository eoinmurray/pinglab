"""Read-only projection of validated runs into Demolab's discovery protocol."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from .contracts import PingstoreError, validate_operational_run_directory
from .layout import has_presentation_content, presentation_directory


def discover_runs(source: Path) -> list[dict[str, str]]:
    """Discover immediate visible runs; never select, copy, or modify a run.

    Validate every candidate before returning anything. Demolab currently has no
    separate validation callback, so metadata-only discovery would let it consume
    unverified payloads. Hidden entries and symlink candidates are never followed.
    """
    source = source.expanduser().absolute()
    if any(path.is_symlink() for path in (source, *source.parents)):
        raise PingstoreError(f"discovery source must not use symlinks: {source}")
    if not source.is_dir():
        raise PingstoreError(
            f"discovery source must be an existing runs directory: {source}"
        )

    records = []
    for directory in sorted(source.iterdir()):
        if (
            directory.name.startswith(".")
            or directory.is_symlink()
            or not directory.is_dir()
        ):
            continue
        try:
            run = validate_operational_run_directory(directory)
            # The run contract requires a string; Demolab additionally requires
            # a parseable, timezone-aware timestamp. Never substitute file times.
            created_at = datetime.fromisoformat(
                run["created_at"].replace("Z", "+00:00")
            )
            if created_at.utcoffset() is None:
                raise PingstoreError("created_at must include a timezone")
            timestamp = created_at.astimezone(timezone.utc).isoformat()
        except (OSError, ValueError, OverflowError) as exc:
            raise PingstoreError(f"cannot discover {directory.name}: {exc}") from exc
        files = presentation_directory(directory, run)
        if files is None or not has_presentation_content(files):
            continue
        records.append(
            {
                "id": run["run_id"],
                "experiment": run["experiment"],
                "label": run["run_id"],
                "created_at": timestamp,
                "presentation": files.relative_to(source).as_posix(),
            }
        )
    return records
