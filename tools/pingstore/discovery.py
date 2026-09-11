"""Read-only projection of validated runs into Demolab's discovery protocol."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from .contracts import PingstoreError, validate_operational_run_directory
from .layout import has_presentation_content, presentation_directory


def _validate_source(source: Path) -> Path:
    source = source.expanduser().absolute()
    if any(path.is_symlink() for path in (source, *source.parents)):
        raise PingstoreError(f"discovery source must not use symlinks: {source}")
    if not source.is_dir():
        raise PingstoreError(
            f"discovery source must be an existing runs directory: {source}"
        )
    return source


def validated_run_graph(
    source: Path, selected_ids: set[str] | None = None
) -> dict[str, dict]:
    """Validate all visible runs, or selected runs and their complete ancestry."""
    source = source.expanduser().absolute()
    if selected_ids is None:
        pending = [
            path.name
            for path in sorted(source.iterdir())
            if not path.name.startswith(".") and path.is_dir() and not path.is_symlink()
        ]
    else:
        pending = list(selected_ids)
    records = {}
    while pending:
        identity = pending.pop()
        if identity in records:
            continue
        if Path(identity).name != identity or identity.startswith("."):
            raise PingstoreError(f"unsafe run identity: {identity}")
        record = validate_operational_run_directory(source / identity)
        records[identity] = record
        if selected_ids is not None:
            pending.extend(ref["run_id"] for ref in record["inputs"].values())
    for child, record in records.items():
        for reference in record["inputs"].values():
            parent = records.get(reference["run_id"])
            if parent is None or parent["payload_digest"] != reference["payload_digest"]:
                raise PingstoreError(
                    f"{child}: missing or changed input {reference['run_id']}"
                )
    return records


def discover_store(source: Path) -> tuple[dict[str, dict], list[dict[str, str]]]:
    """Validate the complete store and project its populated present runs.

    Validate every candidate before returning anything. Demolab currently has no
    separate validation callback, so metadata-only discovery would let it consume
    unverified payloads. Hidden entries and symlink candidates are never followed.
    """
    source = _validate_source(source)
    graph = validated_run_graph(source)
    records = []
    for identity, run in sorted(graph.items()):
        directory = source / identity
        try:
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
    return graph, records


def discover_runs(source: Path) -> list[dict[str, str]]:
    """Emit Demolab discovery rows after complete store validation."""
    return discover_store(source)[1]
