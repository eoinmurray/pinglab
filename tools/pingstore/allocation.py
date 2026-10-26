"""Durable allocation counters, independent of scientific run retention."""

from __future__ import annotations

import os
import re
from pathlib import Path

from .contracts import EXPERIMENT_RE, PingstoreError, load_json, write_json_atomic

COUNTER_RE = re.compile(
    r"^\.?(exp[0-9]{3})-(?:(?:compute|analyse|present)-)?r([0-9]+)-"
)


def allocation_watermarks(store: Path) -> dict[str, int]:
    directory = store / ".allocation"
    counters = {}
    if directory.is_symlink() or (directory.exists() and not directory.is_dir()):
        raise PingstoreError("allocation state must be a real directory")
    if directory.exists():
        for path in sorted(directory.iterdir()):
            if (
                path.is_symlink()
                or not path.is_file()
                or path.suffix != ".json"
                or not EXPERIMENT_RE.fullmatch(path.stem)
            ):
                raise PingstoreError(f"invalid allocation state: {path}")
            value = load_json(path).get("high_watermark")
            if type(value) is not int or value < 0:
                raise PingstoreError(f"invalid allocation counter: {path}")
            counters[path.stem] = value
    runs = store / "runs"
    if runs.exists():
        for path in runs.iterdir():
            if match := COUNTER_RE.match(path.name):
                experiment, number = match.groups()
                counters[experiment] = max(counters.get(experiment, 0), int(number))
    return dict(sorted(counters.items()))


def persist_watermarks(store: Path, counters: dict[str, int]) -> None:
    """Caller holds the allocation lock or exclusive operation lock."""
    for experiment, number in counters.items():
        write_json_atomic(
            store / ".allocation" / f"{experiment}.json", {"high_watermark": number}
        )
    if counters:
        descriptor = os.open(store / ".allocation", os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
