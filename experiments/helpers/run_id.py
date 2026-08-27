"""Per-notebook incrementing run id.

Each notebook entry has its own monotonic counter persisted at
`.artifacts/<slug>/_run.txt` (legacy fallback); completed run IDs also count. `next_run_id(slug)`
returns the next id as "rNNN" without touching disk; `persist(slug, run_id)`
writes the counter back and must be called after any wipe + re-creation of
the figures dir so the count survives.
"""

from __future__ import annotations

from pathlib import Path

from .paths import FIGURES_ROOT, current_run_number

COUNTER_FILE = "_run.txt"


def _counter_path(slug: str) -> Path:
    return FIGURES_ROOT / slug / COUNTER_FILE


def _read_current(slug: str) -> int:
    return current_run_number(slug)


def next_run_id(slug: str) -> str:
    return f"r{_read_current(slug) + 1:03d}"


def persist(slug: str, run_id: str) -> None:
    n = int(run_id.lstrip("r"))
    path = _counter_path(slug)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{n}\n")
