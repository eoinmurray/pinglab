"""Per-notebook incrementing run id.

Each notebook entry has its own monotonic counter persisted at
`src/docs/public/figures/notebook/<slug>/_run.txt`. `next_run_id(slug)`
returns the next id as "rNNN" without touching disk; `persist(slug, run_id)`
writes the counter back and must be called after any wipe + re-creation of
the figures dir so the count survives.
"""
from __future__ import annotations

from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
FIGURES_ROOT = REPO / "src" / "docs" / "public" / "figures" / "notebook"
COUNTER_FILE = "_run.txt"


def _counter_path(slug: str) -> Path:
    return FIGURES_ROOT / slug / COUNTER_FILE


def _read_current(slug: str) -> int:
    path = _counter_path(slug)
    if not path.exists():
        return 0
    try:
        return int(path.read_text().strip())
    except ValueError:
        return 0


def next_run_id(slug: str) -> str:
    return f"r{_read_current(slug) + 1:03d}"


def persist(slug: str, run_id: str) -> None:
    n = int(run_id.lstrip("r"))
    path = _counter_path(slug)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{n}\n")
