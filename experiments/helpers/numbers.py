"""Shared numbers.json writer — one provenance envelope for every runner.

Runners supply only their results `payload`; this owns the envelope (run_id,
git_sha, duration) so numbers.json has an identical shape across every entry.
The data-side counterpart to figsave.save_figure. Scale is NOT written here — it
lives in the runner's hardcoded SCALE constant and is stamped into the manifest
by run_dirs (helpers.provenance), keeping results (numbers.json) and run-scale
(the manifest Methods table) separate as they are today.
"""

from __future__ import annotations

import json
from pathlib import Path

from .fmt import format_duration
from .provenance import git_state

NUMBERS_FILE = "numbers.json"


def write_numbers(figures_dir: Path, *, run_id: str, duration_s: float, payload: dict) -> Path:
    """Write numbers.json into *figures_dir* with the standard envelope wrapping
    the runner's `payload`. Returns the path."""
    sha, _ = git_state()
    doc = {
        "run_id": run_id,
        "git_sha": sha,
        "duration_s": round(duration_s, 1),
        "duration": format_duration(duration_s),
        **payload,
    }
    path = Path(figures_dir) / NUMBERS_FILE
    path.write_text(json.dumps(doc, indent=2) + "\n")
    return path
