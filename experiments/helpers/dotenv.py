"""A tiny, dependency-free `.env` loader for the laptop/dispatch side.

The RunPod fan-out reads credentials straight from `os.environ`
(`RUNPOD_CONTAINER_REGISTRY_AUTH_ID`, `RUNPOD_API_KEY`, the `PINGLAB_*` path
overrides). To spare the human a manual `source .env` before every dispatch,
`load_dotenv()` reads the repo-root `.env` once and populates any of those vars
that aren't already set. Standard dotenv precedence: the real environment wins
over the file (`override=False`), so an explicit `export` on the shell always
takes priority over a stale `.env` line.

Deliberately hand-rolled (KEY=VALUE, blanks/`#` comments skipped, surrounding
quotes stripped, an optional leading `export ` tolerated) so the repo gains no
new dependency — `python-dotenv` is not in the lock. This runs only on the
laptop that fires the fleet; a pod gets its env injected by `dispatch()` and has
no `.env`, so the caller gates the load behind a "not on a pod" check.
"""

from __future__ import annotations

import os
from pathlib import Path

from .paths import REPO


def _parse(text: str) -> dict[str, str]:
    """Parse `.env` text into a dict — KEY=VALUE per line, blanks and `#`
    comments skipped, an optional leading `export ` dropped, and matching
    surrounding single/double quotes stripped from the value."""
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
            val = val[1:-1]
        out[key] = val
    return out


def load_dotenv(path: Path | str | None = None, *, override: bool = False) -> list[str]:
    """Load a `.env` file into `os.environ`; return the keys actually applied.

    `path` defaults to the repo-root `.env` (resolved from `REPO`, so it's found
    regardless of CWD). Missing file ⇒ no-op (returns `[]`). With `override=False`
    (the default) a variable already present in the environment is left untouched
    — env wins over the file — matching standard dotenv precedence.
    """
    env_path = Path(path) if path is not None else REPO / ".env"
    if not env_path.is_file():
        return []
    applied: list[str] = []
    for key, val in _parse(env_path.read_text()).items():
        if not override and key in os.environ:
            continue
        os.environ[key] = val
        applied.append(key)
    return applied
