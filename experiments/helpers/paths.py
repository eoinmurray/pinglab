"""Canonical artifact + figure directories for a notebook slug.

Every runner writes scratch simulation artifacts (npz, weights, logs) under
`src/artifacts/notebooks/<slug>/` (gitignored) and frozen figures + numbers.json
under `artifacts/data/<slug>/` — the demolab published-data layout the Typst
writings read from. `artifacts_and_figures(slug)` returns that pair so individual
runners don't re-spell the paths.

(The figure root used to be the Astro site's `src/docs/public/figures/notebooks/`;
it moved to `artifacts/data/` when the site migrated to Typst.)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
# Per-experiment scratch lives under temp/experiments/<slug> ("notebooks" is the
# deprecated name for this root — everything is an experiment now).
ARTIFACTS_ROOT = REPO / "temp" / "experiments"
FIGURES_ROOT = REPO / "artifacts" / "data"

STATE_ENV = "PINGLAB_RUN_STATE_DIR"
DERIVED_ENV = "PINGLAB_RUN_DERIVED_DIR"
LOG_ENV = "PINGLAB_RUN_LOG_DIR"
REQUIRE_ISOLATED_ENV = "PINGLAB_REQUIRE_ISOLATED"


@dataclass(frozen=True)
class RunnerPaths:
    state: Path
    derived: Path
    logs: Path
    isolated: bool


def _explicit_runner_paths() -> tuple[Path, Path, Path] | None:
    raw = {
        STATE_ENV: os.environ.get(STATE_ENV),
        DERIVED_ENV: os.environ.get(DERIVED_ENV),
        LOG_ENV: os.environ.get(LOG_ENV),
    }
    supplied = {name for name, value in raw.items() if value}
    if supplied and len(supplied) != len(raw):
        missing = sorted(set(raw) - supplied)
        raise RuntimeError(
            "isolated runner paths are all-or-none; missing " + ", ".join(missing)
        )
    if not supplied:
        return None

    state_raw = raw[STATE_ENV]
    derived_raw = raw[DERIVED_ENV]
    log_raw = raw[LOG_ENV]
    assert state_raw is not None and derived_raw is not None and log_raw is not None
    paths = (
        Path(state_raw).expanduser(),
        Path(derived_raw).expanduser(),
        Path(log_raw).expanduser(),
    )
    if not all(path.is_absolute() for path in paths):
        raise RuntimeError("isolated runner paths must be absolute")
    resolved = (paths[0].resolve(), paths[1].resolve(), paths[2].resolve())
    if len(set(resolved)) != len(resolved):
        raise RuntimeError("isolated state, derived, and log paths must be distinct")
    active_artifacts = (REPO / "artifacts").resolve()
    if resolved[1] == active_artifacts or active_artifacts in resolved[1].parents:
        raise RuntimeError(
            "isolated derived output cannot live under repository artifacts/"
        )
    return resolved


def runner_paths(slug: str) -> RunnerPaths:
    """Resolve the standard state/derived/log interface for one runner.

    Collection orchestration supplies all three absolute paths. Ordinary direct
    invocations retain the historical local locations unless
    PINGLAB_REQUIRE_ISOLATED is set, in which case fallback is forbidden.
    """
    explicit = _explicit_runner_paths()
    if explicit is not None:
        state, derived, logs = explicit
        return RunnerPaths(state=state, derived=derived, logs=logs, isolated=True)
    if os.environ.get(REQUIRE_ISOLATED_ENV) == "1":
        raise RuntimeError(
            f"{REQUIRE_ISOLATED_ENV}=1 requires {STATE_ENV}, {DERIVED_ENV}, and {LOG_ENV}"
        )
    state = ARTIFACTS_ROOT / slug
    return RunnerPaths(
        state=state,
        derived=FIGURES_ROOT / slug,
        logs=state / "logs",
        isolated=False,
    )


def artifacts_and_figures(slug: str) -> tuple[Path, Path]:
    """Return (artifacts_dir, figures_dir) for a notebook slug (e.g. "nb024")."""
    paths = runner_paths(slug)
    return paths.state, paths.derived


def log_runner_event(slug: str, event: str, **fields: object) -> None:
    """Append a compact lifecycle event beneath the runner's explicit log root."""
    paths = runner_paths(slug)
    paths.logs.mkdir(parents=True, exist_ok=True)
    record = {"event": event, "experiment": slug, **fields}
    with (paths.logs / f"{slug}.jsonl").open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
