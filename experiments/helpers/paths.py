"""Legacy runner scratch paths and retained Pingstore identity counters.

Completed outputs live in validated Pingstore runs. Preview and publication
resolve selected present exports directly; no active artifact view exists.
New execution code uses pingstore.stages rather than these legacy scratch paths.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from pingstore.stages import execution_origin, make_legacy_run_id

REPO = Path(__file__).resolve().parents[2]
RUNS_ROOT = REPO / ".pingstore" / "runs"

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
    active_artifacts = (REPO / ".artifacts").resolve()
    if resolved[1] == active_artifacts or active_artifacts in resolved[1].parents:
        raise RuntimeError(
            "isolated derived output cannot live under repository .artifacts/"
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
    identity = f"r{current_run_number(slug) + 1:03d}"
    run_id = make_legacy_run_id(slug, identity, execution_origin())
    temporary = RUNS_ROOT / f".{run_id}.tmp"
    return RunnerPaths(
        state=temporary / "export" / "state",
        derived=temporary / "presentation",
        logs=temporary / "export" / "state" / "logs",
        isolated=False,
    )


def current_run_number(slug: str) -> int:
    """Include retained Pingstore identities without a publication-view fallback."""
    import re

    number = 0
    for path in RUNS_ROOT.glob(f"{slug}-r*-*"):
        match = re.match(rf"{slug}-r(\d+)-", path.name)
        if match:
            number = max(number, int(match.group(1)))
    # A colliding hidden directory is rejected by prepare(), not reused.
    return number


def artifacts_and_figures(slug: str) -> tuple[Path, Path]:
    """Return (artifacts_dir, figures_dir) for a notebook slug (e.g. "nb024")."""
    paths = runner_paths(slug)
    return paths.state, paths.derived
