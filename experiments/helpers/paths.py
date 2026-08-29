"""Canonical artifact + figure directories for a notebook slug.

Direct runners write all state and derived output into the hidden Pingstore run
being assembled at `.pingstore/runs/.<run-id>.tmp/`. State lives beneath
`export/state/`; derived output lives directly beneath `presentation/`. On completion
the hidden directory receives `run.json` and is atomically renamed to its
immutable visible run ID. `.artifacts/<slug>/` is only a materialized publication
view consumed by Typst.

(The figure root used to be the Astro site's `src/docs/public/figures/notebooks/`;
it moved to `.artifacts/` when the site migrated to Typst.)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from pingstore.contracts import load_json, validate_run_directory
from pingstore.layout import export_directory
from pingstore.native import execution_origin, make_run_id

REPO = Path(__file__).resolve().parents[2]
FIGURES_ROOT = REPO / ".artifacts"
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
    run_id = make_run_id(slug, identity, execution_origin())
    temporary = RUNS_ROOT / f".{run_id}.tmp"
    return RunnerPaths(
        state=temporary / "export" / "state",
        derived=temporary / "presentation",
        logs=temporary / "export" / "state" / "logs",
        isolated=False,
    )


def current_run_number(slug: str) -> int:
    """Include retained identities, with a legacy view fallback."""
    import re

    try:
        number = int((FIGURES_ROOT / slug / "_run.txt").read_text().strip())
    except (FileNotFoundError, ValueError):
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


def active_run_state(slug: str) -> Path:
    """Return the immutable state directory backing the active artifact view."""
    active_manifest = FIGURES_ROOT / slug / "_manifest.json"
    if not active_manifest.is_file():
        raise FileNotFoundError(f"no active Pingstore run for {slug}")
    manifest_text = active_manifest.read_text()
    manifest = json.loads(manifest_text)
    full_id = manifest.get("pingstore_run_id")
    if full_id:
        from pingstore.contracts import run_root

        candidate = run_root(REPO / ".pingstore", full_id)
        run = validate_run_directory(candidate)
        if run["experiment"] != slug:
            raise RuntimeError(f"active run experiment differs from {slug}")
        return export_directory(candidate, run)
    identity = manifest.get("run_id")
    if not isinstance(identity, str) or not identity:
        raise RuntimeError(f"active manifest for {slug} has no run_id")
    matches = []
    for candidate in RUNS_ROOT.glob(f"{slug}-*"):
        if not (candidate / "run.json").is_file():
            continue
        for relative in ("export/provenance/_manifest.json", "export/provenance/legacy/_manifest.json"):
            stored = candidate / relative
            if stored.is_file() and load_json(stored) == manifest:
                run = validate_run_directory(candidate)
                matches.append(export_directory(candidate, run))
                break
    if len(matches) != 1:
        raise RuntimeError(f"cannot resolve active Pingstore state for {slug}")
    return matches[0]


def log_runner_event(slug: str, event: str, **fields: object) -> None:
    """Append a compact lifecycle event beneath the runner's explicit log root."""
    paths = runner_paths(slug)
    identity = fields.get("run_id")
    if not paths.isolated and identity:
        completed = RUNS_ROOT / make_run_id(slug, str(identity), execution_origin())
        if completed.exists():
            raise RuntimeError("record completion events before finalizing the immutable run")
    paths.logs.mkdir(parents=True, exist_ok=True)
    record = {"event": event, "experiment": slug, **fields}
    with (paths.logs / f"{slug}.jsonl").open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
