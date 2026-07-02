"""Run-provenance manifest for notebook figure directories.

Every notebook run stamps `_manifest.json` into its figures dir (next to
`_run.txt` and numbers.json), written by `run_dirs.prepare` at the start of
the run — so even a run that crashes mid-training leaves a record of when it
started and what code it ran. The docs site reads this sidecar to render the
per-entry status bar (last run date, commit lock, staleness) and the Methods
run-scale table; numbers.json stays the notebook-specific results summary.

Fields:
  run_id   "rNNN" monotonic per-notebook counter (helpers.run_id)
  run_at   ISO-8601 UTC timestamp of the prepare() call
  git_sha  short SHA of HEAD (PINGLAB_GIT_SHA fallback for containers)
  dirty    True if the working tree had uncommitted changes — a dirty run is
           not reproducible from any commit, and the docs badge says so
  host     "local" or "modal:<GPU>" — where the training cells executed
  scale    the runner's declared run scale (max_samples, epochs, t_ms, dt_ms,
           batch_size, hidden, seeds, cells, ...); None until a runner
           declares one. Becomes mandatory for training notebooks once the
           tier-retirement sweep lands (house rule: run scale is declared in
           code and rendered in the entry's Methods table).

Notebooks must not import src/cli, so the git capture is deliberately
duplicated from cli/runlog.py rather than imported — with one difference:
sha and dirty are separate fields here, not a "(dirty)" string suffix, so
the docs build can run `git log <sha>..HEAD` without parsing.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

MANIFEST_FILE = "_manifest.json"


def git_state() -> tuple[str, bool]:
    """Return (short-sha, dirty); falls back to PINGLAB_GIT_SHA / "unknown"."""
    try:
        sha = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
                timeout=2,
            )
            .decode()
            .strip()
        )
        dirty = bool(
            subprocess.call(
                ["git", "diff-index", "--quiet", "HEAD"],
                stderr=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                timeout=2,
            )
        )
        return sha, dirty
    except Exception:
        # No .git (e.g. a container): trust the env sha if provided. It names
        # a real commit, so it is not "dirty" in the diffable sense.
        return os.environ.get("PINGLAB_GIT_SHA", "unknown"), False


def write_manifest(
    figures_dir: Path,
    *,
    slug: str,
    run_id: str,
    scale: dict | None = None,
    host: str = "local",
) -> Path:
    """Write `_manifest.json` into *figures_dir*; return the path."""
    sha, dirty = git_state()
    manifest = {
        "slug": slug,
        "run_id": run_id,
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": sha,
        "dirty": dirty,
        "host": host,
        "scale": scale,
    }
    path = figures_dir / MANIFEST_FILE
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path
