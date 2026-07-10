"""Run-provenance manifest for notebook figure directories.

Every notebook run stamps `_manifest.json` into its figures dir (next to
`_run.txt` and numbers.json), written by `run_dirs.prepare` at the start of
the run — so even a run that crashes mid-training leaves a record of when it
started and what code it ran. The docs site reads this sidecar to render the
per-entry status bar (last run date, commit lock, staleness) and the Methods
run-scale table; numbers.json stays the notebook-specific results summary.

Fields:
  run_id      "rNNN" monotonic per-notebook counter (helpers.run_id)
  run_at      ISO-8601 UTC timestamp of the prepare() call
  git_sha     short SHA of HEAD (PINGLAB_GIT_SHA fallback for containers)
  dirty       True if the working tree had *any* uncommitted change (global;
              informational — reproducibility keys on code_dirty/patch below)
  code_dirty  True if the run's dependency code (tools/, experiments/) had
              uncommitted changes — this is what actually threatens repro
  patch       {file, sha256, lines} of the captured working-tree diff over the
              dependency paths, or None. Present iff code_dirty and capture
              succeeded. A dirty run with a patch is REPRODUCIBLE: checkout
              git_sha, `git apply` the patch, then re-run. This is the whole
              point — you run without committing first, and provenance still
              lets a cold clone reconstruct the exact source.
  host        where the training cells executed (e.g. "local")
  scale       the runner's declared run scale (max_samples, epochs, t_ms, ...)

Experiments must not import the tools' internals, so the git capture is
deliberately duplicated from tools/snn/runlog.py rather than imported — with one difference:
sha and dirty are separate fields here, not a "(dirty)" string suffix, so
the docs build can run `git log <sha>..HEAD` without parsing.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .paths import REPO

MANIFEST_FILE = "_manifest.json"
PATCH_FILE = "_dirty.patch"
RUN_SH_FILE = "run.sh"

# The code a run reproduces from: the runners + their helpers (experiments/)
# and the reusable tools they drive (tools/). A change under these paths is what
# makes a run non-reproducible-from-HEAD; a dirty file elsewhere (e.g. writings/
# or demolab-engine/) does not affect the run, so it is deliberately excluded
# from both the dirty check and the captured patch.
DEP_PATHS = ("tools", "experiments")


def _git_ok(args: list[str]) -> str | None:
    """Run a git command that should exit 0; return stdout, or None on failure."""
    try:
        return subprocess.check_output(
            ["git", *args], cwd=REPO, stderr=subprocess.DEVNULL, timeout=5
        ).decode()
    except Exception:
        return None


def _git_diff(args: list[str]) -> str | None:
    """Run a git *diff* command (exits 1 when differences exist, which is the
    normal case) and return stdout regardless of exit code; None if git is
    unavailable. Never raises on the exit-1-means-changes convention."""
    try:
        r = subprocess.run(
            ["git", *args],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return r.stdout
    except Exception:
        return None


def git_state() -> tuple[str, bool]:
    """Return (short-sha, global-dirty); PINGLAB_GIT_SHA / "unknown" fallback."""
    sha_out = _git_ok(["rev-parse", "--short", "HEAD"])
    if sha_out is None:
        # No .git (e.g. a container): trust the env sha if provided. It names a
        # real commit, so it is not "dirty" in the diffable sense.
        return os.environ.get("PINGLAB_GIT_SHA", "unknown"), False
    dirty = bool(
        subprocess.call(
            ["git", "diff-index", "--quiet", "HEAD"],
            cwd=REPO,
            stderr=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            timeout=5,
        )
    )
    return sha_out.strip(), dirty


def _code_dirty() -> bool:
    """True if the dependency paths have uncommitted changes (tracked or new)."""
    # Tracked changes under the dep paths (exit 1 = there are changes).
    tracked = subprocess.call(
        ["git", "diff", "--quiet", "HEAD", "--", *DEP_PATHS],
        cwd=REPO,
        stderr=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        timeout=5,
    )
    if tracked != 0:
        return True
    # Untracked-but-not-ignored files under the dep paths (e.g. a brand-new
    # nbNNN.py that git diff HEAD would miss).
    others = _git_ok(
        ["ls-files", "--others", "--exclude-standard", "--", *DEP_PATHS]
    )
    return bool(others and others.strip())


def _capture_patch() -> str | None:
    """Build a single patch of all uncommitted changes under the dep paths:
    tracked edits via `git diff HEAD`, plus each untracked dep file rendered as
    a new-file diff via `git diff --no-index`. Returns the patch text, or None
    if nothing was captured. Never touches the index."""
    parts: list[str] = []

    tracked = _git_diff(["diff", "HEAD", "--", *DEP_PATHS])
    if tracked and tracked.strip():
        parts.append(tracked)

    others = _git_ok(["ls-files", "--others", "--exclude-standard", "--", *DEP_PATHS])
    for rel in (others or "").splitlines():
        rel = rel.strip()
        if not rel:
            continue
        # --no-index emits a valid new-file hunk and exits 1; _git_diff keeps
        # the stdout. Paths are given relative to REPO (the cwd).
        d = _git_diff(["diff", "--no-index", "--", "/dev/null", rel])
        if d and d.strip():
            parts.append(d)

    if not parts:
        return None
    return "\n".join(parts)


def write_run_sh(record_dir: Path, slug: str) -> Path:
    """Write an executable `run.sh` reproducer into *record_dir* (RULES §4.7).

    The committed twin of the tool's scratch `run.sh`: a cold clone checks out
    the `git_sha` recorded in `_manifest.json`, `git apply`s any `_dirty.patch`,
    then runs this to regenerate the record. Science params are hardcoded in the
    runner (only meta flags reach the CLI, gated by the arg allowlist), so the
    canonical reproducer is the bare, no-arg full run — not a replay of this
    invocation's meta flags (a `--plot-only` run must still emit a from-scratch
    reproducer). Returns the run.sh path."""
    path = record_dir / RUN_SH_FILE
    path.write_text(
        "#!/usr/bin/env sh\n"
        f"# Reproducer for {slug} — regenerates this record from a clean run.\n"
        "# Auto-generated by helpers/provenance.write_run_sh (RULES §4.7); do not edit.\n"
        'cd "$(git rev-parse --show-toplevel)"\n'
        f"exec uv run python experiments/{slug}.py\n"
    )
    path.chmod(0o755)
    return path


def write_manifest(
    figures_dir: Path,
    *,
    slug: str,
    run_id: str,
    scale: dict | None = None,
    host: str = "local",
) -> Path:
    """Write `_manifest.json` and the `run.sh` reproducer (and, for a dirty dep
    tree, `_dirty.patch`) into *figures_dir*; return the manifest path."""
    sha, dirty = git_state()

    patch_meta: dict | None = None
    code_dirty = _code_dirty()
    if code_dirty:
        patch_text = _capture_patch()
        if patch_text:
            (figures_dir / PATCH_FILE).write_text(patch_text)
            patch_meta = {
                "file": PATCH_FILE,
                "sha256": hashlib.sha256(patch_text.encode()).hexdigest(),
                "lines": patch_text.count("\n") + 1,
            }

    manifest = {
        "slug": slug,
        "run_id": run_id,
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": sha,
        "dirty": dirty,
        "code_dirty": code_dirty,
        "patch": patch_meta,
        "host": host,
        "scale": scale,
    }
    path = figures_dir / MANIFEST_FILE
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    write_run_sh(figures_dir, slug)
    return path
