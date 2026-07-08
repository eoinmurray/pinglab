"""Ad-hoc, provenance-keyed backup of a run's scratch to Cloudflare R2.

The expensive, irreplaceable-cheaply inputs a run produces — most of all the
exp022 weight bank — live only under `temp/experiments/<slug>/` (gitignored) and
on the mutable RunPod network volume. Git protects the *derived* published
figures, not these *sources*. This is a deliberately manual tool: you decide
which runs matter and archive them by hand, rather than backing up everything
RunPod ever writes (most of which is regenerable scratch).

A snapshot is keyed by the commit that PRODUCED the run, read from the run's own
`config.json` sidecars (fallback: the published `_manifest.json`, then HEAD):

    r2:<bucket>/archive/<slug>/<producing-sha>/

Keying by the producing sha makes snapshots immutable per-commit: re-archiving a
partially-wiped bank lands under a *different* sha and can never overwrite a good
one. And archive uses `rclone copy` (never `sync`), so a partial local tree can
only ADD to a snapshot — it can never delete objects already on R2. Those two
properties close the footgun that let a gutted bank clobber good data.

Usage (always via uv, never bare python):

    uv run python experiments/helpers/archive.py archive exp022
    uv run python experiments/helpers/archive.py list    exp022
    uv run python experiments/helpers/archive.py restore exp022            # latest snapshot
    uv run python experiments/helpers/archive.py restore exp022 cc36be1    # a specific sha

Config via env (defaults match the existing rclone remote + bucket):
    PINGLAB_R2_REMOTE  rclone remote name           (default "r2")
    PINGLAB_R2_BUCKET  bucket under that remote      (default "pinglab")

Requires `rclone` on PATH with the remote already configured (`rclone listremotes`).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ARTIFACTS_ROOT = REPO / "temp" / "experiments"      # per-run scratch (gitignored)
PUBLISHED_ROOT = REPO / "artifacts" / "data"        # per-run figures + _manifest.json

REMOTE = os.environ.get("PINGLAB_R2_REMOTE", "r2")
BUCKET = os.environ.get("PINGLAB_R2_BUCKET", "pinglab")
PREFIX = "archive"
MANIFEST = "MANIFEST.json"


# ── rclone plumbing ──────────────────────────────────────────────────

def _dest(slug: str, sha: str) -> str:
    return f"{REMOTE}:{BUCKET}/{PREFIX}/{slug}/{sha}"


def _rclone(args: list[str], *, capture: bool = False, check: bool = True) -> str:
    """Run an rclone subcommand. Streams to the terminal unless capture=True."""
    cmd = ["rclone", *args]
    if capture:
        p = subprocess.run(cmd, capture_output=True, text=True)
        if check and p.returncode != 0:
            raise SystemExit(f"$ {' '.join(cmd)}\n{p.stdout}\n{p.stderr}")
        return p.stdout
    p = subprocess.run(cmd)
    if check and p.returncode != 0:
        raise SystemExit(f"rclone failed ({p.returncode}): {' '.join(cmd)}")
    return ""


def _ensure_rclone_remote() -> None:
    try:
        remotes = _rclone(["listremotes"], capture=True).split()
    except FileNotFoundError:
        raise SystemExit("rclone not found on PATH — install it or check your shell.")
    if f"{REMOTE}:" not in remotes:
        raise SystemExit(
            f"rclone remote {REMOTE!r} not configured. Have: {remotes or '(none)'}. "
            f"Set PINGLAB_R2_REMOTE or run `rclone config`.")


def _remote_dir_exists(path: str) -> bool:
    out = _rclone(["lsf", path], capture=True, check=False)
    return bool(out.strip())


# ── Provenance: the commit that produced the run ─────────────────────

def _producing_sha(slug: str) -> str:
    """The commit that produced <slug>'s scratch: the modal git_sha across its
    config.json sidecars, else the published _manifest.json, else HEAD."""
    src = ARTIFACTS_ROOT / slug
    shas: Counter[str] = Counter()
    for cfg in src.rglob("config.json"):
        try:
            s = json.loads(cfg.read_text()).get("git_sha")
        except Exception:  # noqa: BLE001 — a stray unreadable sidecar must not block
            s = None
        if s:
            shas[str(s)] += 1
    if shas:
        return shas.most_common(1)[0][0]

    manifest = PUBLISHED_ROOT / slug / "_manifest.json"
    if manifest.exists():
        try:
            s = json.loads(manifest.read_text()).get("git_sha")
            if s:
                return str(s)
        except Exception:  # noqa: BLE001
            pass

    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                          capture_output=True, text=True)
    if head.returncode == 0 and head.stdout.strip():
        print("  ! no git_sha in the run's sidecars — keying by HEAD "
              f"({head.stdout.strip()}); this snapshot may not match the run's code.")
        return head.stdout.strip()
    raise SystemExit(f"could not determine a producing sha for {slug!r}.")


# ── Local stats + manifest ───────────────────────────────────────────

def _local_stats(path: Path) -> tuple[int, int]:
    n, total = 0, 0
    for f in path.rglob("*"):
        if f.is_file():
            n += 1
            total += f.stat().st_size
    return n, total


def _human(n: int) -> str:
    x = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if x < 1024 or unit == "TiB":
            return f"{x:.1f} {unit}"
        x /= 1024
    return f"{x:.1f} TiB"


# ── Commands ─────────────────────────────────────────────────────────

def cmd_archive(slug: str) -> None:
    src = ARTIFACTS_ROOT / slug
    if not src.is_dir() or not any(src.iterdir()):
        raise SystemExit(f"nothing to archive: {src.relative_to(REPO)} is missing or empty.")

    sha = _producing_sha(slug)
    dest = _dest(slug, sha)
    n_files, size = _local_stats(src)
    print(f"archiving {src.relative_to(REPO)}  ({n_files} files · {_human(size)})")
    print(f"       → {dest}  [producing sha {sha}]")

    if _remote_dir_exists(dest):
        # Immutable per-commit: same sha already there. copy (never sync) only
        # adds/updates, so this is a safe idempotent top-up, never a deletion.
        print("  note: a snapshot for this sha already exists — copy will "
              "add/refresh objects only (existing objects are never deleted).")

    manifest = {
        "archive": "pinglab run snapshot",
        "slug": slug,
        "producing_git_sha": sha,
        "snapshot_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_files": n_files,
        "size_bytes": size,
        "size_human": _human(size),
        "source": f"temp/experiments/{slug}",
        "restore": f"uv run python experiments/helpers/archive.py restore {slug} {sha}",
    }
    mpath = src.parent / f"._{slug}_{sha}_manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2) + "\n")
    try:
        _rclone(["copy", str(src), dest, "--transfers", "16", "--checkers", "16",
                 "--stats", "30s", "--stats-one-line"])
        _rclone(["copyto", str(mpath), f"{dest}/{MANIFEST}"])
    finally:
        mpath.unlink(missing_ok=True)

    print("verifying (rclone check)...")
    _rclone(["check", str(src), dest, "--exclude", MANIFEST])
    print(f"\n✓ archived {slug} @ {sha} → {dest}")
    print(f"  restore: uv run python experiments/helpers/archive.py restore {slug} {sha}")


def _snapshots(slug: str) -> list[str]:
    base = f"{REMOTE}:{BUCKET}/{PREFIX}/{slug}"
    out = _rclone(["lsf", "--dirs-only", base], capture=True, check=False)
    return sorted(d.rstrip("/") for d in out.split())


def _read_remote_manifest(slug: str, sha: str) -> dict:
    raw = _rclone(["cat", f"{_dest(slug, sha)}/{MANIFEST}"], capture=True, check=False)
    try:
        return json.loads(raw)
    except Exception:  # noqa: BLE001
        return {}


def cmd_list(slug: str) -> None:
    snaps = _snapshots(slug)
    if not snaps:
        print(f"no snapshots for {slug!r} under {REMOTE}:{BUCKET}/{PREFIX}/{slug}/")
        return
    print(f"snapshots for {slug} ({REMOTE}:{BUCKET}/{PREFIX}/{slug}/):")
    for sha in snaps:
        m = _read_remote_manifest(slug, sha)
        when = m.get("snapshot_at", "?")
        size = m.get("size_human", "?")
        nf = m.get("n_files", "?")
        print(f"  {sha:<12}  {when:<25}  {nf} files · {size}")


def _latest(slug: str) -> str:
    snaps = _snapshots(slug)
    if not snaps:
        raise SystemExit(f"no snapshots for {slug!r} to restore.")
    if len(snaps) == 1:
        return snaps[0]
    dated = [(_read_remote_manifest(slug, sha).get("snapshot_at", ""), sha)
             for sha in snaps]
    dated.sort()
    return dated[-1][1]


def cmd_restore(slug: str, sha: str | None) -> None:
    sha = sha or _latest(slug)
    dest = _dest(slug, sha)
    if not _remote_dir_exists(dest):
        raise SystemExit(f"no snapshot at {dest} — run `list {slug}` to see what exists.")
    local = ARTIFACTS_ROOT / slug
    local.mkdir(parents=True, exist_ok=True)
    print(f"restoring {dest}  →  {local.relative_to(REPO)}  [sha {sha}]")
    _rclone(["copy", dest, str(local), "--exclude", MANIFEST,
             "--transfers", "16", "--checkers", "16", "--stats", "30s",
             "--stats-one-line"])
    print(f"\n✓ restored {slug} @ {sha} → {local.relative_to(REPO)}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Ad-hoc provenance-keyed backup of a run's scratch to R2.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("archive", help="back up temp/experiments/<slug> to R2")
    a.add_argument("slug")
    ls = sub.add_parser("list", help="list a slug's snapshots on R2")
    ls.add_argument("slug")
    r = sub.add_parser("restore", help="pull a snapshot back to temp/experiments/<slug>")
    r.add_argument("slug")
    r.add_argument("sha", nargs="?", default=None, help="snapshot sha (default: latest)")
    args = ap.parse_args()

    _ensure_rclone_remote()
    if args.cmd == "archive":
        cmd_archive(args.slug)
    elif args.cmd == "list":
        cmd_list(args.slug)
    elif args.cmd == "restore":
        cmd_restore(args.slug, args.sha)


if __name__ == "__main__":
    sys.exit(main())
