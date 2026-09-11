"""Historical snapshot recovery through R2.

Creation of collection campaigns and campaign archives is retired. Existing
remote snapshots remain available to these explicitly requested recovery commands.

Usage (always via uv):

    uv run python experiments/helpers/archive.py list exp022
    uv run python experiments/helpers/archive.py restore exp022 <snapshot>
    uv run python experiments/helpers/archive.py restore-campaign exp022 <snapshot> --destination <empty-dir>

Requires a configured rclone remote. PINGLAB_R2_REMOTE defaults to "r2" and
PINGLAB_R2_BUCKET defaults to "pinglab". Operations remain explicitly requested.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ARTIFACTS_ROOT = REPO / ".pingstore" / "runs"

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


def _human(n: int) -> str:
    x = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if x < 1024 or unit == "TiB":
            return f"{x:.1f} {unit}"
        x /= 1024
    return f"{x:.1f} TiB"


# ── Commands ─────────────────────────────────────────────────────────

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
    local = ARTIFACTS_ROOT / f".{slug}-restored-{sha}-r2.tmp" / "export" / "state"
    local.mkdir(parents=True, exist_ok=True)
    print(f"restoring {dest}  →  {local.relative_to(REPO)}  [sha {sha}]")
    _rclone(["copy", dest, str(local), "--exclude", MANIFEST,
             "--transfers", "16", "--checkers", "16", "--stats", "30s",
             "--stats-one-line"])
    print(f"\n✓ restored {slug} @ {sha} → {local.relative_to(REPO)}")


def cmd_restore_campaign(slug: str, snapshot_id: str, destination: Path) -> None:
    dest = _dest(slug, snapshot_id)
    if not _remote_dir_exists(dest):
        raise SystemExit(f"no snapshot at {dest}")
    local = destination.resolve()
    if local.exists() and any(local.iterdir()):
        raise SystemExit(f"restore destination must be absent or empty: {local}")
    local.mkdir(parents=True, exist_ok=True)
    _rclone(["copy", dest, str(local), "--exclude", MANIFEST,
             "--transfers", "16", "--checkers", "16", "--stats", "30s",
             "--stats-one-line"])
    _rclone(["check", str(local), dest, "--exclude", MANIFEST, "--download"])
    print(f"\n✓ restored {slug} campaign snapshot {snapshot_id} → {local}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Ad-hoc provenance-keyed backup of a run's scratch to R2.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ls = sub.add_parser("list", help="list a slug's snapshots on R2")
    ls.add_argument("slug")
    r = sub.add_parser("restore", help="pull a snapshot into a hidden Pingstore run")
    r.add_argument("slug")
    r.add_argument("sha", nargs="?", default=None, help="snapshot sha (default: latest)")
    rc = sub.add_parser("restore-campaign", help="restore a campaign snapshot separately")
    rc.add_argument("slug")
    rc.add_argument("snapshot_id")
    rc.add_argument("--destination", required=True, type=Path)
    args = ap.parse_args()

    _ensure_rclone_remote()
    if args.cmd == "list":
        cmd_list(args.slug)
    elif args.cmd == "restore":
        cmd_restore(args.slug, args.sha)
    elif args.cmd == "restore-campaign":
        cmd_restore_campaign(args.slug, args.snapshot_id, args.destination)


if __name__ == "__main__":
    sys.exit(main())
