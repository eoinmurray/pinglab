"""Ad-hoc, provenance-keyed backup of a run's scratch to Cloudflare R2.

The expensive, irreplaceable-cheaply inputs a run produces — most of all the
exp022 weight bank — retained under the active immutable Pingstore run and
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
    uv run python experiments/helpers/archive.py archive-campaign <campaign>/campaign.json
    uv run python experiments/helpers/archive.py list    exp022
    uv run python experiments/helpers/archive.py restore exp022            # latest snapshot
    uv run python experiments/helpers/archive.py restore exp022 cc36be1    # a specific sha
    uv run python experiments/helpers/archive.py restore-campaign exp022 <snapshot> --destination <empty-dir>

Config via env (defaults match the existing rclone remote + bucket):
    PINGLAB_R2_REMOTE  rclone remote name           (default "r2")
    PINGLAB_R2_BUCKET  bucket under that remote      (default "pinglab")

Requires `rclone` on PATH with the remote already configured (`rclone listremotes`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ARTIFACTS_ROOT = REPO / ".pingstore" / "runs"
PUBLISHED_ROOT = REPO / ".artifacts"

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
    from experiments.helpers.paths import active_run_state

    src = active_run_state(slug)
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


def _file_inventory(path: Path) -> tuple[list[dict], str]:
    files = []
    for source in sorted(item for item in path.rglob("*") if item.is_file()):
        hasher = hashlib.sha256()
        with source.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(chunk)
        files.append({
            "path": source.relative_to(path).as_posix(),
            "size_bytes": source.stat().st_size,
            "sha256": hasher.hexdigest(),
        })
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":"))
    return files, hashlib.sha256(canonical.encode()).hexdigest()


def verified_campaign_source(manifest_path: Path) -> tuple[dict, Path]:
    """Return exactly the complete external cell bank named by an exp022 manifest."""
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from experiments import exp022
    from experiments.exp022_support import campaign

    manifest = exp022._checked_manifest(
        manifest_path, allow_generated_dirty=True,
    )
    source = (Path(manifest["campaign_root"]) / "cells").resolve()
    if source == (ARTIFACTS_ROOT / "exp022").resolve():
        raise SystemExit("campaign archive source must not fall back to the legacy local bank")
    if len(manifest["cells"]) != len(exp022.CANONICAL_CELLS):
        raise SystemExit("campaign archive requires the complete exp022 registry")
    status = campaign.summarize_status(manifest)
    if status["retry_cells"] or status["recoverable_cells"] or any(
        row["state"] != "complete" for row in status["cells"]
    ):
        raise SystemExit("campaign archive refused: every cell must be complete and valid")
    return manifest, source


def _human(n: int) -> str:
    x = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if x < 1024 or unit == "TiB":
            return f"{x:.1f} {unit}"
        x /= 1024
    return f"{x:.1f} TiB"


# ── Commands ─────────────────────────────────────────────────────────

def cmd_archive(slug: str) -> None:
    from experiments.helpers.paths import active_run_state

    src = active_run_state(slug)
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
        "source": str(src.relative_to(REPO)),
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


def cmd_archive_campaign(manifest_path: Path) -> None:
    manifest, src = verified_campaign_source(manifest_path.resolve())
    sha = manifest["repository"]["commit"]
    snapshot_id = f"{sha}-{manifest['manifest_sha256'][:12]}"
    dest = _dest("exp022", snapshot_id)
    files, tree_sha = _file_inventory(src)
    size = sum(item["size_bytes"] for item in files)
    print(f"archiving verified campaign {manifest['campaign_id']} ({len(files)} files · {_human(size)})")
    print(f"       → {dest}  [producing sha {sha}]")
    if _remote_dir_exists(dest):
        raise SystemExit(f"immutable campaign snapshot already exists: {dest}")
    snapshot = {
        "archive": "pinglab exp022 campaign snapshot",
        "slug": "exp022",
        "snapshot_id": snapshot_id,
        "campaign_id": manifest["campaign_id"],
        "campaign_manifest_sha256": manifest["manifest_sha256"],
        "producing_git_sha": sha,
        "snapshot_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_files": len(files),
        "size_bytes": size,
        "size_human": _human(size),
        "source": str(src),
        "tree_sha256": tree_sha,
        "files": files,
        "restore": (
            "uv run python experiments/helpers/archive.py restore-campaign "
            f"exp022 {snapshot_id} --destination <separate-empty-directory>"
        ),
    }
    mpath = Path(manifest["campaign_root"]) / "submissions" / f"archive-{snapshot_id}.json"
    mpath.write_text(json.dumps(snapshot, indent=2) + "\n")
    _rclone(["copy", str(src), dest, "--transfers", "16", "--checkers", "16",
             "--stats", "30s", "--stats-one-line"])
    _rclone(["copyto", str(mpath), f"{dest}/{MANIFEST}"])
    _rclone(["check", str(src), dest, "--exclude", MANIFEST, "--download"])
    remote = _read_remote_manifest("exp022", snapshot_id)
    if remote.get("tree_sha256") != tree_sha:
        raise SystemExit("remote campaign snapshot manifest hash does not match local inventory")
    print(f"\n✓ archived exp022 campaign {manifest['campaign_id']} → {dest}")


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
    a = sub.add_parser("archive", help="back up the active Pingstore run state to R2")
    a.add_argument("slug")
    ac = sub.add_parser("archive-campaign", help="archive the exact verified exp022 campaign bank")
    ac.add_argument("manifest", type=Path)
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
    if args.cmd == "archive":
        cmd_archive(args.slug)
    elif args.cmd == "archive-campaign":
        cmd_archive_campaign(args.manifest)
    elif args.cmd == "list":
        cmd_list(args.slug)
    elif args.cmd == "restore":
        cmd_restore(args.slug, args.sha)
    elif args.cmd == "restore-campaign":
        cmd_restore_campaign(args.slug, args.snapshot_id, args.destination)


if __name__ == "__main__":
    sys.exit(main())
