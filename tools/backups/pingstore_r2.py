"""Append verified complete Pingstore runs to an R2 backup using rclone."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from pingstore.contracts import PingstoreError, file_sha256
from pingstore.discovery import discover_store
from pingstore.stages import operation_lock

REPO = Path(__file__).resolve().parents[2]


def rclone(*arguments, capture=False):
    return subprocess.run(
        ["rclone", *map(str, arguments), "--contimeout", "15s", "--timeout", "2m"],
        check=True,
        stdout=subprocess.PIPE if capture else None,
    ).stdout


def inventory(destination):
    rows = json.loads(
        rclone(
            "lsjson",
            destination,
            "--recursive",
            "--files-only",
            "--fast-list",
            "--no-modtime",
            "--no-mimetype",
            capture=True,
        )
    )
    result = {}
    for row in rows:
        path = row["Path"]
        if path in result or "\n" in path or "\r" in path:
            raise PingstoreError("invalid or duplicate remote object path")
        result[path] = row["Size"]
    return result


def local_files(source, identity):
    directory = source / identity
    files = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            relative = path.relative_to(source).as_posix()
            if "\n" in relative or "\r" in relative:
                raise PingstoreError("newline in backup file path")
            files[relative] = path.stat().st_size
    return files


def manifest_levels(records, new_ids):
    remaining = set(new_ids)
    levels = []
    while remaining:
        ready = sorted(
            identity
            for identity in remaining
            if not {ref["run_id"] for ref in records[identity]["inputs"].values()}
            & remaining
        )
        if not ready:
            raise PingstoreError("run inputs contain a cycle")
        levels.append(ready)
        remaining.difference_update(ready)
    return levels


def build_plan(source, destination):
    records, _ = discover_store(source)
    remote = inventory(destination)
    rows, new_ids, existing_ids = [], [], []
    for identity, record in sorted(records.items()):
        files = local_files(source, identity)
        remote_files = {
            path: size
            for path, size in remote.items()
            if path.startswith(identity + "/")
        }
        for path, size in remote_files.items():
            if path not in files or files[path] != size:
                raise PingstoreError(
                    f"{identity}: remote file inventory conflicts with local run"
                )
        if identity + "/run.json" in remote:
            if remote_files != files:
                raise PingstoreError(
                    f"{identity}: completed remote backup has missing files"
                )
            for name in ("run.json", "README.md"):
                data = rclone(
                    "cat", destination + "/" + identity + "/" + name, capture=True
                )
                if data != (source / identity / name).read_bytes():
                    raise PingstoreError(
                        f"{identity}: backed-up {name} differs; overwrite refused"
                    )
            existing_ids.append(identity)
        else:
            new_ids.append(identity)
        rows.append(
            {
                "run_id": identity,
                "payload_digest": record["payload_digest"],
                "run_json_sha256": file_sha256(source / identity / "run.json"),
                "readme_sha256": file_sha256(source / identity / "README.md"),
                "files": files,
            }
        )
    levels = manifest_levels(records, new_ids)
    plan = {
        "source": str(source),
        "destination": destination,
        "runs": rows,
        "new_runs": new_ids,
        "existing_runs": existing_ids,
        "remote_inventory": remote,
        "manifest_levels": levels,
    }
    digest = hashlib.sha256(
        json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {**plan, "plan_hash": "sha256:" + digest}


def file_list(directory, name, paths):
    path = directory / name
    path.write_text("".join(value + "\n" for value in sorted(paths)))
    return path


def copy_files(source, destination, paths):
    rclone(
        "copy",
        source,
        destination,
        "--files-from-raw",
        paths,
        "--immutable",
        "--checksum",
        "--no-update-modtime",
        "--transfers",
        "4",
        "--checkers",
        "4",
        "--stats",
        "30s",
        "--stats-one-line",
        "--log-level",
        "INFO",
    )


def apply_plan(source, destination, plan):
    if not plan["new_runs"]:
        print(
            f"Already backed up: {len(plan['existing_runs'])} matching runs. Nothing uploaded."
        )
        return
    expected = {row["run_id"]: row for row in plan["runs"]}
    new_files = {
        path: size
        for identity in plan["new_runs"]
        for path, size in expected[identity]["files"].items()
    }
    payload_files = set(new_files) - {
        identity + "/run.json" for identity in plan["new_runs"]
    }
    with tempfile.TemporaryDirectory(prefix="pingstore-r2-backup-") as temporary:
        scratch = Path(temporary)
        paths = file_list(scratch, "payload-files.txt", payload_files)
        print(
            f"Uploading {len(plan['new_runs'])} new runs; run.json completion records follow verification.",
            flush=True,
        )
        copy_files(source, destination, paths)
        print(
            "Verifying uploaded scientific files and README bytes by downloading and comparing them.",
            flush=True,
        )
        rclone(
            "check",
            source,
            destination,
            "--files-from-raw",
            paths,
            "--download",
            "--checkers",
            "4",
            "--stats",
            "30s",
            "--stats-one-line",
            "--log-level",
            "INFO",
        )
        records, _ = discover_store(source)
        for identity in plan["new_runs"]:
            row = expected[identity]
            if (
                records[identity]["payload_digest"] != row["payload_digest"]
                or file_sha256(source / identity / "run.json") != row["run_json_sha256"]
                or file_sha256(source / identity / "README.md") != row["readme_sha256"]
            ):
                raise PingstoreError(
                    f"{identity}: local run changed during backup; completion refused"
                )
        uploaded = inventory(destination)
        for identity in plan["new_runs"]:
            actual = {
                path: size
                for path, size in uploaded.items()
                if path.startswith(identity + "/")
            }
            expected_payload = {
                path: size
                for path, size in expected[identity]["files"].items()
                if path in payload_files
            }
            if actual != expected_payload:
                raise PingstoreError(
                    f"{identity}: remote upload changed before completion"
                )
        for index, level in enumerate(plan["manifest_levels"]):
            manifests = file_list(
                scratch,
                f"manifests-{index}.txt",
                (identity + "/run.json" for identity in level),
            )
            copy_files(source, destination, manifests)
        manifests = file_list(
            scratch,
            "all-manifests.txt",
            (identity + "/run.json" for identity in plan["new_runs"]),
        )
        rclone(
            "check",
            source,
            destination,
            "--files-from-raw",
            manifests,
            "--download",
            "--checkers",
            "4",
            "--log-level",
            "INFO",
        )
        uploaded = inventory(destination)
        for identity in plan["new_runs"]:
            actual = {
                path: size
                for path, size in uploaded.items()
                if path.startswith(identity + "/")
            }
            if actual != expected[identity]["files"]:
                raise PingstoreError("final backup inventory differs: " + identity)
    print(
        f"Backed up and verified {len(plan['new_runs'])} new runs; {len(plan['existing_runs'])} existing runs unchanged."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=REPO / ".pingstore/runs")
    remote = os.environ.get("PINGLAB_R2_REMOTE", "r2")
    bucket = os.environ.get("PINGLAB_R2_BUCKET", "pinglab")
    parser.add_argument("--destination", default=f"{remote}:{bucket}/pingstore/runs")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--dry-run", action="store_true")
    action.add_argument("--confirm", metavar="PLAN_HASH")
    args = parser.parse_args()
    source = args.source.expanduser().absolute()
    if not source.is_dir():
        parser.error("source must be an existing Pingstore runs directory")
    destination = args.destination.rstrip("/")
    if ":" not in destination or not destination.endswith("/pingstore/runs"):
        parser.error("destination must be an rclone remote ending in /pingstore/runs")
    try:
        with operation_lock(source.parent, exclusive=False):
            plan = build_plan(source, destination)
            if args.dry_run:
                print(f"Destination: {destination}")
                print(
                    f"New runs: {len(plan['new_runs'])}; matching existing runs: {len(plan['existing_runs'])}"
                )
                size = sum(
                    sum(row["files"].values())
                    for row in plan["runs"]
                    if row["run_id"] in plan["new_runs"]
                )
                print(f"New-run bytes: {size} ({size / 2**30:.2f} GiB)")
                for identity in plan["new_runs"]:
                    print(identity)
                print(f"Plan: {plan['plan_hash']}")
            else:
                if plan["plan_hash"] != args.confirm:
                    raise PingstoreError("backup plan changed; run --dry-run again")
                apply_plan(source, destination, plan)
        return 0
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"Pingstore R2 backup failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
