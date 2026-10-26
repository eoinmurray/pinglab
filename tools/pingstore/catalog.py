"""Publish a checksummed public catalog only after remote bytes are verified.

The catalog is a small discovery index, not another scientific export. It keeps
remote-only runs discoverable and supplies hashes for on-demand website reads.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from .contracts import PingstoreError, file_sha256, load_json, validate_run

SCHEMA = "pingstore.catalog/v1"


def catalog_path(destination):
    return destination.rsplit("/", 1)[0] + "/catalog.json"


def read_catalog(destination):
    from .backup import rclone

    # Missing is allowed only when listing confirms this is the first catalog.
    parent = destination.rsplit("/", 1)[0]
    names = json.loads(rclone("lsjson", parent, "--files-only", capture=True))
    if not any(row["Path"] == "catalog.json" for row in names):
        return {"schema": SCHEMA, "runs": {}}
    value = json.loads(rclone("cat", catalog_path(destination), capture=True))
    validate_catalog(value)
    return value


def _check_catalog_entries(*, value):
    """Authenticate each catalog record and its declared payload inventory."""
    for identity, entry in value["runs"].items():
        record = validate_run(entry["record"])
        if record["run_id"] != identity or not record["execution"].get("completed_at"):
            raise PingstoreError("invalid catalog run identity or completion")
        files = entry["files"]
        if "run.json" not in files or "README.md" not in files:
            raise PingstoreError("catalog run is incomplete")
        payload = []
        for name, fingerprint in sorted(
            files.items(), key=lambda item: Path(item[0]).parts
        ):
            if (
                any(part in ("", ".", "..") for part in name.split("/"))
                or any(c in name for c in "\\\r\n\0")
                or type(fingerprint["bytes"]) is not int
                or fingerprint["bytes"] < 0
                or len(fingerprint["sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in fingerprint["sha256"])
            ):
                raise PingstoreError("invalid catalog file")
            if name.startswith("export/"):
                payload.append(
                    dict(
                        path=name,
                        size_bytes=fingerprint["bytes"],
                        sha256=fingerprint["sha256"],
                    )
                )
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if record["payload_digest"] != "sha256:" + digest:
            raise PingstoreError(f"catalog payload inventory mismatch: {identity}")
        for reference in record["inputs"].values():
            parent = value["runs"].get(reference["run_id"])
            if (
                parent is None
                or parent["record"]["payload_digest"] != reference["payload_digest"]
            ):
                raise PingstoreError(f"catalog missing or changed ancestor: {identity}")


def validate_catalog(value):
    if value.get("schema") != SCHEMA or not isinstance(value.get("runs"), dict):
        raise PingstoreError("invalid R2 catalog")
    _check_catalog_entries(value=value)

    visiting, visited = set(), set()

    def visit(identity):
        if identity in visiting:
            raise PingstoreError("catalog inputs contain a cycle")
        if identity in visited:
            return
        visiting.add(identity)
        for reference in value["runs"][identity]["record"]["inputs"].values():
            visit(reference["run_id"])
        visiting.remove(identity)
        visited.add(identity)

    for identity in value["runs"]:
        visit(identity)


def publish_catalog(source: Path, destination: str, plan: dict):
    from .backup import file_list, inventory, rclone

    previous = read_catalog(destination)
    entries = dict(previous["runs"])
    verify = []
    for row in plan["runs"]:
        identity = row["run_id"]
        directory = source / identity
        files = {
            name.removeprefix(identity + "/"): {
                "bytes": size,
                "sha256": file_sha256(source / name),
            }
            for name, size in row["files"].items()
        }
        entry = {"record": load_json(directory / "run.json"), "files": files}
        if identity in entries and entries[identity] != entry:
            raise PingstoreError(
                f"cataloged run changed: {identity}; overwrite refused"
            )
        if identity not in entries:
            # New uploads were verified by backup.apply_plan. Bootstrap existing
            # backups by checking their bytes too, not merely file sizes.
            if identity in plan["existing_runs"]:
                verify.extend(row["files"])
            entries[identity] = entry
    with tempfile.TemporaryDirectory(prefix="pingstore-catalog-") as temporary:
        scratch = Path(temporary)
        if verify:
            paths = file_list(scratch, "verify.txt", verify)
            print(
                f"Verifying {len(verify)} existing backup files before catalog publication.",
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
                "16",
                "--stats",
                "30s",
                "--stats-one-line",
                "--log-level",
                "INFO",
            )
        remote = inventory(destination)
        remote_ids = {
            name.split("/")[0] for name in remote if name.endswith("/run.json")
        }
        for identity in sorted(remote_ids - entries.keys()):
            from .contracts import validate_run_directory

            directory = scratch / identity
            print(
                f"Validating remote-only historical run {identity} for catalog.",
                flush=True,
            )
            rclone("copy", destination + "/" + identity, directory)
            record = validate_run_directory(directory)
            files = {
                path.relative_to(directory).as_posix(): {
                    "bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
                for path in sorted(directory.rglob("*"))
                if path.is_file() and path.name != ".protected"
            }
            entries[identity] = {"record": record, "files": files}
            shutil.rmtree(directory)
        for identity, entry in entries.items():
            expected = {
                f"{identity}/{name}": item["bytes"]
                for name, item in entry["files"].items()
            }
            actual = {
                name: size
                for name, size in remote.items()
                if name.startswith(identity + "/") and name != identity + "/.protected"
            }
            if expected != actual:
                raise PingstoreError(f"catalog remote inventory changed: {identity}")
        value = {
            "schema": SCHEMA,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "runs": entries,
        }
        validate_catalog(value)
        path = scratch / "catalog.json"
        path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
        # Re-read before replacing the one mutable index. Immutable runs are never overwritten.
        if read_catalog(destination) != previous:
            raise PingstoreError("catalog changed concurrently; retry backup")
        condition = "If-None-Match: *"
        if "updated_at" in previous:
            old_bytes = rclone("cat", catalog_path(destination), capture=True)
            if json.loads(old_bytes) != previous:
                raise PingstoreError("catalog changed concurrently; retry backup")
            # R2's ETag for this small, single-PUT JSON object is its MD5.
            etag = hashlib.md5(old_bytes, usedforsecurity=False).hexdigest()
            condition = f'If-Match: "{etag}"'
        rclone(
            "copyto",
            path,
            catalog_path(destination),
            "--header-upload",
            "Cache-Control: no-cache",
            "--header-upload",
            condition,
        )
        if rclone("cat", catalog_path(destination), capture=True) != path.read_bytes():
            raise PingstoreError("catalog publication verification failed")
    print(f"Published checksummed catalog for {len(entries)} runs.", flush=True)
