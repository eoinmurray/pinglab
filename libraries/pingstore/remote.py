"""Incremental rclone backup. Completion receipts are written after payload transfer.

Use a single writer for each destination, on this machine. Other machines must not
write concurrently. No scientific payload is downloaded to validate a backup.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from .core import (
    PingstoreError,
    inventory,
    operation_lock,
    run_id,
    utc_now,
    validate_run,
    validate_run_directory,
    write_json_atomic,
)

DEFAULT_DESTINATION = "r2:pinglab/pingstore-v2"


def rclone(*args, capture=False):
    try:
        result = subprocess.run(
            ["rclone", *map(str, args)],
            check=True,
            text=True,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.PIPE if capture else None,
        )
    except subprocess.CalledProcessError as error:
        # rclone documents 3/4 as directory/file not found. Authentication and
        # network failures keep their errors; they never mean an empty backup.
        if args[0] in {"lsjson", "purge", "deletefile"} and error.returncode in {3, 4}:
            return "[]" if args[0] == "lsjson" else None
        raise
    return result.stdout if capture else None


class Remote:
    def __init__(self, destination=DEFAULT_DESTINATION):
        self.destination = destination.rstrip("/")
        if (
            ":" not in self.destination
            or self.destination.endswith(":")
            or self.destination.endswith(":/")
        ):
            raise PingstoreError("destination must be a non-root rclone remote prefix")

    def path(self, *parts):
        return self.destination + "/" + "/".join(parts)

    def _json(self, *parts):
        return json.loads(rclone("cat", self.path(*parts), capture=True))

    def _write(self, parts, value):
        with tempfile.TemporaryDirectory(prefix="pingstore-metadata-") as temporary:
            path = Path(temporary) / "metadata.json"
            write_json_atomic(path, value)
            rclone("copyto", path, self.path(*parts))

    def _receipts(self):
        rows = json.loads(
            rclone("lsjson", self.path("receipts"), "--files-only", capture=True)
        )
        return {
            row["Name"][:-5]: row
            for row in rows
            if row["Name"].endswith(".json") and run_id(row["Name"][:-5])
        }

    def index(self):
        """Rebuild the small index if interrupted publication left it out of date."""
        rows = json.loads(
            rclone(
                "lsjson",
                self.destination,
                "--files-only",
                "--max-depth",
                "1",
                capture=True,
            )
        )
        value = (
            self._json("index.json")
            if any(r["Name"] == "index.json" for r in rows)
            else {"runs": {}}
        )
        receipts = self._receipts()
        previous = value.get("receipts", {})
        records = {}
        for identity, row in receipts.items():
            if previous.get(identity) == row and identity in value["runs"]:
                records[identity] = value["runs"][identity]
            else:
                records[identity] = self._json("receipts", identity + ".json")
            validate_run(records[identity]["record"])
            if records[identity]["record"]["id"] != identity:
                raise PingstoreError("backup receipt identity mismatch")
        return {"updated_at": utc_now(), "runs": records, "receipts": receipts}

    def backup(self, store, identities=None):
        with operation_lock(store.root):
            index = self.index()
            identities = list(
                dict.fromkeys(
                    identities
                    if identities is not None
                    else [r["id"] for r in store.list()]
                )
            )
            uploaded, skipped = [], []
            for identity in identities:
                source = store.get(identity)
                existing = index["runs"].get(identity)
                if existing is not None:
                    if existing["record"] != source.record:
                        raise PingstoreError(
                            f"remote ID already identifies different metadata: {identity}"
                        )
                    skipped.append(identity)
                    continue
                before = inventory(source.directory)
                # rclone's transfer checks use the backend's supported checksums;
                # multipart ETags are not treated as full-file SHA-256 values.
                rclone(
                    "copy",
                    source.directory,
                    self.path("runs", identity),
                    "--immutable",
                    "--checksum",
                    "--transfers",
                    "16",
                    "--checkers",
                    "16",
                )
                source.check_unchanged()
                if inventory(source.directory) != before:
                    raise PingstoreError(
                        f"local file inventory changed during backup: {identity}"
                    )
                receipt = {
                    "record": source.record,
                    "files": before,
                    "bytes": sum(before.values()),
                    "completed_at": utc_now(),
                }
                self._write(("receipts", identity + ".json"), receipt)
                index["runs"][identity] = receipt
                uploaded.append(identity)
            index["receipts"] = self._receipts()
            index["updated_at"] = utc_now()
            if uploaded or index["receipts"]:
                self._write(("index.json",), index)
            return {"uploaded": uploaded, "skipped": skipped}

    def restore(self, store, identities):
        with operation_lock(store.root):
            index = self.index()
            selected = list(dict.fromkeys(run_id(i) for i in identities))
            for identity in selected:
                if identity not in index["runs"]:
                    raise PingstoreError(f"no completed backup: {identity}")
                if (store.runs / identity).exists():
                    raise FileExistsError(store.runs / identity)
            restored = []
            for identity in selected:
                # Keep an interrupted transfer outside the completed-run namespace.
                temporary = store.root / "restore" / identity
                temporary.mkdir(parents=True, exist_ok=True)
                receipt = index["runs"][identity]
                rclone(
                    "copy",
                    self.path("runs", identity),
                    temporary,
                    "--immutable",
                    "--checksum",
                    "--transfers",
                    "16",
                )
                if inventory(temporary) != receipt["files"]:
                    raise PingstoreError(f"restored inventory differs: {identity}")
                store.runs.mkdir(parents=True, exist_ok=True)
                if validate_run_directory(temporary) != receipt["record"]:
                    raise PingstoreError(f"restored metadata differs: {identity}")
                store._commit(temporary, receipt["record"])
                restored.append(identity)
            return restored

    def delete(self, store, identities, *, allow_dependants=False):
        with operation_lock(store.root):
            index = self.index()
            selected = {run_id(i) for i in identities}
            dependants = {
                i: r["record"]["depends_on"]
                for i, r in index["runs"].items()
                if i not in selected
                and selected.intersection(r["record"]["depends_on"])
            }
            dependants.update(store.dependants(selected))
            if dependants and not allow_dependants:
                raise PingstoreError(
                    f"dependent runs: {dependants}; pass allow_dependants=True to proceed"
                )
            # Remove completion markers first; interrupted deletion cannot advertise a complete backup.
            for identity in sorted(selected):
                rclone("deletefile", self.path("receipts", identity + ".json"))
                index["runs"].pop(identity, None)
                index["receipts"].pop(identity, None)
                self._write(("index.json",), index)
                rclone("purge", self.path("runs", identity))
            return sorted(selected)
