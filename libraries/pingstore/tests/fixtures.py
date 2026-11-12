"""Scientific fixture writers, separate from the storage integrity contract.

Experiment tests can keep their scientific execution dictionaries in details.json
while exercising the same small run metadata used in production. Explicit payload
hashes here prove that a consumer left its input unchanged; storage never calls
this test helper.
"""

import hashlib
import json
from pathlib import Path

from pingstore.core import load_json, write_json_atomic
from pingstore.migration import convert


def payload_digest(directory):
    rows = []
    for path in sorted((Path(directory) / "export").rglob("*")):
        if path.is_file():
            rows.append(
                (
                    path.relative_to(directory).as_posix(),
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                )
            )
    return "sha256:" + hashlib.sha256(json.dumps(rows).encode()).hexdigest()


def read_details(directory):
    return load_json(Path(directory) / "details.json")


def write_fixture_json(path, value):
    path = Path(path)
    if (
        path.name in {"details.json", "run.json"}
        and isinstance(value, dict)
        and "run_id" in value
    ):
        if value.get("schema") in {"pingstore.run/v2", "pingstore.run/v3"}:
            write_json_atomic(path.with_name("run.json"), value)
            return
        original = dict(value)
        original.setdefault(
            "created_at",
            original.get("execution", {}).get("completed_at", "2026-11-12T12:00:00Z"),
        )
        record, details = convert(original)
        write_json_atomic(path.with_name("run.json"), record)
        write_json_atomic(path.with_name("details.json"), details)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
