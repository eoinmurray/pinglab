"""Evict verified local backup copies without deleting remote scientific history.

Unlike prune, eviction may remove report-used runs. It requires the public
catalog and a byte-for-byte remote check, retains local protections and pending
inputs, and persists allocation counters before removing any local directory.
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
import time
from http.client import HTTPException
from pathlib import Path
from urllib.request import Request, urlopen

from .allocation import allocation_watermarks, persist_watermarks
from .backup import file_list, local_files, rclone
from .catalog import read_catalog
from .contracts import PingstoreError, file_sha256, validate_operational_run_directory
from .prune import _hidden_inputs, _plan_hash
from .stages import operation_lock
from .usage import ancestry


def build_plan(
    repo: Path,
    destination: str,
    run_ids=None,
    asset_base="https://assets.eoinmurray.info",
):
    repo = repo.resolve()
    source = repo / ".pingstore/runs"
    catalog = read_catalog(destination)
    request = Request(
        asset_base.rstrip("/") + "/pingstore/catalog.json",
        headers={"User-Agent": "Pinglab-Pingstore/1.0"},
    )
    for attempt in range(3):
        try:
            with urlopen(request, timeout=30) as response:
                public = json.load(response)
            break
        except (OSError, HTTPException, json.JSONDecodeError) as error:
            if attempt == 2:
                raise PingstoreError(
                    f"public catalog could not be read: {error}"
                ) from error
            time.sleep(0.2 * (attempt + 1))
    if public != catalog:
        raise PingstoreError("public and authenticated backup catalogs differ")
    records = {identity: entry["record"] for identity, entry in catalog["runs"].items()}
    local_ids = {p.name for p in source.iterdir() if not p.name.startswith(".")}
    if run_ids and set(run_ids) - local_ids:
        raise PingstoreError("eviction selection contains a missing local run")
    selected = set(run_ids) if run_ids else local_ids
    reasons = {}
    from collections import defaultdict

    pending = defaultdict(set)
    local_records = {}
    for identity in sorted(local_ids):
        local_records[identity] = validate_operational_run_directory(source / identity)
    _hidden_inputs(source, local_records, pending)
    combined = {**records, **local_records}
    protected = ancestry(
        combined,
        {id for id in local_ids if (source / id / ".protected").exists()}
        | set(pending),
    )
    rows = []
    for identity in sorted(selected):
        entry = catalog["runs"].get(identity)
        if identity in protected:
            reasons[identity] = "protected or required by unfinished work"
            continue
        if entry is None:
            reasons[identity] = "not in verified R2 catalog"
            continue
        files = {
            name.removeprefix(identity + "/"): {
                "bytes": size,
                "sha256": file_sha256(source / name),
            }
            for name, size in local_files(source, identity).items()
        }
        if local_records[identity] != entry["record"] or files != entry["files"]:
            raise PingstoreError(
                f"local run differs from verified backup catalog: {identity}"
            )
        rows.append(
            {
                "run_id": identity,
                "files": files,
                "bytes": sum(file["bytes"] for file in files.values()),
            }
        )
    watermarks = allocation_watermarks(source.parent)
    for identity in records:
        experiment, counter = identity.split("-")[:2]
        watermarks[experiment] = max(watermarks.get(experiment, 0), int(counter[1:]))
    plan = {
        "schema": "pingstore.evict-plan/v1",
        "root": str(repo),
        "destination": destination,
        "asset_base": asset_base,
        "selection": sorted(run_ids) if run_ids else None,
        "catalog": catalog,
        "evict": rows,
        "keep": reasons,
        "allocation_watermarks": watermarks,
    }
    return {**plan, "plan_hash": _plan_hash(plan)}


def apply_plan(
    repo: Path,
    destination: str,
    expected_hash: str,
    run_ids=None,
    asset_base="https://assets.eoinmurray.info",
):
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", expected_hash):
        raise PingstoreError("--confirm requires the full dry-run plan hash")
    source = repo.resolve() / ".pingstore/runs"
    with operation_lock(source.parent, exclusive=True):
        plan = build_plan(repo, destination, run_ids, asset_base)
        if plan["plan_hash"] != expected_hash:
            raise PingstoreError("eviction plan changed; run --dry-run again")
        if not plan["evict"]:
            return plan
        with tempfile.TemporaryDirectory(prefix="pingstore-evict-check-") as temporary:
            paths = file_list(
                Path(temporary),
                "verify.txt",
                (
                    f"{row['run_id']}/{name}"
                    for row in plan["evict"]
                    for name in row["files"]
                ),
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
        if build_plan(repo, destination, run_ids, asset_base) != plan:
            raise PingstoreError("eviction controls changed during verification")
        persist_watermarks(source.parent, plan["allocation_watermarks"])
        staging = source.parent / f".evict-{expected_hash[7:19]}"
        if staging.exists():
            raise PingstoreError("unfinished eviction staging requires recovery")
        staging.mkdir()
        moved = []
        try:
            for row in plan["evict"]:
                identity = row["run_id"]
                (source / identity).rename(staging / identity)
                moved.append(identity)
        except BaseException:
            for identity in moved:
                (staging / identity).rename(source / identity)
            staging.rmdir()
            raise
        shutil.rmtree(staging)
        return plan


def render_plan(plan):
    lines = [
        f"Plan: {plan['plan_hash']}",
        f"Evict: {len(plan['evict'])} local copies ({sum(row['bytes'] for row in plan['evict']) / 2**30:.2f} GiB)",
    ]
    lines.extend(row["run_id"] for row in plan["evict"])
    lines.extend(f"Keep {id}: {reason}" for id, reason in plan["keep"].items())
    lines.append(
        "Remote runs are retained. Confirmation re-downloads and verifies selected backup bytes."
    )
    return "\n".join(lines)
