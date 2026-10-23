"""Legacy PDF backup planning; public PDF uploads are disabled."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .contracts import PingstoreError, file_sha256


def build_plan(source: Path, destination: str):
    if not source.is_dir() or source.is_symlink():
        raise PingstoreError("PDF source must be an existing real directory")
    if ":" not in destination or not destination.endswith("/papers"):
        raise PingstoreError(
            "PDF destination must be an rclone remote ending in /papers"
        )
    files = {
        path.relative_to(source).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in sorted(source.rglob("*"))
        if path.is_file()
        and path.suffix.lower() == ".pdf"
        and not path.is_symlink()
        and not any(part.startswith(".") for part in path.relative_to(source).parts)
    }
    if any(
        any(c in path for c in "\\\r\n\0")
        or any(part in ("", ".", "..") for part in path.split("/"))
        for path in files
    ):
        raise PingstoreError("invalid PDF backup path")
    plan = {"source": str(source.resolve()), "destination": destination, "files": files}
    digest = hashlib.sha256(
        json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {**plan, "plan_hash": "sha256:" + digest}


def apply_plan(source: Path, destination: str, expected_hash: str):
    raise PingstoreError(
        "Public PDF uploads are disabled. Keep PDFs local and publish source links instead."
    )
