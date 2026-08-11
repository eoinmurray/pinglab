"""Creation of isolated, provenance-bound run directories."""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .contract import (
    CONTRACT_VERSION,
    ContractError,
    validate_run_manifest,
    write_json_atomic,
)

RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _git_value(cwd: Path, *args: str) -> str | None:
    result = subprocess.run(
        ["git", *args], cwd=cwd, check=False, capture_output=True, text=True
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def source_provenance(cwd: Path, lockfile_name: str = "uv.lock") -> dict:
    """Capture current Git and lockfile identity without requiring a clean tree."""
    cwd = cwd.resolve()
    root_text = _git_value(cwd, "rev-parse", "--show-toplevel")
    if root_text is None:
        return {"git_commit": None, "git_clean": None, "lockfile": None}

    repository = Path(root_text)
    commit = _git_value(repository, "rev-parse", "HEAD")
    status = _git_value(repository, "status", "--porcelain", "--untracked-files=normal")
    lockfile = repository / lockfile_name
    lock = None
    if lockfile.is_file():
        lock = {
            "path": lockfile.relative_to(repository).as_posix(),
            "sha256": _sha256(lockfile),
        }
    return {
        "git_commit": commit,
        "git_clean": status == "" if status is not None else None,
        "lockfile": lock,
    }


def initialize_run(
    root: Path,
    *,
    run_id: str,
    kind: str,
    experiment: str | None,
    collection: str | None,
    command: list[str],
    upstream: list[str] | None = None,
    provenance_notes: str = "",
    repository: Path | None = None,
) -> dict:
    """Create a new isolated run root, refusing every pre-existing destination."""
    root = root.resolve()
    if root.exists():
        raise ContractError(f"run destination already exists: {root}")
    if not RUN_ID_RE.fullmatch(run_id):
        raise ContractError(
            "run ID must start with an alphanumeric character and contain only "
            "letters, digits, dot, underscore, or hyphen"
        )
    if kind not in {"adhoc", "campaign"}:
        raise ContractError("init kind must be 'adhoc' or 'campaign'")
    if kind == "adhoc" and (not experiment or collection is not None):
        raise ContractError("an ad-hoc run requires only --experiment")
    if kind == "campaign" and (not collection or experiment is not None):
        raise ContractError("a campaign run requires only --collection")
    if not command or not all(isinstance(part, str) and part for part in command):
        raise ContractError("init requires a non-empty execution command")

    manifest = {
        "contract_version": CONTRACT_VERSION,
        "run_id": run_id,
        "kind": kind,
        "status": "planned",
        "created_at_utc": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "source": source_provenance(repository or Path.cwd()),
        "execution": {
            "experiment": experiment,
            "collection": collection,
            "command": command,
        },
        "upstream": list(upstream or []),
        "archive": None,
        "provenance_notes": provenance_notes,
    }
    validate_run_manifest(manifest)

    try:
        if kind == "adhoc":
            assert experiment is not None
            (root / "state").mkdir(parents=True)
            (root / "derived" / "artifacts" / "data" / experiment).mkdir(parents=True)
            (root / "logs").mkdir()
        else:
            (root / "exp022").mkdir(parents=True)
            (root / "downstream").mkdir()
            (root / "derived" / "artifacts").mkdir(parents=True)
            (root / "logs").mkdir()
        write_json_atomic(root / "run.json", manifest)
    except Exception:
        shutil.rmtree(root, ignore_errors=True)
        raise
    return manifest
