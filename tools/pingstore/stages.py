"""Independent experiment executions; no stage dispatch or publication CLI."""

from __future__ import annotations

import contextlib
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .contracts import (
    EXPERIMENT_RE,
    RUN_SCHEMA,
    STAGE_ID_RE,
    PingstoreError,
    file_sha256,
    load_json,
    payload_digest,
    run_root,
    validate_operational_run_directory,
    write_json_atomic,
)
from .layout import (
    canonical_export_file,
    canonical_export_unit,
    export_directory,
    initialize_layout,
    normalize_export_layout,
    presentation_directory,
)
from .locking import operation_lock
from .native import execution_origin
from .registry import memberships

STAGES = ("compute", "analyse", "present")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class SourceRun:
    directory: Path
    record: dict

    @property
    def export(self) -> Path:
        return export_directory(self.directory, self.record)

    @property
    def outputs(self) -> Path:
        """The complete scientific export, including an explicit export_root's siblings."""
        return self.directory / "export"

    def unit(self, *parts: str | Path) -> Path:
        return canonical_export_unit(self.outputs, *parts)

    def file(self, *parts: str | Path) -> Path:
        return canonical_export_file(self.outputs, *parts)

    @property
    def presentation(self) -> Path:
        files = presentation_directory(self.directory, self.record)
        if files is None:
            raise PingstoreError(f"not a presentation source: {self.directory.name}")
        return files

    @property
    def reference(self) -> dict:
        return {
            "run_id": self.record["run_id"],
            "payload_digest": self.record["payload_digest"],
        }

    def check_unchanged(self) -> None:
        record = validate_operational_run_directory(self.directory)
        if record["payload_digest"] != self.record["payload_digest"]:
            raise PingstoreError(f"source changed during execution: {self.directory.name}")


def source_run(root: Path, run_id: str, *, stage: str | None = None,
               experiment: str | None = None, reference: dict | None = None) -> SourceRun:
    directory = run_root(root.absolute(), run_id)
    if any(path.is_symlink() for path in (directory, *directory.parents)):
        raise PingstoreError("source paths must not use symlinks")
    record = validate_operational_run_directory(directory)
    if stage is not None and record.get("stage") != stage:
        raise PingstoreError(f"{run_id} is not a {stage} run")
    if experiment is not None and record["experiment"] != experiment:
        raise PingstoreError(f"{run_id} does not belong to {experiment}")
    source = SourceRun(directory, record)
    if reference is not None and source.reference != reference:
        raise PingstoreError(f"upstream identity or checksum changed: {run_id}")
    return source


def _reserve_stage(root: Path, experiment: str, stage: str,
                   *, origin: str | None = None) -> str:
    """Atomically reserve an identity, including before scheduler submission."""
    if stage not in STAGES or not EXPERIMENT_RE.fullmatch(experiment):
        raise PingstoreError("invalid experiment or stage")
    origin = origin or execution_origin()
    if not re.fullmatch(r"[a-z0-9][a-z0-9.-]*", origin):
        raise PingstoreError("invalid execution origin")
    runs = root / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(rf"^\.?{experiment}-(?:(?:compute|analyse|present)-)?r(\d+)-")
    number = max((int(match.group(1)) for path in runs.iterdir()
                  if (match := pattern.match(path.name))), default=0)
    while True:
        number += 1
        identity = f"{experiment}-r{number:03d}-{stage}"
        directory = runs / f".{identity}.tmp"
        if (runs / identity).exists():
            continue
        try:
            directory.mkdir()
        except FileExistsError:
            continue
        initialize_layout(directory, experiment)
        write_json_atomic(directory / ".reservation.json", {
            "schema": RUN_SCHEMA,
            "run_id": identity, "experiment": experiment, "stage": stage,
            "origin": origin, "reserved_at": utc_now(),
        })
        return identity


def reserve_stage(root: Path, experiment: str, stage: str,
                  *, origin: str | None = None) -> str:
    with operation_lock(root, exclusive=False):
        return _reserve_stage(root, experiment, stage, origin=origin)


def stage_reservation(directory: Path) -> dict:
    """Read a v4 reservation without rewriting incomplete historical executions."""
    path = directory / ".reservation.json"
    if any(candidate.is_symlink() for candidate in (directory, path.parent, path)):
        raise PingstoreError("stage reservation must not use symlinks")
    if not path.is_file() and (
        (directory / "provenance/reservation.json").exists()
        or (directory / "export/provenance/reservation.json").exists()
    ):
        raise PingstoreError(
            "legacy v2/v3 reservation is historical evidence; reserve a fresh v4 run"
        )
    reservation = load_json(path)
    if reservation.get("schema") != RUN_SCHEMA:
        raise PingstoreError("stage execution requires a v4 reservation")
    match = STAGE_ID_RE.fullmatch(str(reservation.get("run_id", "")))
    if (match is None or match.group(1) != reservation.get("experiment")
            or match.group(3) != reservation.get("stage")):
        raise PingstoreError("stage execution requires a source-neutral reservation; reserve a fresh identity")
    if not re.fullmatch(r"[a-z0-9][a-z0-9.-]*", str(reservation.get("origin", ""))):
        raise PingstoreError("invalid reservation execution origin")
    return reservation


def _capture_code(repo: Path, directory: Path) -> dict:
    def git(*args: str, allowed: tuple[int, ...] = (0,)) -> str:
        result = subprocess.run(["git", *args], cwd=repo, capture_output=True,
                                text=True, timeout=30)
        if result.returncode not in allowed:
            raise PingstoreError(f"cannot capture source provenance: {result.stderr.strip()}")
        return result.stdout

    paths = ("experiments", "tools", "pyproject.toml", "uv.lock")
    commit = git("rev-parse", "HEAD").strip()
    dirty_paths = git("status", "--porcelain", "--", *paths)
    record = {"git_commit": commit, "dirty": bool(git("status", "--porcelain")),
              "code_dirty": bool(dirty_paths)}
    lock = repo / "uv.lock"
    if lock.is_file():
        record["lockfile_sha256"] = file_sha256(lock)
    return record


@dataclass
class StageRun:
    directory: Path
    record: dict

    @property
    def run_id(self) -> str:
        return self.record["run_id"]

    @property
    def export(self) -> Path:
        return self.directory / "export"

    @property
    def scratch(self) -> Path:
        """Temporary execution attachments, removed before atomic completion."""
        path = self.directory / ".scratch"
        path.mkdir(parents=True, exist_ok=True)
        return path


@contextlib.contextmanager
def _stage_run(repo: Path, experiment: str, stage: str, *,
               inputs: dict[str, SourceRun] | None = None, run_id: str | None = None,
               configuration: dict | None = None, operation: str = "execute"):
    """Complete one execution atomically; never materialize or dispatch a stage."""
    root = repo / ".pingstore"
    inputs = inputs or {}
    for source in inputs.values():
        source.check_unchanged()
    identity = run_id or reserve_stage(root, experiment, stage)
    destination = run_root(root, identity)
    directory = destination.with_name(f".{identity}.tmp")
    if destination.exists() or not directory.is_dir():
        raise PingstoreError(f"run must be an unused reserved identity: {identity}")
    reservation = stage_reservation(directory)
    if (reservation["run_id"] != identity or reservation["stage"] != stage
            or reservation["experiment"] != experiment):
        raise PingstoreError("reservation does not match execution")
    if (directory / "run.json").exists():
        raise PingstoreError("an interrupted execution needs explicit recovery or a new run")
    lock = directory / ".writer.lock"
    descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as handle:
        handle.write(f"{os.getpid()}\n")
    command = [sys.executable, *sys.argv]
    record = {
        "schema": RUN_SCHEMA, "run_id": identity, "experiment": experiment,
        "collection": memberships(repo)[experiment], "stage": stage,
        "origin": reservation["origin"], "created_at": utc_now(),
        "inputs": {name: source.reference for name, source in inputs.items()},
        "execution": {"operation": operation, "command": command,
                      "host": execution_origin(), "cwd": str(repo),
                      "started_at": utc_now(), "configuration": configuration},
        "provenance": _capture_code(repo, directory),
    }
    write_json_atomic(directory / "run.json", record)
    inputs_text = "\n".join(
        f"- `{name}`: `{source.record['run_id']}` (`{source.record['payload_digest']}`)"
        for name, source in inputs.items()
    ) or "- None"
    (directory / "README.md").write_text(
        f"# {identity}\n\n"
        f"{stage.capitalize()} run for `{experiment}`. Machine-readable details are in `run.json`.\n\n"
        f"## Inputs\n\n{inputs_text}\n\n"
        f"## History\n\n- {record['created_at']}: execution started on `{record['origin']}` "
        f"from Git commit `{record['provenance']['git_commit']}`.\n"
    )
    run = StageRun(directory, record)
    try:
        yield run
        for source in inputs.values():
            source.check_unchanged()
        record["execution"]["completed_at"] = utc_now()
        with (directory / "README.md").open("a") as handle:
            handle.write(f"- {record['execution']['completed_at']}: run completed successfully.\n")
        scratch = run.directory / ".scratch"
        if scratch.exists():
            shutil.rmtree(scratch)
        normalize_export_layout(directory, record)
        lock.unlink()
        (directory / ".reservation.json").unlink()
        record["payload_digest"] = payload_digest(directory)
        write_json_atomic(directory / "run.json", record)
        validate_operational_run_directory(directory)
        if destination.exists():
            raise PingstoreError(f"completed destination already exists: {identity}")
        os.rename(directory, destination)
        run.directory = destination
    except BaseException:
        print(f"[incomplete] {directory}", file=sys.stderr)
        raise
    print(identity)


@contextlib.contextmanager
def stage_run(repo: Path, experiment: str, stage: str, *,
              inputs: dict[str, SourceRun] | None = None, run_id: str | None = None,
              configuration: dict | None = None, operation: str = "execute"):
    with operation_lock(repo / ".pingstore", exclusive=False):
        with _stage_run(repo, experiment, stage, inputs=inputs, run_id=run_id,
                        configuration=configuration, operation=operation) as run:
            yield run
