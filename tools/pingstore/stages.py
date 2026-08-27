"""Independent experiment executions; no stage dispatch or publication CLI."""

from __future__ import annotations

import contextlib
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .contracts import (
    EXPERIMENT_RE, RUN_SCHEMA, PingstoreError, file_sha256, load_json,
    payload_digest, run_root, validate_operational_run_directory, write_json_atomic,
)
from .layout import (
    display_manifest, export_directory, has_presentation_content,
    initialize_layout, presentation_directory,
)
from .native import execution_origin
from .registry import memberships

STAGES = ("compute", "analyse", "present")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class SourceRun:
    directory: Path
    record: dict
    manifest_sha256: str

    @property
    def export(self) -> Path:
        return export_directory(self.directory, self.record)

    @property
    def outputs(self) -> Path:
        """The complete export, including siblings of an explicit export_root."""
        return self.directory / "export"

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
            "run_json_sha256": self.manifest_sha256,
        }

    def check_unchanged(self) -> None:
        record = validate_operational_run_directory(self.directory)
        if (record["payload_digest"] != self.record["payload_digest"]
                or file_sha256(self.directory / "run.json") != self.manifest_sha256):
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
    source = SourceRun(directory, record, file_sha256(directory / "run.json"))
    if reference is not None and source.reference != reference:
        raise PingstoreError(f"upstream identity or checksum changed: {run_id}")
    return source


def reserve_stage(root: Path, experiment: str, stage: str,
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
        identity = f"{experiment}-r{number:03d}-{stage}-{origin}"
        directory = runs / f".{identity}.tmp"
        if (runs / identity).exists():
            continue
        try:
            directory.mkdir()
        except FileExistsError:
            continue
        initialize_layout(directory, experiment)
        write_json_atomic(directory / "provenance/reservation.json", {
            "schema": RUN_SCHEMA,
            "run_id": identity, "experiment": experiment, "stage": stage,
            "origin": origin, "reserved_at": utc_now(),
        })
        return identity


def stage_reservation(directory: Path) -> dict:
    """Read a v3 reservation without rewriting incomplete legacy executions."""
    path = directory / "provenance/reservation.json"
    if any(candidate.is_symlink() for candidate in (directory, path.parent, path)):
        raise PingstoreError("stage reservation must not use symlinks")
    if not path.is_file() and (directory / "export/provenance/reservation.json").exists():
        raise PingstoreError(
            "legacy v2 reservation is historical evidence; reserve a fresh v3 run"
        )
    reservation = load_json(path)
    if reservation.get("schema") != RUN_SCHEMA:
        raise PingstoreError("stage execution requires a v3 reservation")
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
    patch = git("diff", "--binary", "HEAD", "--", *paths)
    for name in git("ls-files", "--others", "--exclude-standard", "--", *paths).splitlines():
        patch += git("diff", "--no-index", "--binary", "--", "/dev/null", name,
                     allowed=(0, 1))
    record = {"git_commit": commit, "dirty": bool(git("status", "--porcelain")),
              "code_dirty": bool(patch), "patch": None}
    if patch:
        destination = directory / "provenance/source.patch"
        destination.write_text(patch)
        record["patch"] = {"path": "provenance/source.patch",
                           "sha256": file_sha256(destination)}
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
    def provenance(self) -> Path:
        return self.directory / "provenance"


@contextlib.contextmanager
def stage_run(repo: Path, experiment: str, stage: str, *,
              inputs: dict[str, SourceRun] | None = None, run_id: str | None = None,
              configuration: dict | None = None, export_root: str = "export",
              operation: str = "execute"):
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
    lock = directory / "provenance/writer.lock"
    descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as handle:
        handle.write(f"{os.getpid()}\n")
    command = [sys.executable, *sys.argv]
    record = {
        "schema": RUN_SCHEMA, "run_id": identity, "experiment": experiment,
        "collection": memberships(repo)[experiment], "stage": stage,
        "origin": reservation["origin"], "created_at": utc_now(),
        "export_root": export_root,
        "inputs": {name: source.reference for name, source in inputs.items()},
        "execution": {"operation": operation, "command": command,
                      "host": execution_origin(), "cwd": str(repo),
                      "started_at": utc_now(), "configuration": configuration},
        "provenance": _capture_code(repo, directory),
    }
    write_json_atomic(directory / "run.json", record)
    write_json_atomic(directory / "provenance/command.json", record["execution"])
    replay = list(command)
    if "--run-id" in replay:
        index = replay.index("--run-id")
        del replay[index:index + 2]
    (directory / "provenance/run.sh").write_text(
        "#!/bin/sh\n# Replay this stage with the same inputs and a fresh identity.\n"
        + "cd " + shlex.quote(str(repo)) + "\nexec " + shlex.join(replay) + "\n"
    )
    run = StageRun(directory, record)
    try:
        yield run
        for source in inputs.values():
            source.check_unchanged()
        record["execution"]["completed_at"] = utc_now()
        # Scientific stages have no preview sidecars. Presentation metadata
        # is only a projection; the complete provenance stays in run.json.
        if stage == "present" and has_presentation_content(run.export):
            display_manifest(directory, {
                "slug": experiment, "run_id": identity, "stage": stage,
                "run_at": record["created_at"], "host": record["origin"],
                "git_sha": record["provenance"]["git_commit"],
                "dirty": record["provenance"]["dirty"], "scale": configuration,
            }, identity)
        lock.unlink()
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
