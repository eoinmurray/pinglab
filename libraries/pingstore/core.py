"""A directory of immutable runs. Local operations never contact remote storage."""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import re
import secrets
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

EXPERIMENT_RE = re.compile(r"^exp[0-9]{3}$")
STAGE_ID_RE = re.compile(
    r"^(exp[0-9]{3})-r([0-9]{3,})-([a-z]+)(?:-phase-([a-z][a-z0-9-]*))?$"
)
RUN_ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,159}$")
PHASE_RE = re.compile(r"^[a-z][a-z0-9-]*$")
RUN_FIELDS = {
    "id",
    "experiment",
    "stage",
    "created_at",
    "description",
    "depends_on",
    "metadata",
}
_LOCKS = threading.local()


class PingstoreError(ValueError):
    """An invalid run or unsafe storage operation."""


def utc_now():
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def run_id(value):
    if (
        not isinstance(value, str)
        or not RUN_ID_RE.fullmatch(value)
        or value in {".", ".."}
    ):
        raise PingstoreError(f"invalid run ID: {value!r}")
    return value


def load_json(path):
    try:
        value = json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise PingstoreError(f"missing or invalid JSON: {path}") from error
    if not isinstance(value, dict):
        raise PingstoreError(f"expected a JSON object: {path}")
    return value


def write_json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    try:
        with temporary.open("x") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_run(value):
    if not isinstance(value, dict) or set(value) - RUN_FIELDS:
        raise PingstoreError(
            "run.json must contain only simple run metadata; migrate old records first"
        )
    identity = run_id(value.get("id"))
    for key in ("experiment", "created_at"):
        if not isinstance(value.get(key), str) or not value[key]:
            raise PingstoreError(f"run requires {key}")
    try:
        created = datetime.fromisoformat(value["created_at"].replace("Z", "+00:00"))
    except ValueError as error:
        raise PingstoreError("invalid created_at timestamp") from error
    if created.utcoffset() is None:
        raise PingstoreError("created_at requires a timezone")
    for key in ("stage", "description"):
        if key in value and not isinstance(value[key], str):
            raise PingstoreError(f"{key} must be a string")
    deps = value.get("depends_on")
    if not isinstance(deps, list):
        raise PingstoreError("depends_on must be a list of run IDs")
    for dependency in deps:
        run_id(dependency)
    if len(set(deps)) != len(deps) or identity in deps:
        raise PingstoreError("duplicate or self dependency")
    if not isinstance(value.get("metadata", {}), dict):
        raise PingstoreError("metadata must be an object")
    json.dumps(value, allow_nan=False)
    return value


def safe_file(directory, relative):
    directory, relative = Path(directory), Path(relative)
    if relative.is_absolute() or ".." in relative.parts or str(relative) == ".":
        raise PingstoreError("file path must stay inside the run")
    target = directory / relative
    for candidate in (directory, target, *target.parents):
        if candidate.is_symlink():
            raise PingstoreError(f"run files must not use symlinks: {candidate}")
    return target


def run_root(root, identity):
    return Path(root) / "runs" / run_id(identity)


def validate_run_directory(directory):
    directory = Path(directory)
    record = validate_run(load_json(safe_file(directory, "run.json")))
    if directory.name not in {record["id"], f".{record['id']}.tmp"}:
        raise PingstoreError("run directory and record identity differ")
    return record


validate_operational_run_directory = validate_run_directory


@contextlib.contextmanager
def operation_lock(root, *, exclusive=True):
    """Reentrant local writer lock; never used for ordinary reads or remote fencing."""
    root = Path(root).absolute()
    held = getattr(_LOCKS, "held", None)
    if held is None:
        held = _LOCKS.held = set()
    if root in held:
        yield
        return
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".write.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        held.add(root)
        try:
            yield
        finally:
            held.remove(root)
            fcntl.flock(handle, fcntl.LOCK_UN)


def inventory(directory):
    """File names and sizes only; never hash scientific payloads."""
    directory = Path(directory)
    files = {}
    for path in directory.rglob("*"):
        if path.is_symlink() or not (path.is_dir() or path.is_file()):
            raise PingstoreError(f"unsupported run entry: {path}")
        if path.is_file():
            files[path.relative_to(directory).as_posix()] = path.stat().st_size
    return files


@dataclass
class SourceRun:
    directory: Path
    record: dict
    _details: dict | None = field(default=None, repr=False)

    @property
    def run_id(self):
        return self.record["id"]

    @property
    def details(self):
        """Experiment-owned configuration, input roles and scientific provenance."""
        if self._details is None:
            path = safe_file(self.directory, "details.json")
            self._details = load_json(path) if path.exists() else {}
        return self._details

    @property
    def export(self):
        return self.directory / "export"

    @property
    def outputs(self):
        return self.export

    @property
    def presentation(self):
        return self.export

    @property
    def reference(self):
        return {"run_id": self.run_id}

    def file(self, *parts):
        from .artifacts import canonical_export_file

        return safe_file(
            self.export,
            canonical_export_file(self.export, *parts).relative_to(self.export),
        )

    def unit(self, *parts):
        from .artifacts import canonical_export_unit

        return safe_file(
            self.export,
            canonical_export_unit(self.export, *parts).relative_to(self.export),
        )

    def restore_bundle(self, prefix, destination):
        from .artifacts import restore_native

        mapping = self.details.get("scientific_artifacts", {}).get("native_paths")
        if mapping:
            return restore_native(self.export, mapping, prefix, destination)
        source = safe_file(self.export, prefix)
        inventory(source)
        shutil.copytree(source, destination)
        return Path(destination)

    def check_unchanged(self):
        """Check metadata only. Payloads are immutable by contract, not rehashed."""
        if validate_run_directory(self.directory) != self.record:
            raise PingstoreError(f"run metadata changed: {self.run_id}")


class Store:
    def __init__(self, root=Path(".pingstore")):
        self.root = Path(root).absolute()
        self.runs = self.root / "runs"

    def get(self, identity):
        directory = run_root(self.root, identity)
        return SourceRun(directory, validate_run_directory(directory))

    def list(self, *, experiment=None, stage=None):
        if not self.runs.exists():
            return []
        records = [
            self.get(path.name).record
            for path in sorted(self.runs.iterdir())
            if not path.name.startswith(".") and (path.is_dir() or path.is_symlink())
        ]
        return [
            r
            for r in records
            if (experiment is None or r["experiment"] == experiment)
            and (stage is None or r.get("stage") == stage)
        ]

    def latest(self, experiment, stage="present"):
        records = self.list(experiment=experiment, stage=stage)
        return (
            max(
                records,
                key=lambda r: (
                    datetime.fromisoformat(r["created_at"].replace("Z", "+00:00")),
                    r["id"],
                ),
            )
            if records
            else None
        )

    def dependants(self, identities):
        selected = set(identities)
        return {
            r["id"]: sorted(set(r["depends_on"]) & selected)
            for r in self.list()
            if r["id"] not in selected and selected.intersection(r["depends_on"])
        }

    def insert(self, source, *, record=None):
        source = Path(source).absolute()
        record = validate_run(record or load_json(source / "run.json"))
        identity = record["id"]
        with operation_lock(self.root):
            destination = run_root(self.root, identity)
            if destination.exists():
                raise FileExistsError(destination)
            if (
                source == destination
                or source.is_relative_to(destination)
                or destination.is_relative_to(source)
            ):
                raise PingstoreError("source must be outside the destination")
            inventory(source)
            temporary = self.runs / f".{identity}.tmp"
            self.runs.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, temporary)
            write_json_atomic(temporary / "run.json", record)
            return self._commit(temporary, record)

    def _commit(self, temporary, record):
        validate_run(record)
        destination = run_root(self.root, record["id"])
        if destination.exists():
            raise FileExistsError(destination)
        write_json_atomic(temporary / "run.json", record)
        files = inventory(temporary)
        write_json_atomic(
            self.root / "cache" / f"{record['id']}.json",
            {"bytes": sum(files.values()), "file_count": len(files)},
        )
        os.rename(temporary, destination)
        return self.get(record["id"])

    @contextlib.contextmanager
    def create(
        self,
        experiment,
        *,
        identity=None,
        stage=None,
        description="",
        depends_on=(),
        metadata=None,
    ):
        identity = identity or f"{experiment}-{secrets.token_hex(12)}"
        record = validate_run(
            dict(
                id=identity,
                experiment=experiment,
                created_at=utc_now(),
                description=description,
                depends_on=list(depends_on),
                metadata=metadata or {},
                **({"stage": stage} if stage is not None else {}),
            )
        )
        with operation_lock(self.root):
            destination = run_root(self.root, identity)
            if destination.exists():
                raise FileExistsError(destination)
            temporary = destination.with_name(f".{identity}.tmp")
            temporary.mkdir(parents=True)
        run = SourceRun(temporary, record)
        yield run
        with operation_lock(self.root):
            record["created_at"] = utc_now()
            self._commit(temporary, record)
            run.directory = destination

    def delete(self, identities, *, allow_dependants=False):
        identities = list(dict.fromkeys(run_id(i) for i in identities))
        with operation_lock(self.root):
            for identity in identities:
                self.get(identity)
            dependants = self.dependants(identities)
            if dependants and not allow_dependants:
                raise PingstoreError(
                    f"dependent runs: {dependants}; pass allow_dependants=True to proceed"
                )
            for identity in identities:
                shutil.rmtree(run_root(self.root, identity))
                (self.root / "cache" / f"{identity}.json").unlink(missing_ok=True)
        return identities

    def backup(self, destination, identities=None):
        from .remote import Remote

        return Remote(destination).backup(self, identities)

    def restore(self, destination, identities):
        from .remote import Remote

        return Remote(destination).restore(self, identities)


def source_run(
    root, identity, *, stage=None, experiment=None, reference=None, phase=None
):
    source = Store(root).get(identity)
    if stage is not None and source.record.get("stage") != stage:
        raise PingstoreError(f"{identity} is not a {stage} run")
    if experiment is not None and source.record["experiment"] != experiment:
        raise PingstoreError(f"{identity} does not belong to {experiment}")
    if reference is not None and reference != source.reference:
        raise PingstoreError(
            f"input identity differs: {identity}; use a plain run ID reference"
        )
    if phase is not None and source.record.get("metadata", {}).get("phase") != phase:
        raise PingstoreError(f"{identity} is not phase {phase}")
    return source


def execution_origin(host=None):
    return (
        "hpc"
        if os.environ.get("SLURM_JOB_ID")
        else "modal"
        if os.environ.get("MODAL_TASK_ID")
        else "local"
    )


def initialize_layout(directory, experiment=None):
    (Path(directory) / "export").mkdir(parents=True, exist_ok=True)
    (Path(directory) / "README.md").touch(exist_ok=True)


def reserve_stage(root, experiment, stage, *, origin=None, phase=None):
    if not EXPERIMENT_RE.fullmatch(experiment) or not PHASE_RE.fullmatch(stage):
        raise PingstoreError("invalid experiment or stage label")
    if phase is not None and not PHASE_RE.fullmatch(phase):
        raise PingstoreError("invalid phase label")
    # Numeric timestamp + random suffix preserves existing command-line ID formats.
    number = (
        datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
        + f"{secrets.randbelow(1_000_000):06d}"
    )
    identity = f"{experiment}-r{number}-{stage}" + (f"-phase-{phase}" if phase else "")
    with operation_lock(root):
        directory = run_root(root, identity).with_name(f".{identity}.tmp")
        directory.mkdir(parents=True)
        initialize_layout(directory)
        write_json_atomic(
            directory / ".reservation.json",
            dict(
                run_id=identity,
                experiment=experiment,
                stage=stage,
                origin=origin or execution_origin(),
                reserved_at=utc_now(),
                **({"phase": phase} if phase else {}),
            ),
        )
    return identity


def stage_reservation(directory):
    value = load_json(safe_file(directory, ".reservation.json"))
    run_id(value["run_id"])
    return value


def capture_code(repo, directory=None):
    def git(*args):
        return subprocess.check_output(
            ["git", *args], cwd=repo, text=True, timeout=30
        ).strip()

    return {
        "git_commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
        "code_dirty": bool(
            git(
                "status",
                "--porcelain",
                "--",
                "experiments",
                "libraries",
                "pyproject.toml",
                "uv.lock",
            )
        ),
    }


@dataclass
class StageRun:
    directory: Path
    details: dict

    @property
    def run_id(self):
        return self.details["run_id"]

    @property
    def export(self):
        return self.directory / "export"

    @property
    def scratch(self):
        path = self.directory / ".scratch"
        path.mkdir(parents=True, exist_ok=True)
        return path


def complete_stage(directory, details):
    """Complete an experiment-owned execution without interpreting scientific files."""
    directory = Path(directory)
    store = Store(directory.parent.parent)
    execution = (
        details.get("scientific_execution")
        or details.get("completed_hpc_import", {}).get("source_execution")
        or details["execution"]
    )
    metadata = {"origin": details.get("origin", "local")}
    if "phase" in details:
        metadata["phase"] = details["phase"]
    if execution.get("elapsed_seconds") is not None:
        metadata["duration_seconds"] = execution["elapsed_seconds"]
    elif execution.get("started_at") and execution.get("completed_at"):
        metadata["duration_seconds"] = (
            datetime.fromisoformat(execution["completed_at"].replace("Z", "+00:00"))
            - datetime.fromisoformat(execution["started_at"].replace("Z", "+00:00"))
        ).total_seconds()
    record = dict(
        id=details["run_id"],
        experiment=details["experiment"],
        stage=details["stage"],
        created_at=details["execution"]["completed_at"],
        description=details["execution"].get("operation", ""),
        depends_on=list(
            dict.fromkeys(pin["run_id"] for pin in details.get("inputs", {}).values())
        ),
        metadata=metadata,
    )
    write_json_atomic(directory / "details.json", details)
    scratch = directory / ".scratch"
    if scratch.exists():
        shutil.rmtree(scratch)
    (directory / ".reservation.json").unlink(missing_ok=True)
    with operation_lock(store.root):
        return store._commit(directory, record)


@contextlib.contextmanager
def stage_run(
    repo,
    experiment,
    stage,
    *,
    inputs=None,
    run_id=None,
    configuration=None,
    operation="execute",
    phase=None,
):
    store = Store(Path(repo) / ".pingstore")
    inputs = inputs or {}
    identity = run_id or reserve_stage(store.root, experiment, stage, phase=phase)
    directory = store.runs / f".{identity}.tmp"
    reservation = stage_reservation(directory)
    if any(
        reservation.get(k) != v
        for k, v in {
            "run_id": identity,
            "experiment": experiment,
            "stage": stage,
            "phase": phase,
        }.items()
    ):
        raise PingstoreError("reservation does not match execution")
    if (directory / "details.json").exists():
        raise PingstoreError(
            "interrupted execution; use a new run or explicit recovery"
        )
    descriptor = os.open(
        directory / ".writer.lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600
    )
    os.close(descriptor)
    started = time.perf_counter()
    details = dict(
        run_id=identity,
        experiment=experiment,
        stage=stage,
        origin=reservation["origin"],
        inputs={role: source.reference for role, source in inputs.items()},
        execution=dict(
            operation=operation, started_at=utc_now(), configuration=configuration or {}
        ),
        provenance=capture_code(repo),
    )
    if phase:
        details.update(scope="phase", phase=phase)
    write_json_atomic(directory / "details.json", details)
    run = StageRun(directory, details)
    yield run
    for source in inputs.values():
        source.check_unchanged()
    details["execution"].update(completed_at=utc_now())
    details["execution"].setdefault("elapsed_seconds", time.perf_counter() - started)
    (directory / ".writer.lock").unlink()
    completed = complete_stage(directory, details)
    run.directory = completed.directory
    print(identity)


def file_sha256(path):
    """Explicit scientific file check; ordinary storage operations never call this."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_runs(source):
    """Demolab's presentation-only metadata contract, without ancestry validation."""
    store = Store(Path(source).parent)
    return [
        dict(
            id=r["id"],
            experiment=r["experiment"],
            label=r["id"],
            created_at=r["created_at"],
            presentation=f"{r['id']}/export",
        )
        for r in store.list(stage="present")
        if (store.runs / r["id"] / "export").is_dir()
    ]
