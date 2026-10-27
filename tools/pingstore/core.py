"""Pingstore implementation: immutable scientific runs and storage operations.

Run validation, allocation, provenance, artifact normalization, discovery and
hash-bound maintenance live here. The experiment library and command interface
consume this module; core never imports either interface.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from http.client import HTTPException
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.request import Request, urlopen

import yaml

# Run records, validation and checksums

RUN_SCHEMA = "pingstore.run/v4"
EXPERIMENT_RE = re.compile(r"^exp[0-9]{3}$")
STAGE_ID_RE = re.compile(
    r"^(exp[0-9]{3})-r([0-9]{3,})-(compute|analyse|present)(?:(?<=compute)-phase-([a-z][a-z0-9]*(?:-[a-z0-9]+)*))?$"
)
PHASE_RE = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*")
VIEW_RE = re.compile(r"^[a-z0-9][a-z0-9./-]*$")


class PingstoreError(ValueError):
    """Raised when the flat store violates its filesystem contract."""


def validate_compute_scope(value: dict[str, Any]) -> None:
    """Validate explicit scopes without rewriting older v4 full-run evidence."""
    match = STAGE_ID_RE.fullmatch(str(value.get("run_id", "")))
    if match is None:
        raise PingstoreError("invalid scoped run identity")
    suffix = match.group(4)
    scope, phase = value.get("scope"), value.get("phase")
    if value.get("stage") != "compute":
        if suffix is not None or "scope" in value or "phase" in value:
            raise PingstoreError("only compute runs may declare scope or phase")
    elif scope == "phase":
        if (
            not isinstance(phase, str)
            or not PHASE_RE.fullmatch(phase)
            or suffix != phase
        ):
            raise PingstoreError("compute phase metadata and identity must agree")
    elif scope in (None, "full"):
        if (
            suffix is not None
            or "phase" in value
            or ("scope" in value and scope is None)
        ):
            raise PingstoreError("full compute runs must not declare a phase")
    else:
        raise PingstoreError("compute scope must be full or phase")


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise PingstoreError(f"missing Pingstore document: {path}") from exc
    except json.JSONDecodeError as exc:
        raise PingstoreError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PingstoreError(f"{path} must contain a JSON object")
    return value


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_input_references(*, value, run_id):
    """Validate explicit input identities and payload digests."""
    if not isinstance(value.get("inputs"), dict):
        raise PingstoreError("runs require explicit inputs (empty for new compute)")
    for role, reference in value["inputs"].items():
        if not isinstance(role, str) or not role or not isinstance(reference, dict):
            raise PingstoreError("invalid input role/reference")
        if set(reference) != {"run_id", "payload_digest"}:
            raise PingstoreError("input pins contain only run_id and payload_digest")
        if not isinstance(reference.get("run_id"), str) or not STAGE_ID_RE.fullmatch(
            reference["run_id"]
        ):
            raise PingstoreError("input must name a completed v4 run")
        if reference["run_id"] == run_id:
            raise PingstoreError("run cannot be its own input")
        if not re.fullmatch(
            r"sha256:[0-9a-f]{64}", str(reference.get("payload_digest", ""))
        ):
            raise PingstoreError("input requires a payload checksum")


def validate_run(value: dict[str, Any]) -> dict[str, Any]:
    if value.get("schema") != RUN_SCHEMA:
        raise PingstoreError(f"operational run schema must be {RUN_SCHEMA}")
    run_id = value.get("run_id")
    experiment = value.get("experiment")
    if not isinstance(run_id, str) or not STAGE_ID_RE.fullmatch(run_id):
        raise PingstoreError("run_id must encode experiment, counter and stage")
    if not isinstance(experiment, str) or not EXPERIMENT_RE.fullmatch(experiment):
        raise PingstoreError("experiment must be expNNN")
    if not run_id.startswith(experiment + "-"):
        raise PingstoreError("run_id must begin with experiment-")
    stage = value.get("stage")
    match = STAGE_ID_RE.fullmatch(run_id)
    if stage not in ("compute", "analyse", "present"):
        raise PingstoreError("stage must be compute, analyse or present")
    if match is None or match.group(1) != experiment or match.group(3) != stage:
        raise PingstoreError("run_id must encode experiment, counter and stage")
    validate_compute_scope(value)
    _validate_input_references(value=value, run_id=run_id)
    for key in ("collection", "origin", "created_at"):
        if not isinstance(value.get(key), str) or not value[key]:
            raise PingstoreError(f"{key} must be a non-empty string")
    if not isinstance(value.get("execution"), dict):
        raise PingstoreError("execution must be an object")
    if not isinstance(value.get("provenance"), dict):
        raise PingstoreError("provenance must be an object")
    digest = value.get("payload_digest")
    if not isinstance(digest, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
        raise PingstoreError("payload_digest must be a prefixed SHA-256")
    return value


def validate_collections(value: dict[str, Any]) -> dict[str, Any]:
    for name, run_ids in value.items():
        if not isinstance(name, str) or not VIEW_RE.fullmatch(name):
            raise PingstoreError(f"invalid collection view name: {name!r}")
        if not isinstance(run_ids, list) or not all(
            isinstance(run_id, str) and STAGE_ID_RE.fullmatch(run_id)
            for run_id in run_ids
        ):
            raise PingstoreError(f"collection view {name!r} must be a run-ID array")
        if len(run_ids) != len(set(run_ids)):
            raise PingstoreError(f"collection view {name!r} contains duplicate runs")
    return value


def run_root(root: Path, run_id: str) -> Path:
    if not STAGE_ID_RE.fullmatch(run_id):
        raise PingstoreError(f"invalid run ID: {run_id}")
    return root / "runs" / run_id


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def payload_inventory(directory: Path) -> list[dict[str, Any]]:
    """Inventory immutable scientific bytes beneath export/."""
    root = directory / "export"
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise PingstoreError(f"unsupported payload entry: {path}")
        relative = path.relative_to(directory).as_posix()
        if path.is_file():
            rows.append(
                {
                    "path": relative,
                    "size_bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
            )
    return rows


def payload_digest(directory: Path) -> str:
    encoded = json.dumps(
        payload_inventory(directory), sort_keys=True, separators=(",", ":")
    ).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _validate_flat_exports(*, flat, directory):
    """Reject ambiguous or unsafe paths in a presentation export."""
    if flat is not None:
        for path in flat.iterdir():
            if path.is_symlink() or not path.is_file():
                raise PingstoreError(f"presentation must be flat regular files: {path}")
    else:
        export = directory / "export"

        for path in export.rglob("*"):
            relative = path.relative_to(export)
            if len(relative.parts) > 2:
                raise PingstoreError(
                    f"scientific export exceeds one unit-directory level: {path}"
                )
            if path.is_dir() and len(relative.parts) != 1:
                raise PingstoreError(f"invalid scientific unit directory: {path}")
            if path.is_file() and canonical_role_name(path.name) != path.name:
                raise PingstoreError(f"noncanonical scientific role filename: {path}")
        for unit in (path for path in export.iterdir() if path.is_dir()):
            files = [path for path in unit.iterdir() if path.is_file()]
            if len(files) < 2:
                raise PingstoreError(
                    f"scientific unit directories require at least two files: {unit}"
                )


def validate_layout(directory: Path) -> None:
    if directory.is_symlink() or not directory.is_dir():
        raise PingstoreError(f"run must be a real directory: {directory}")
    manifest = directory / "run.json"
    if manifest.is_symlink() or not manifest.is_file():
        raise PingstoreError(f"run.json must be a regular file: {manifest}")
    run = validate_run(load_json(manifest))
    names = {p.name for p in directory.iterdir()}
    if names - {".protected"} != {"run.json", "README.md", "export"}:
        raise PingstoreError(
            "v4 run must contain exactly run.json, README.md and export/ plus optional .protected"
        )
    directories = {"export"}
    flat = directory / "export" if run["stage"] == "present" else None
    for name in names:
        path = directory / name
        if path.is_symlink() or (path.is_dir() != (name in directories)):
            raise PingstoreError(f"invalid run entry: {path}")
        if name in {"run.json", "README.md", ".protected"} and not path.is_file():
            raise PingstoreError(f"run entry must be a regular file: {path}")
    _validate_flat_exports(flat=flat, directory=directory)
    for name in directories & names:
        for path in (directory / name).rglob("*"):
            if path.is_symlink() or not (path.is_file() or path.is_dir()):
                raise PingstoreError(f"unsupported payload entry: {path}")


def validate_run_directory(directory: Path) -> dict[str, Any]:
    """Validate structure, identity and checksums before reading or publishing."""
    validate_layout(directory)
    run = validate_run(load_json(directory / "run.json"))
    if directory.name not in {run["run_id"], f".{run['run_id']}.tmp"}:
        raise PingstoreError("run directory and run.json identity differ")
    if "data_root" in run:
        raise PingstoreError("data_root is obsolete; use export_root beneath export/")
    if "export_root" in run:
        raise PingstoreError(
            "export_root is obsolete; unit directories live directly under export/"
        )
    if payload_digest(directory) != run["payload_digest"]:
        raise PingstoreError(f"payload checksum mismatch: {directory}")
    return run


def validate_operational_run_directory(directory: Path) -> dict[str, Any]:
    """Validate one operational v4 run before consuming evidence."""
    if any(path.is_symlink() for path in (directory, *directory.parents)):
        raise PingstoreError("operational input paths must not use symlinks")
    return validate_run_directory(directory)


# Canonical scientific export layout

RECORD_NAMES = {
    "_manifest.json",
    "_run.txt",
    "_provenance.json",
    "run.sh",
    "_dirty.patch",
    "reproducer.json",
}
ROLE_ALIASES = {
    "snapshot.npz": "recording.npz",
    "recordings.npz": "recording.npz",
}


def canonical_role_name(name: str) -> str:
    """Return the standard scientific role filename, preserving extensions."""
    if name in ROLE_ALIASES:
        return ROLE_ALIASES[name]
    if name.endswith("--snapshot.npz"):
        return name.removesuffix("--snapshot.npz") + "--recording.npz"
    if name.endswith("-recordings.npz"):
        return name.removesuffix("-recordings.npz") + "-recording.npz"
    return name


def initialize_layout(root: Path, experiment: str) -> None:
    (root / "export").mkdir(parents=True, exist_ok=True)
    readme = root / "README.md"
    if not readme.exists():
        readme.write_text(f"# {experiment} run\n\n## History\n\n")


def export_directory(root: Path, run: dict) -> Path:
    """Resolve scientific output from an already validated v4 record."""
    return root / "export"


def canonical_export_relative(relative: Path, *, export_root: str = "export") -> Path:
    """Map a scientific file to a root file or candidate unit/role path."""
    parts = relative.parts
    prefix = Path(export_root).parts
    if (
        prefix
        and prefix[0] == "export"
        and tuple(parts[: len(prefix) - 1]) == prefix[1:]
    ):
        parts = parts[len(prefix) - 1 :]
    if len(parts) <= 1:
        return Path(canonical_role_name(parts[0]))
    if len(parts) == 2:
        return Path(parts[0]) / canonical_role_name(parts[1])
    directories, filename = list(parts[:-1]), parts[-1]
    bundle = next(
        (index for index, name in enumerate(directories) if name.endswith(".bundle")),
        None,
    )
    if bundle is None:
        return Path("--".join(directories)) / canonical_role_name(filename)
    unit = "--".join(directories[: bundle + 1])
    remainder = directories[bundle + 1 :]
    role = "--".join([*remainder, filename]) if remainder else filename
    role = canonical_role_name(role)
    return Path(unit) / role


def canonical_export_mapping(
    relatives: list[Path], *, export_root: str = "export"
) -> dict[str, str]:
    """Map files, flattening units that contain only one scientific role."""
    candidates = {
        relative.as_posix(): canonical_export_relative(
            relative, export_root=export_root
        )
        for relative in relatives
    }
    counts: dict[Path, int] = {}
    for target in candidates.values():
        if target.parent != Path("."):
            counts[target.parent] = counts.get(target.parent, 0) + 1
    mapping = {}
    for source, target in candidates.items():
        if target.parent != Path(".") and counts[target.parent] == 1:
            target = Path(f"{target.parent.name}--{target.name}")
        mapping[source] = target.as_posix()
    return mapping


def canonical_export_unit(root: Path, *parts: str | Path) -> Path:
    values = []
    for part in parts:
        values.extend(Path(part).parts)
    direct = root.joinpath(*values)
    return direct if direct.exists() else root / "--".join(values)


def canonical_export_file(root: Path, *parts: str | Path) -> Path:
    relative = Path()
    for part in parts:
        relative /= Path(part)
    direct = root / relative
    if direct.exists():
        return direct
    bundled = canonical_export_relative(relative)
    candidate = root / bundled
    if candidate.exists() or bundled.parent == Path("."):
        return candidate
    return root / f"{bundled.parent.name}--{bundled.name}"


def _rewrite_paths(value, mapping: dict[str, str]):
    if isinstance(value, dict):
        return {key: _rewrite_paths(item, mapping) for key, item in value.items()}
    if isinstance(value, list):
        return [_rewrite_paths(item, mapping) for item in value]
    if isinstance(value, str):
        if value in mapping:
            return mapping[value]
        if value.startswith("export/") and value[7:] in mapping:
            return "export/" + mapping[value[7:]]
    return value


def normalize_export_layout(directory: Path, record: dict) -> dict[str, str]:
    """Normalize a hidden or staged run without mutating a visible source run."""
    export = directory / "export"
    export_root = record.get("export_root", "export")
    files = [path for path in sorted(export.rglob("*")) if path.is_file()]

    artifacts = record.get("scientific_artifacts", {})
    retained_native = artifacts.get("native_paths", {})
    snapshot_stages = artifacts.get("snapshot_stages")
    native = native_artifact_files(export) | set(retained_native.values())
    mapping = canonical_export_mapping(
        [path.relative_to(export) for path in files], export_root=export_root
    )
    if len(set(mapping.values())) != len(mapping):
        raise PingstoreError(f"{directory.name}: canonical export paths collide")
    protected = {mapping[source] for source in native}
    native_mapping = {
        source: mapping[stored] for source, stored in retained_native.items()
    }
    native_mapping.update(
        {
            source: mapping[source]
            for source in sorted(native)
            if source not in retained_native.values()
        }
    )
    temporary = directory / ".normalized-export.tmp"
    if temporary.exists():
        raise PingstoreError(f"{directory.name}: stale export normalization directory")
    temporary.mkdir()
    try:
        for source in files:
            target = temporary / mapping[source.relative_to(export).as_posix()]
            target.parent.mkdir(parents=True, exist_ok=True)
            os.link(source, target)
        for target in temporary.rglob("*.json"):
            if target.relative_to(temporary).as_posix() in protected:
                continue
            try:
                value = json.loads(target.read_text())
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            revised = _rewrite_paths(value, mapping)
            if revised != value:
                write_json_atomic(target, revised)
        if native or snapshot_stages:
            validate_native_artifacts(
                temporary,
                native_mapping,
                snapshot_stages=snapshot_stages,
            )
        shutil.rmtree(export)
        os.replace(temporary, export)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    record.pop("export_root", None)
    revised = _rewrite_paths(record, mapping)
    record.clear()
    record.update(revised)
    if native:
        record.setdefault("scientific_artifacts", {})["native_paths"] = native_mapping
    return mapping


def presentation_directory(root: Path, run: dict) -> Path | None:
    """Resolve publishable output; stage is authoritative, not the folder name."""
    return root / "export" if run["stage"] == "present" else None


def has_presentation_content(directory: Path) -> bool:
    return any(
        path.name not in RECORD_NAMES and path.stat().st_size > 0
        for path in directory.iterdir()
    )


# Native library artifacts and retention


def native_artifact_files(export):
    """Find complete bundles, training checkpoints and simulation snapshots."""
    files = set()
    for path in export.rglob("*.json"):
        if path.name not in {"manifest.json", "snapshot.json"}:
            continue
        metadata = load_json(path)
        if metadata.get("schema") in {
            "snnlang.bundle/v1",
            "tools/snnsim.training-checkpoint/v1",
        }:
            files.update(p for p in path.parent.rglob("*") if p.is_file())
        elif metadata.get("schema") in {
            "snnlab.run-snapshot/v1",
            "pinglab.external-run-snapshot/v1",
        }:
            files.add(path)
            parameters = path.with_name("parameters.npz")
            if parameters.exists():
                files.add(parameters)
    return {path.relative_to(export).as_posix() for path in files}


def restore_native(export, mapping, prefix, destination):
    """Reconstruct original names without modifying retained scientific bytes."""
    prefix = Path(prefix)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    selected = {
        source: target
        for source, target in mapping.items()
        if Path(source).is_relative_to(prefix)
    }
    if not selected:
        raise PingstoreError(f"no retained native artifact: {prefix}")
    import shutil

    for source, target in selected.items():
        relative = Path(source).relative_to(prefix)
        if (
            ".." in relative.parts
            or Path(target).is_absolute()
            or ".." in Path(target).parts
        ):
            raise PingstoreError("invalid native artifact path")
        output = destination / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(export / target, output)
    return destination


def _snapshot_retention_policy(*, snapshot_stages):
    """Validate the declared retention stages for every snapshot root."""
    expected = {}
    for root, stages in (snapshot_stages or {}).items():
        path = Path(root)
        if (
            path.is_absolute()
            or ".." in path.parts
            or not isinstance(stages, list)
            or not stages
            or any(stage not in {"start", "end"} for stage in stages)
            or len(set(stages)) != len(stages)
        ):
            raise PingstoreError(f"invalid snapshot retention policy: {root}")
        expected[path] = set(stages)
    return expected


def _native_artifact_roles(*, export, mapping):
    """Authenticate native snapshot files and collect bundle and checkpoint roots."""
    bundles, checkpoints, snapshots = [], [], {}
    for source, target in mapping.items():
        if Path(source).name not in {"manifest.json", "snapshot.json"}:
            continue
        metadata = load_json(export / target)
        if metadata.get("schema") == "snnlang.bundle/v1":
            bundles.append(Path(source).parent)
        elif metadata.get("schema") == "tools/snnsim.training-checkpoint/v1":
            checkpoints.append(Path(source).parent)
        elif metadata.get("schema") in {
            "snnlab.run-snapshot/v1",
            "pinglab.external-run-snapshot/v1",
        }:
            root = Path(source).parent
            stage = root.name
            if stage not in {"start", "end"} or metadata.get("stage") != stage:
                raise PingstoreError(f"invalid native snapshot stage: {source}")
            snapshots.setdefault(root.parent, set()).add(stage)
            inventory = metadata.get("parameters", {})
            if not metadata.get("parameters_included"):
                raise PingstoreError(f"parameter snapshot required: {source}")
            if inventory:
                parameters = (root / "parameters.npz").as_posix()
                if parameters not in mapping or file_sha256(
                    export / mapping[parameters]
                ) != metadata.get("parameter_file_sha256"):
                    raise PingstoreError(
                        f"snapshot parameter checksum mismatch: {source}"
                    )
    return bundles, checkpoints, snapshots


def validate_native_artifacts(export, mapping, *, snapshot_stages=None):
    """Authenticate native evidence, requiring pairs unless retention is explicit.

    snapshot_stages maps original execution directories to retained native stages.
    Store this policy in run.json so an intentionally single-stage capture remains
    distinguishable from a broken start/end pair.
    """
    expected = _snapshot_retention_policy(snapshot_stages=snapshot_stages)
    bundles, checkpoints, snapshots = _native_artifact_roles(
        export=export, mapping=mapping
    )
    for root in snapshots.keys() | expected.keys():
        stages = snapshots.get(root, set())
        if root in expected:
            if stages != expected[root]:
                raise PingstoreError(
                    f"snapshot stages differ from retention policy: {root}"
                )
        elif stages != {"start", "end"}:
            raise PingstoreError(f"missing start/end snapshot pair: {root}")
    if bundles:
        from snnlab.lang import load_bundle

        with tempfile.TemporaryDirectory(prefix="pinglab-native-") as temporary:
            for index, prefix in enumerate(bundles):
                load_bundle(
                    restore_native(
                        export, mapping, prefix, Path(temporary) / str(index)
                    )
                )
    if checkpoints:
        from snnlab.sim import load_training_checkpoint

        with tempfile.TemporaryDirectory(prefix="pinglab-checkpoint-") as temporary:
            for index, prefix in enumerate(checkpoints):
                load_training_checkpoint(
                    restore_native(
                        export, mapping, prefix, Path(temporary) / str(index)
                    )
                )


# Authored report metadata


class _ReportLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        self.flatten_mapping(node)
        keys = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str) or key in keys:
                raise yaml.constructor.ConstructorError(
                    None,
                    None,
                    "report metadata keys must be unique strings",
                    key_node.start_mark,
                )
            keys.add(key)
        return super().construct_mapping(node, deep=deep)


def articles(repo: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for path in sorted(
        (repo / "experiments").glob("exp[0-9][0-9][0-9]/report/report.mdx")
    ):
        match = re.match(r"\A---\r?\n(.*?)\r?\n---(?:\r?\n|\Z)", path.read_text(), re.S)
        if match is None:
            raise PingstoreError(f"report requires YAML frontmatter: {path}")
        try:
            value = yaml.load(match[1], Loader=_ReportLoader)
        except yaml.YAMLError as exc:
            raise PingstoreError(f"invalid report frontmatter: {path}") from exc
        if not isinstance(value, dict) or value.get("id") != path.parent.parent.name:
            raise PingstoreError(f"report ID must match its experiment: {path}")
        inputs = value.get("inputs", [])
        if (
            not isinstance(inputs, list)
            or any(
                not isinstance(key, str) or not re.fullmatch(r"exp[0-9]{3}", key)
                for key in inputs
            )
            or len(set(inputs)) != len(inputs)
        ):
            raise PingstoreError(f"invalid report inputs: {path}")
        pins = value.get("runs", {})
        if not isinstance(pins, dict) or any(
            key not in inputs
            or not isinstance(run_id, str)
            or not re.fullmatch(
                r"exp[0-9]{3}-r[0-9]+-(compute|analyse|present)", run_id
            )
            for key, run_id in pins.items()
        ):
            raise PingstoreError(f"invalid report presentation pins: {path}")
        value["inputs"] = inputs
        value["runs"] = pins
        result[path.parent.parent.name] = value
    return result


# Experiment collection membership

COLLECTION_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")


def memberships(repo: Path) -> dict[str, str]:
    """Return each report's primary collection for new run provenance."""
    result: dict[str, str] = {}
    for experiment, metadata in articles(repo).items():
        collection = metadata.get("collection")
        collections = [collection] if isinstance(collection, str) else collection
        if (
            not isinstance(collections, list)
            or not collections
            or any(
                not isinstance(item, str) or not COLLECTION_RE.fullmatch(item)
                for item in collections
            )
            or len(set(collections)) != len(collections)
        ):
            raise PingstoreError(f"invalid collection for {experiment}: {collection}")
        result[experiment] = collections[0]
    return result


def membership(repo: Path, experiment: str) -> str:
    """Return one active membership or fail with a useful contract error."""
    try:
        return memberships(repo)[experiment]
    except KeyError as exc:
        raise PingstoreError(
            f"active report for {experiment} must declare collection metadata"
        ) from exc


def _runnable_experiments(repo: Path) -> set[str]:
    experiments = repo / "experiments"
    legacy = {path.stem for path in experiments.glob("exp[0-9][0-9][0-9].py")}
    staged = {
        path.name
        for path in experiments.glob("exp[0-9][0-9][0-9]")
        if path.is_dir()
        and any(
            (path / f"{stage}.py").is_file()
            for stage in ("compute", "analyse", "present")
        )
    }
    return legacy | staged


def coverage(repo: Path) -> dict[str, Any]:
    declared = memberships(repo)
    registered = set(declared)
    runnable = _runnable_experiments(repo)
    capture_routes: dict[str, str] = {}
    experiments = repo / "experiments"
    for experiment in sorted(runnable):
        stage_dir = experiments / experiment
        if stage_dir.is_dir() and any(
            (stage_dir / f"{stage}.py").is_file()
            for stage in ("compute", "analyse", "present")
        ):
            capture_routes[experiment] = "independent-stages"
            continue
        text = (experiments / f"{experiment}.py").read_text()
        if "published_run(" in text:
            capture_routes[experiment] = "atomic-published-run"
        elif "finalize_prepared_run(" in text:
            capture_routes[experiment] = "legacy-success-finalizer"
    return {
        "runnable": sorted(runnable),
        "registered": sorted(registered),
        "missing_membership": sorted(runnable - registered),
        "article_only": sorted(registered - runnable),
        "capture_routes": capture_routes,
        "missing_capture": sorted(runnable - set(capture_routes)),
        "passed": runnable <= registered and runnable == set(capture_routes),
    }


# Durable run allocation counters

COUNTER_RE = re.compile(
    r"^\.?(exp[0-9]{3})-(?:(?:compute|analyse|present)-)?r([0-9]+)-"
)


def allocation_watermarks(store: Path) -> dict[str, int]:
    directory = store / ".allocation"
    counters = {}
    if directory.is_symlink() or (directory.exists() and not directory.is_dir()):
        raise PingstoreError("allocation state must be a real directory")
    if directory.exists():
        for path in sorted(directory.iterdir()):
            if (
                path.is_symlink()
                or not path.is_file()
                or path.suffix != ".json"
                or not EXPERIMENT_RE.fullmatch(path.stem)
            ):
                raise PingstoreError(f"invalid allocation state: {path}")
            value = load_json(path).get("high_watermark")
            if type(value) is not int or value < 0:
                raise PingstoreError(f"invalid allocation counter: {path}")
            counters[path.stem] = value
    runs = store / "runs"
    if runs.exists():
        for path in runs.iterdir():
            if match := COUNTER_RE.match(path.name):
                experiment, number = match.groups()
                counters[experiment] = max(counters.get(experiment, 0), int(number))
    return dict(sorted(counters.items()))


def persist_watermarks(store: Path, counters: dict[str, int]) -> None:
    """Caller holds the allocation lock or exclusive operation lock."""
    for experiment, number in counters.items():
        write_json_atomic(
            store / ".allocation" / f"{experiment}.json", {"high_watermark": number}
        )
    if counters:
        descriptor = os.open(store / ".allocation", os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


# Atomic run creation and explicit source access

STAGES = ("compute", "analyse", "present")
LOCK_NAME = ".operation.lock"


def _safe_label(value: object, fallback: str) -> str:
    text = re.sub(r"[^a-z0-9.-]+", "-", str(value).lower()).strip("-.")
    return text or fallback


def execution_origin(host: str | None = None) -> str:
    """Describe the local process or active Slurm allocation."""
    slurm_job = os.environ.get("SLURM_JOB_ID")
    if slurm_job:
        cluster = os.environ.get("SLURM_CLUSTER_NAME") or host or platform.node()
        return (
            f"slurm-{_safe_label(cluster, 'cluster')}-{_safe_label(slurm_job, 'job')}"
        )
    raw = host or "local"
    return "local" if raw == "local" else _safe_label(raw, "local")


@contextlib.contextmanager
def operation_lock(root: Path, *, exclusive: bool):
    """Coordinate v4 writers with exclusive pruning."""
    root.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(root / LOCK_NAME, os.O_RDWR | os.O_CREAT, 0o600)
    mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
    try:
        try:
            fcntl.flock(descriptor, mode | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            action = "prune" if exclusive else "reserve or execute a run"
            raise PingstoreError(
                f"cannot {action} while another Pingstore operation is active"
            ) from exc
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


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

    def restore_bundle(self, prefix: str | Path, destination: Path) -> Path:
        """Restore a native bundle into caller-owned scratch for authenticated loading."""

        return restore_native(
            self.outputs,
            self.record["scientific_artifacts"]["native_paths"],
            prefix,
            destination,
        )

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
            raise PingstoreError(
                f"source changed during execution: {self.directory.name}"
            )


def source_run(
    root: Path,
    run_id: str,
    *,
    stage: str | None = None,
    experiment: str | None = None,
    reference: dict | None = None,
    phase: str | None = None,
) -> SourceRun:
    directory = run_root(root.absolute(), run_id)
    if any(path.is_symlink() for path in (directory, *directory.parents)):
        raise PingstoreError("source paths must not use symlinks")
    record = validate_operational_run_directory(directory)
    if stage is not None and record.get("stage") != stage:
        raise PingstoreError(f"{run_id} is not a {stage} run")
    if experiment is not None and record["experiment"] != experiment:
        raise PingstoreError(f"{run_id} does not belong to {experiment}")
    if phase is not None:
        if record.get("scope") != "phase" or record.get("phase") != phase:
            raise PingstoreError(f"{run_id} is not the required compute phase {phase}")
    elif record.get("scope") == "phase":
        raise PingstoreError("phase input requires an explicitly phase-aware consumer")
    source = SourceRun(directory, record)
    if reference is not None and source.reference != reference:
        raise PingstoreError(f"upstream identity or checksum changed: {run_id}")
    return source


def _allocate_stage(
    root: Path,
    experiment: str,
    stage: str,
    *,
    origin: str | None = None,
    phase: str | None = None,
) -> str:
    """Atomically reserve an identity, including before scheduler submission."""
    if stage not in STAGES or not EXPERIMENT_RE.fullmatch(experiment):
        raise PingstoreError("invalid experiment or stage")
    if phase is not None and (
        stage != "compute"
        or not isinstance(phase, str)
        or not PHASE_RE.fullmatch(phase)
    ):
        raise PingstoreError("phase must be a valid compute phase name")
    origin = origin or execution_origin()
    if not re.fullmatch(r"[a-z0-9][a-z0-9.-]*", origin):
        raise PingstoreError("invalid execution origin")
    runs = root / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    number = allocation_watermarks(root).get(experiment, 0)
    while True:
        number += 1
        persist_watermarks(root, {experiment: number})
        identity = f"{experiment}-r{number:03d}-{stage}"
        if phase is not None:
            identity += f"-phase-{phase}"
        directory = runs / f".{identity}.tmp"
        if (runs / identity).exists():
            continue
        try:
            directory.mkdir()
        except FileExistsError:
            continue
        initialize_layout(directory, experiment)
        reservation = {
            "schema": RUN_SCHEMA,
            "run_id": identity,
            "experiment": experiment,
            "stage": stage,
            "origin": origin,
            "reserved_at": utc_now(),
        }
        if stage == "compute":
            reservation["scope"] = "phase" if phase is not None else "full"
            if phase is not None:
                reservation["phase"] = phase
        write_json_atomic(directory / ".reservation.json", reservation)
        return identity


def _reserve_stage(
    root: Path,
    experiment: str,
    stage: str,
    *,
    origin: str | None = None,
    phase: str | None = None,
) -> str:
    # Different phase suffixes must never claim the same experiment-wide counter.
    root.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(root / ".allocation.lock", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        return _allocate_stage(root, experiment, stage, origin=origin, phase=phase)
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def reserve_stage(
    root: Path,
    experiment: str,
    stage: str,
    *,
    origin: str | None = None,
    phase: str | None = None,
) -> str:
    with operation_lock(root, exclusive=False):
        return _reserve_stage(root, experiment, stage, origin=origin, phase=phase)


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
    if (
        match is None
        or match.group(1) != reservation.get("experiment")
        or match.group(3) != reservation.get("stage")
    ):
        raise PingstoreError(
            "stage execution requires a source-neutral reservation; reserve a fresh identity"
        )
    if not re.fullmatch(r"[a-z0-9][a-z0-9.-]*", str(reservation.get("origin", ""))):
        raise PingstoreError("invalid reservation execution origin")
    validate_compute_scope(reservation)
    return reservation


def capture_code(repo: Path, directory: Path) -> dict:
    def git(*args: str, allowed: tuple[int, ...] = (0,)) -> str:
        result = subprocess.run(
            ["git", *args], cwd=repo, capture_output=True, text=True, timeout=30
        )
        if result.returncode not in allowed:
            raise PingstoreError(
                f"cannot capture source provenance: {result.stderr.strip()}"
            )
        return result.stdout

    paths = ("experiments", "tools", "pyproject.toml", "uv.lock")
    commit = git("rev-parse", "HEAD").strip()
    dirty_paths = git("status", "--porcelain", "--", *paths)
    record = {
        "git_commit": commit,
        "dirty": bool(git("status", "--porcelain")),
        "code_dirty": bool(dirty_paths),
    }
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
def _stage_run(
    repo: Path,
    experiment: str,
    stage: str,
    *,
    inputs: dict[str, SourceRun] | None = None,
    run_id: str | None = None,
    configuration: dict | None = None,
    operation: str = "execute",
    phase: str | None = None,
):
    """Complete one execution atomically; never materialize or dispatch a stage."""
    root = repo / ".pingstore"
    inputs = inputs or {}
    for source in inputs.values():
        source.check_unchanged()
    identity = run_id or reserve_stage(root, experiment, stage, phase=phase)
    destination = run_root(root, identity)
    directory = destination.with_name(f".{identity}.tmp")
    if destination.exists() or not directory.is_dir():
        raise PingstoreError(f"run must be an unused reserved identity: {identity}")
    reservation = stage_reservation(directory)
    if (
        reservation["run_id"] != identity
        or reservation["stage"] != stage
        or reservation["experiment"] != experiment
        or reservation.get("phase") != phase
    ):
        raise PingstoreError("reservation does not match execution")
    if (directory / "run.json").exists():
        raise PingstoreError(
            "an interrupted execution needs explicit recovery or a new run"
        )
    lock = directory / ".writer.lock"
    descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as handle:
        handle.write(f"{os.getpid()}\n")
    command = [sys.executable, *sys.argv]
    record = {
        "schema": RUN_SCHEMA,
        "run_id": identity,
        "experiment": experiment,
        "collection": memberships(repo)[experiment],
        "stage": stage,
        "origin": reservation["origin"],
        "created_at": utc_now(),
        "inputs": {name: source.reference for name, source in inputs.items()},
        "execution": {
            "operation": operation,
            "command": command,
            "host": execution_origin(),
            "cwd": str(repo),
            "started_at": utc_now(),
            "configuration": configuration,
        },
        "provenance": capture_code(repo, directory),
    }
    if stage == "compute":
        record["scope"] = "phase" if phase is not None else "full"
        if phase is not None:
            record["phase"] = phase
    write_json_atomic(directory / "run.json", record)
    inputs_text = (
        "\n".join(
            f"- `{name}`: `{source.record['run_id']}` (`{source.record['payload_digest']}`)"
            for name, source in inputs.items()
        )
        or "- None"
    )
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
            handle.write(
                f"- {record['execution']['completed_at']}: run completed successfully.\n"
            )
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
def stage_run(
    repo: Path,
    experiment: str,
    stage: str,
    *,
    inputs: dict[str, SourceRun] | None = None,
    run_id: str | None = None,
    configuration: dict | None = None,
    operation: str = "execute",
    phase: str | None = None,
):
    with (
        operation_lock(repo / ".pingstore", exclusive=False),
        _stage_run(
            repo,
            experiment,
            stage,
            inputs=inputs,
            run_id=run_id,
            configuration=configuration,
            operation=operation,
            phase=phase,
        ) as run,
    ):
        yield run


# Validated run discovery


def _validate_source(source: Path) -> Path:
    source = source.expanduser().absolute()
    if any(path.is_symlink() for path in (source, *source.parents)):
        raise PingstoreError(f"discovery source must not use symlinks: {source}")
    if not source.is_dir():
        raise PingstoreError(
            f"discovery source must be an existing runs directory: {source}"
        )
    return source


def validated_run_graph(
    source: Path, selected_ids: set[str] | None = None
) -> dict[str, dict]:
    """Validate all visible runs, or selected runs and their complete ancestry."""
    source = source.expanduser().absolute()
    if selected_ids is None:
        pending = [
            path.name
            for path in sorted(source.iterdir())
            if not path.name.startswith(".") and path.is_dir() and not path.is_symlink()
        ]
    else:
        pending = list(selected_ids)
    records = {}
    while pending:
        identity = pending.pop()
        if identity in records:
            continue
        if Path(identity).name != identity or identity.startswith("."):
            raise PingstoreError(f"unsafe run identity: {identity}")
        try:
            record = validate_operational_run_directory(source / identity)
        except PingstoreError as exc:
            raise PingstoreError(f"{identity}: {exc}") from exc
        records[identity] = record
        if selected_ids is not None:
            pending.extend(ref["run_id"] for ref in record["inputs"].values())
    for child, record in records.items():
        for reference in record["inputs"].values():
            parent = records.get(reference["run_id"])
            if (
                parent is None
                or parent["payload_digest"] != reference["payload_digest"]
            ):
                raise PingstoreError(
                    f"{child}: missing or changed input {reference['run_id']}"
                )
    return records


def discover_store(source: Path) -> tuple[dict[str, dict], list[dict[str, str]]]:
    """Validate the complete store and project its populated present runs.

    Validate every candidate before returning anything. Demolab currently has no
    separate validation callback, so metadata-only discovery would let it consume
    unverified payloads. Hidden entries and symlink candidates are never followed.
    """
    source = _validate_source(source)
    graph = validated_run_graph(source)
    records = []
    for identity, run in sorted(graph.items()):
        directory = source / identity
        try:
            # The run contract requires a string; Demolab additionally requires
            # a parseable, timezone-aware timestamp. Never substitute file times.
            created_at = datetime.fromisoformat(
                run["created_at"].replace("Z", "+00:00")
            )
            if created_at.utcoffset() is None:
                raise PingstoreError("created_at must include a timezone")
            timestamp = created_at.astimezone(timezone.utc).isoformat()
        except (OSError, ValueError, OverflowError) as exc:
            raise PingstoreError(f"cannot discover {directory.name}: {exc}") from exc
        files = presentation_directory(directory, run)
        if files is None or not has_presentation_content(files):
            continue
        records.append(
            {
                "id": run["run_id"],
                "experiment": run["experiment"],
                "label": run["run_id"],
                "created_at": timestamp,
                "presentation": files.relative_to(source).as_posix(),
            }
        )
    return graph, records


def discover_runs(source: Path) -> list[dict[str, str]]:
    """Emit Demolab discovery rows after complete store validation."""
    return discover_store(source)[1]


# Report usage and input ancestry


def ancestry(records: dict[str, dict], roots: set[str]) -> set[str]:
    seen: set[str] = set()
    pending = list(roots)
    while pending:
        identity = pending.pop()
        if identity in seen:
            continue
        if identity not in records:
            raise PingstoreError(f"missing report or protected input: {identity}")
        seen.add(identity)
        pending.extend(ref["run_id"] for ref in records[identity]["inputs"].values())
    return seen


def report_usage(
    repo: Path,
    records: dict[str, dict],
    *,
    experiments: set[str] | None = None,
) -> tuple[dict, dict[str, set[str]]]:
    """Resolve all reports, allowing missing out-of-scope pins for scoped pruning.

    Keep unavailable pins in the selections so confirmation still detects edits.
    Existing selections and their ancestry remain validated across experiments.
    """
    latest = {}
    for identity, record in records.items():
        completed = record["execution"].get("completed_at")
        if record["stage"] != "present" or not completed:
            continue
        key = (
            datetime.fromisoformat(completed.replace("Z", "+00:00")).timestamp(),
            identity,
        )
        experiment = record["experiment"]
        if experiment not in latest or key > latest[experiment]:
            latest[experiment] = key
    selections = {}
    used_by = defaultdict(set)
    for article, metadata in articles(repo).items():
        selected = dict(metadata["runs"])
        for experiment in metadata["inputs"]:
            if experiment not in selected and experiment in latest:
                selected[experiment] = latest[experiment][1]
        roots = set()
        for experiment, identity in selected.items():
            record = records.get(identity)
            if record is None:
                # Check both the declared input and the run ID: a mismatched
                # pin must not hide a missing reference to the pruning scope.
                if (
                    experiments is not None
                    and experiment not in experiments
                    and identity.split("-", 1)[0] not in experiments
                ):
                    continue
                raise PingstoreError(f"report {article!r} names missing run {identity}")
            if record["experiment"] != experiment or record["stage"] != "present":
                raise PingstoreError(
                    f"report {article!r} pin is not a presentation for {experiment}"
                )
            roots.add(identity)
        selections[article] = {"inputs": metadata["inputs"], "runs": selected}
        for identity in ancestry(records, roots):
            used_by[identity].add(article)
    return selections, dict(used_by)


# Hash-bound local pruning

PLAN_SCHEMA = "pingstore.prune-plan/v5"


def _directory_bytes(directory: Path) -> int:
    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())


def _hidden_inputs(
    runs: Path, records: dict[str, dict], reasons: dict[str, set[str]]
) -> list[dict]:
    hidden = []
    for directory in sorted(
        path for path in runs.iterdir() if path.name.startswith(".")
    ):
        if not directory.is_dir() or directory.is_symlink():
            continue
        state = {"name": directory.name}
        manifest = directory / "run.json"
        reservation = directory / ".reservation.json"
        if manifest.is_file() and not manifest.is_symlink():
            state["run_json_sha256"] = file_sha256(manifest)
            record = load_json(manifest)
            inputs = record.get("inputs", {})
            if not isinstance(inputs, dict):
                raise PingstoreError(
                    f"{directory.name}: incomplete run has invalid inputs"
                )
            for reference in inputs.values():
                if not isinstance(reference, dict):
                    raise PingstoreError(
                        f"{directory.name}: incomplete run has invalid input"
                    )
                run_id = reference.get("run_id")
                parent = records.get(run_id)
                if parent is None or parent["payload_digest"] != reference.get(
                    "payload_digest"
                ):
                    raise PingstoreError(
                        f"{directory.name}: missing or changed input {run_id}"
                    )
                reasons[run_id].add(f"incomplete-input:{directory.name}")
        if reservation.is_file() and not reservation.is_symlink():
            state["reservation_sha256"] = file_sha256(reservation)
        hidden.append(state)
    return hidden


def _plan_hash(plan: dict) -> str:
    encoded = json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _experiment_scope(
    experiments: list[str] | tuple[str, ...] | set[str] | None,
    records: dict[str, dict],
) -> set[str] | None:
    if experiments is None:
        return None
    scope = set(experiments)
    if not scope:
        raise PingstoreError("experiment-scoped prune requires at least one experiment")
    invalid = sorted(value for value in scope if not EXPERIMENT_RE.fullmatch(value))
    if invalid:
        raise PingstoreError(f"invalid experiment filter: {', '.join(invalid)}")
    available = {record["experiment"] for record in records.values()}
    missing = sorted(scope - available)
    if missing:
        raise PingstoreError(f"experiment filter has no runs: {', '.join(missing)}")
    return scope


def _prune_rows(*, records, runs, keep, reasons, used_by, used, protected, markers):
    """Describe each retained or removable run without modifying storage."""
    rows = []
    for run_id, record in sorted(records.items()):
        directory = runs / run_id
        rows.append(
            {
                "run_id": run_id,
                "stage": record["stage"],
                "experiment": record["experiment"],
                "used": run_id in used,
                "used_by": sorted(used_by.get(run_id, ())),
                "protected": run_id in protected,
                "protected_marker_sha256": markers.get(run_id),
                "payload_digest": record["payload_digest"],
                "run_json_sha256": file_sha256(directory / "run.json"),
                "bytes": _directory_bytes(directory),
                "reasons": sorted(reasons[run_id]) if run_id in keep else ["unused"],
            }
        )
    return rows


def _protected_runs(*, runs, records, reasons):
    """Authenticate protection markers and retain their complete ancestry."""
    markers = {}
    for run_id in records:
        marker = runs / run_id / ".protected"
        if marker.is_symlink() or (marker.exists() and not marker.is_file()):
            raise PingstoreError(f"protection marker must be a regular file: {marker}")
        if marker.exists():
            markers[run_id] = file_sha256(marker)
            reasons[run_id].add("protected-marker")
    protected = ancestry(records, set(markers))
    for run_id in protected - set(markers):
        reasons[run_id].add("protected-ancestor")
    return markers, protected


def prune_plan(
    repo: Path,
    experiments: list[str] | tuple[str, ...] | set[str] | None = None,
    stages: list[str] | tuple[str, ...] | set[str] | None = None,
) -> dict:
    repo = repo.resolve()
    runs = repo / ".pingstore/runs"
    if not runs.is_dir() or runs.is_symlink():
        raise PingstoreError(f"prune requires a real runs directory: {runs}")
    for path in runs.iterdir():
        if path.is_symlink() or not path.is_dir():
            raise PingstoreError(
                f"prune does not accept unsupported runs entry: {path}"
            )
    # Unused runs can have missing historical inputs. Validate every payload,
    # then require complete, digest-matched ancestry for everything we retain.
    records = {
        path.name: validate_operational_run_directory(path)
        for path in sorted(runs.iterdir())
        if not path.name.startswith(".")
    }
    scope = _experiment_scope(experiments, records)
    stage_scope = set(stages) if stages is not None else None
    if stage_scope is not None and (
        not stage_scope or stage_scope - {"compute", "analyse", "present"}
    ):
        raise PingstoreError("invalid or empty stage filter")
    reasons: dict[str, set[str]] = defaultdict(set)
    if stage_scope is not None:
        for run_id, record in records.items():
            if record["stage"] not in stage_scope:
                reasons[run_id].add("out-of-stage-scope")

    if scope is not None:
        for run_id, record in records.items():
            if record["experiment"] not in scope:
                reasons[run_id].add("out-of-scope")

    selections, used_by = report_usage(repo, records, experiments=scope)
    used = set(used_by)
    for run_id, reports in used_by.items():
        reasons[run_id].update(f"used-by:{article}" for article in reports)
    markers, protected = _protected_runs(runs=runs, records=records, reasons=reasons)
    hidden = _hidden_inputs(runs, records, reasons)
    keep = ancestry(records, set(reasons))
    validated_run_graph(runs, keep)
    for run_id in keep - set(reasons):
        reasons[run_id].add("required-ancestor")
    watermarks = allocation_watermarks(repo / ".pingstore")

    rows = _prune_rows(
        records=records,
        runs=runs,
        keep=keep,
        reasons=reasons,
        used_by=used_by,
        used=used,
        protected=protected,
        markers=markers,
    )
    plan = {
        "schema": PLAN_SCHEMA,
        "policy": "keep-report-usage-manual-protection-incomplete-inputs-and-ancestry",
        "experiments": sorted(scope) if scope is not None else None,
        "stages": sorted(stage_scope) if stage_scope is not None else None,
        "allocation_watermarks": watermarks,
        "report_selections": selections,
        "hidden": hidden,
        "keep": [row for row in rows if row["run_id"] in keep],
        "prune": [row for row in rows if row["run_id"] not in keep],
    }
    return {**plan, "plan_hash": _plan_hash(plan)}


def _live_writer(directory: Path) -> bool:
    lock = directory / ".writer.lock"
    if not lock.is_file():
        return False
    try:
        pid = int(lock.read_text().strip())
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    return True


def _validate_survivors(runs: Path, expected: set[str]) -> None:
    records, _ = discover_store(runs)
    if set(records) != expected:
        raise PingstoreError("pruned store contains an unexpected completed-run set")


def _check_controls(repo: Path, runs: Path, plan: dict) -> None:
    """Detect manual protection or report-selection edits during confirmation."""

    records = {
        row["run_id"]: validate_run(load_json(runs / row["run_id"] / "run.json"))
        for row in plan["keep"] + plan["prune"]
    }
    scope = set(plan["experiments"]) if plan["experiments"] is not None else None
    selections, _ = report_usage(repo, records, experiments=scope)
    if selections != plan["report_selections"]:
        raise PingstoreError("report selections changed during pruning")
    for row in plan["keep"] + plan["prune"]:
        marker = runs / row["run_id"] / ".protected"
        if marker.is_symlink() or (marker.exists() and not marker.is_file()):
            raise PingstoreError("protection marker changed during pruning")
        digest = file_sha256(marker) if marker.exists() else None
        if digest != row["protected_marker_sha256"]:
            raise PingstoreError("protection marker changed during pruning")


def apply_prune(
    repo: Path,
    expected_hash: str,
    experiments: list[str] | tuple[str, ...] | set[str] | None = None,
    stages: list[str] | tuple[str, ...] | set[str] | None = None,
) -> dict:
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", expected_hash):
        raise PingstoreError(
            "--confirm requires the complete sha256 plan hash from --dry-run"
        )
    repo = repo.resolve()
    store = repo / ".pingstore"
    runs = store / "runs"
    staged = store / f".prune-{expected_hash[7:19]}-runs.tmp"
    previous = store / f".prune-{expected_hash[7:19]}-runs.old"
    with operation_lock(store, exclusive=True):
        try:
            plan = prune_plan(
                repo,
                experiments,
                stages,
            )
            if plan["plan_hash"] != expected_hash:
                raise PingstoreError(
                    f"prune plan changed: expected {expected_hash}, now {plan['plan_hash']}"
                )
            for state in plan["hidden"]:
                directory = runs / state["name"]
                if _live_writer(directory):
                    raise PingstoreError(
                        f"active writer prevents pruning: {directory.name}"
                    )
            if not plan["prune"]:
                return plan
            if staged.exists() or previous.exists():
                raise PingstoreError(
                    "unfinished prune staging directory requires recovery"
                )

            keep_ids = {row["run_id"] for row in plan["keep"]}
            # Persist counters before any run can disappear; dry-runs never write them.
            persist_watermarks(store, plan["allocation_watermarks"])
            staged.mkdir()
            for path in sorted(runs.iterdir()):
                if path.name.startswith(".") or path.name in keep_ids:
                    if path.is_dir() and not path.is_symlink():
                        shutil.copytree(
                            path,
                            staged / path.name,
                            copy_function=os.link,
                            symlinks=True,
                        )
                    else:
                        raise PingstoreError(
                            f"unsupported entry in runs directory: {path}"
                        )
            _validate_survivors(staged, keep_ids)
            _check_controls(repo, runs, plan)
            os.replace(runs, previous)
            try:
                os.replace(staged, runs)
                _validate_survivors(runs, keep_ids)
                _check_controls(repo, previous, plan)
            except BaseException:
                if runs.exists():
                    os.replace(runs, staged)
                os.replace(previous, runs)
                raise
            shutil.rmtree(previous)
            return plan
        finally:
            if staged.exists() and runs.exists():
                shutil.rmtree(staged)


def render_prune_plan(plan: dict) -> str:
    keep_bytes = sum(row["bytes"] for row in plan["keep"])
    prune_bytes = sum(row["bytes"] for row in plan["prune"])
    lines = [
        f"Plan: {plan['plan_hash']}",
        "Scope: "
        + (
            ", ".join(plan["experiments"]) if plan["experiments"] else "all experiments"
        ),
        "Stages: " + (", ".join(plan.get("stages") or []) or "all stages"),
        "Policy: used report ancestry + manual .protected ancestry + incomplete inputs",
        f"Used: {sum(row['used'] for row in plan['keep'])} runs",
        f"Protected: {sum(row['protected'] for row in plan['keep'])} runs",
        f"Keep: {len(plan['keep'])} runs ({keep_bytes / 2**30:.2f} GiB)",
        f"Prune: {len(plan['prune'])} runs ({prune_bytes / 2**30:.2f} GiB)",
        "",
        "KEEP",
    ]
    lines.extend(
        f"{row['run_id']}\t{row['bytes']}\t{'used' if row['used'] else 'unused'}\t{','.join(row['reasons'])}"
        for row in plan["keep"]
    )
    lines.extend(["", "PRUNE"])
    lines.extend(
        f"{row['run_id']}\t{row['bytes']}\t{'used' if row['used'] else 'unused'}\t{','.join(row['reasons'])}"
        for row in plan["prune"]
    )
    scope_args = " ".join(
        f"--experiment {shlex.quote(experiment)}"
        for experiment in (plan["experiments"] or [])
    )
    command = "uv run pingstore prune"
    if scope_args:
        command += f" {scope_args}"
    for stage in plan.get("stages") or []:
        command += f" --stage {shlex.quote(stage)}"
    command += f" --confirm {plan['plan_hash']}"
    lines.extend(["", f"Confirm with: {command}"])
    return "\n".join(lines)


# Verified R2 backups


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
        if path.is_file() and path != directory / ".protected":
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


def records_for_backup(source, destination):
    records = {
        path.name: validate_operational_run_directory(path)
        for path in sorted(source.iterdir())
        if not path.name.startswith(".")
    }
    missing_inputs = {
        reference["run_id"]
        for record in records.values()
        for reference in record["inputs"].values()
    } - records.keys()
    if not missing_inputs:
        return discover_store(source)[0]

    remote = read_catalog(destination)
    combined = {identity: entry["record"] for identity, entry in remote["runs"].items()}
    combined.update(records)
    for identity, record in records.items():
        for reference in record["inputs"].values():
            parent = combined.get(reference["run_id"])
            if (
                parent is None
                or parent["payload_digest"] != reference["payload_digest"]
            ):
                raise PingstoreError(
                    f"{identity}: missing or changed cataloged input {reference['run_id']}"
                )
    return records


def backup_plan(source, destination):
    records = records_for_backup(source, destination)
    remote = inventory(destination)
    rows, new_ids, existing_ids = [], [], []
    for identity, record in sorted(records.items()):
        files = local_files(source, identity)
        remote_files = {
            path: size
            for path, size in remote.items()
            if path.startswith(identity + "/") and path != identity + "/.protected"
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
        "16",
        "--checkers",
        "16",
        "--stats",
        "30s",
        "--stats-one-line",
        "--log-level",
        "INFO",
    )


def apply_backup(source, destination, plan):
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
            "16",
            "--stats",
            "30s",
            "--stats-one-line",
            "--log-level",
            "INFO",
        )
        records = records_for_backup(source, destination)
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
                if path.startswith(identity + "/") and path != identity + "/.protected"
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
            "16",
            "--log-level",
            "INFO",
        )
        uploaded = inventory(destination)
        for identity in plan["new_runs"]:
            actual = {
                path: size
                for path, size in uploaded.items()
                if path.startswith(identity + "/") and path != identity + "/.protected"
            }
            if actual != expected[identity]["files"]:
                raise PingstoreError("final backup inventory differs: " + identity)
    print(
        f"Backed up and verified {len(plan['new_runs'])} new runs; {len(plan['existing_runs'])} existing runs unchanged."
    )


def backup_runs(
    source: Path, destination: str, *, dry_run: bool, confirm: str | None
) -> int:
    """Inspect a backup plan or apply an explicitly confirmed, unchanged plan."""
    try:
        with operation_lock(source.parent, exclusive=False):
            plan = backup_plan(source, destination)
            if dry_run:
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
                if plan["plan_hash"] != confirm:
                    raise PingstoreError("backup plan changed; run --dry-run again")
                apply_backup(source, destination, plan)

                publish_catalog(source, destination, plan)
        return 0
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"pingstore backup: {exc}", file=sys.stderr)
        return 1


# Checksummed public catalogs

SCHEMA = "pingstore.catalog/v1"


def catalog_path(destination):
    return destination.rsplit("/", 1)[0] + "/catalog.json"


def read_catalog(destination):

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


def replace_catalog(destination, previous, value):
    """Conditionally replace the mutable index, preserving concurrent publications."""

    validate_catalog(value)
    with tempfile.TemporaryDirectory(prefix="pingstore-catalog-publish-") as temporary:
        path = Path(temporary) / "catalog.json"
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


def publish_catalog(source: Path, destination: str, plan: dict):

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
        replace_catalog(destination, previous, value)
    print(f"Published checksummed catalog for {len(entries)} runs.", flush=True)


# Verified local eviction


def eviction_plan(
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


def apply_eviction(
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
        plan = eviction_plan(repo, destination, run_ids, asset_base)
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
        if eviction_plan(repo, destination, run_ids, asset_base) != plan:
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


def render_eviction_plan(plan):
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


# Retired PDF backup interface


def paper_backup_plan(source: Path, destination: str):
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


def apply_paper_backup(source: Path, destination: str, expected_hash: str):
    raise PingstoreError(
        "Public PDF uploads are disabled. Keep PDFs local and publish source links instead."
    )


# Campaign payload contracts

CONTRACT_VERSION = "pingstore.campaign/v1"
MANIFEST_NAMES = frozenset({"run.json", "inventory.json"})
KINDS = frozenset({"adhoc", "campaign", "legacy"})
STATUSES = frozenset({"planned", "running", "complete", "failed", "legacy"})
ROLES = frozenset({"state", "derived", "log"})
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
PREFIXED_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class ContractError(ValueError):
    """Raised when a campaign document or payload violates the contract."""


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{label} must be an object")
    return value


def _require_string(value: Any, label: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise ContractError(f"{label} must be a non-empty string")
    return value


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise ContractError(f"{label} must be a lowercase SHA-256")
    return value


def _require_utc_timestamp(value: Any, label: str) -> str:
    text = _require_string(value, label)
    if not text.endswith("Z"):
        raise ContractError(f"{label} must be an RFC 3339 UTC timestamp ending in Z")
    try:
        datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError as exc:
        raise ContractError(f"{label} is not a valid timestamp: {text!r}") from exc
    return text


def load_campaign_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise ContractError(f"missing contract file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ContractError(f"invalid JSON in {path}: {exc}") from exc
    return _require_mapping(value, str(path))


def write_campaign_json_atomic(path: Path, value: dict[str, Any]) -> None:
    """Write one JSON document without exposing a partial destination file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_manifest_execution(*, value):
    """Validate execution identity, command and graph provenance."""
    execution = _require_mapping(value.get("execution"), "run.json execution")
    experiment = execution.get("experiment")
    collection = execution.get("collection")
    for item, label in ((experiment, "experiment"), (collection, "collection")):
        if item is not None:
            _require_string(item, f"run.json execution.{label}")
    if experiment is None and collection is None:
        raise ContractError("run.json execution needs an experiment or collection")
    command = execution.get("command")
    if (
        not isinstance(command, list)
        or not command
        or not all(isinstance(part, str) and part for part in command)
    ):
        raise ContractError(
            "run.json execution.command must be a non-empty string array"
        )
    executor = execution.get("executor")
    if executor is not None and executor not in {"legacy", "graph"}:
        raise ContractError("run.json execution.executor must be legacy or graph")
    graph_digest = execution.get("graph_digest")
    training_digest = execution.get("training_digest")
    for digest, label in (
        (graph_digest, "graph_digest"),
        (training_digest, "training_digest"),
    ):
        if digest is not None and (
            not isinstance(digest, str) or not PREFIXED_SHA256_RE.fullmatch(digest)
        ):
            raise ContractError(
                f"run.json execution.{label} must be a prefixed SHA-256 or null"
            )
    if executor == "graph" and graph_digest is None:
        raise ContractError("graph execution requires execution.graph_digest")
    if executor in {None, "legacy"} and (
        graph_digest is not None or training_digest is not None
    ):
        raise ContractError("legacy execution cannot record graph/training digests")


def _validate_manifest_source(*, value):
    """Validate source revision, lockfile and native tool versions."""
    source = _require_mapping(value.get("source"), "run.json source")
    commit = source.get("git_commit")
    if commit is not None and (
        not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit)
    ):
        raise ContractError(
            "run.json source.git_commit must be 40 lowercase hex chars or null"
        )
    if source.get("git_clean") is not None and not isinstance(
        source.get("git_clean"), bool
    ):
        raise ContractError("run.json source.git_clean must be boolean or null")
    lockfile = source.get("lockfile")
    if lockfile is not None:
        lock = _require_mapping(lockfile, "run.json source.lockfile")
        validate_relative_path(lock.get("path"), "run.json source.lockfile.path")
        _require_sha256(lock.get("sha256"), "run.json source.lockfile.sha256")
    versions = source.get("tool_versions")
    if versions is not None:
        version_map = _require_mapping(versions, "run.json source.tool_versions")
        if set(version_map) != {"snnlang", "snnsim", "snnviz"}:
            raise ContractError(
                "run.json source.tool_versions must identify snnlang, snnsim, and snnviz"
            )
        semver = re.compile(
            r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
            r"(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
            r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
        )
        for name, version in version_map.items():
            if not isinstance(version, str) or not semver.fullmatch(version):
                raise ContractError(
                    f"run.json source.tool_versions.{name} must be SemVer"
                )


def validate_run_manifest(value: dict[str, Any]) -> dict[str, Any]:
    if value.get("contract_version") != CONTRACT_VERSION:
        raise ContractError(f"run.json contract_version must be {CONTRACT_VERSION!r}")
    _require_string(value.get("run_id"), "run.json run_id")
    if value.get("kind") not in KINDS:
        raise ContractError(f"run.json kind must be one of {sorted(KINDS)}")
    if value.get("status") not in STATUSES:
        raise ContractError(f"run.json status must be one of {sorted(STATUSES)}")
    _require_utc_timestamp(value.get("created_at_utc"), "run.json created_at_utc")

    _validate_manifest_source(value=value)

    _validate_manifest_execution(value=value)

    if not isinstance(value.get("upstream"), list):
        raise ContractError("run.json upstream must be an array")
    archive = value.get("archive")
    if archive is not None:
        archive_obj = _require_mapping(archive, "run.json archive")
        _require_string(archive_obj.get("archive_id"), "run.json archive.archive_id")
        uri = _require_string(archive_obj.get("uri"), "run.json archive.uri")
        if not uri.startswith(("r2://", "file://")):
            raise ContractError("run.json archive.uri must start with r2:// or file://")
    notes = value.get("provenance_notes")
    if not isinstance(notes, str):
        raise ContractError("run.json provenance_notes must be a string")
    return value


def validate_relative_path(value: Any, label: str = "path") -> str:
    text = _require_string(value, label)
    if "\\" in text:
        raise ContractError(f"{label} must use POSIX separators")
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != text:
        raise ContractError(f"{label} must be a normalized relative POSIX path")
    if text in MANIFEST_NAMES:
        raise ContractError(f"{label} cannot inventory {text}")
    return text


def canonical_payload_digest(files: list[dict[str, Any]]) -> str:
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def validate_inventory(value: dict[str, Any]) -> dict[str, Any]:
    if value.get("contract_version") != CONTRACT_VERSION:
        raise ContractError(
            f"inventory.json contract_version must be {CONTRACT_VERSION!r}"
        )
    _require_string(value.get("run_id"), "inventory.json run_id")
    _require_utc_timestamp(
        value.get("generated_at_utc"), "inventory.json generated_at_utc"
    )
    files = value.get("files")
    if not isinstance(files, list):
        raise ContractError("inventory.json files must be an array")

    paths: list[str] = []
    for index, raw in enumerate(files):
        row = _require_mapping(raw, f"inventory.json files[{index}]")
        path = validate_relative_path(row.get("path"), f"files[{index}].path")
        paths.append(path)
        size = row.get("size_bytes")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ContractError(
                f"files[{index}].size_bytes must be a non-negative integer"
            )
        _require_sha256(row.get("sha256"), f"files[{index}].sha256")
        if row.get("role") not in ROLES:
            raise ContractError(f"files[{index}].role must be one of {sorted(ROLES)}")
        if set(row) != {"path", "size_bytes", "sha256", "role"}:
            raise ContractError(
                f"files[{index}] must contain only path, size_bytes, sha256, role"
            )

    if paths != sorted(paths):
        raise ContractError("inventory.json files must be sorted by path")
    if len(paths) != len(set(paths)):
        raise ContractError("inventory.json contains duplicate paths")
    if value.get("file_count") != len(files):
        raise ContractError("inventory.json file_count does not match files")
    total = sum(row["size_bytes"] for row in files)
    if value.get("total_size_bytes") != total:
        raise ContractError("inventory.json total_size_bytes does not match files")
    _require_sha256(value.get("payload_digest"), "inventory.json payload_digest")
    if value["payload_digest"] != canonical_payload_digest(files):
        raise ContractError("inventory.json payload_digest does not match files")
    return value


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _role_for(relative: Path) -> str:
    first = relative.parts[0] if relative.parts else ""
    if first == "derived":
        return "derived"
    if first == "logs" or relative.suffix == ".log":
        return "log"
    return "state"


def inventory_payload(
    root: Path, *, run_id: str, generated_at_utc: str | None = None
) -> dict[str, Any]:
    root = root.resolve()
    if not root.is_dir():
        raise ContractError(f"run root is not a directory: {root}")
    files: list[dict[str, Any]] = []
    for item in sorted(
        root.rglob("*"), key=lambda path: path.relative_to(root).as_posix()
    ):
        if item.is_symlink():
            raise ContractError(f"payload symlinks are not supported in v1: {item}")
        if not item.is_file() or item.relative_to(root).as_posix() in MANIFEST_NAMES:
            continue
        relative = item.relative_to(root)
        files.append(
            {
                "path": relative.as_posix(),
                "size_bytes": item.stat().st_size,
                "sha256": _hash_file(item),
                "role": _role_for(relative),
            }
        )
    timestamp = generated_at_utc or datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    ).replace("+00:00", "Z")
    result = {
        "contract_version": CONTRACT_VERSION,
        "run_id": run_id,
        "generated_at_utc": timestamp,
        "file_count": len(files),
        "total_size_bytes": sum(row["size_bytes"] for row in files),
        "payload_digest": canonical_payload_digest(files),
        "files": files,
    }
    return validate_inventory(result)


def verify_payload(root: Path, inventory: dict[str, Any]) -> None:
    expected = validate_inventory(inventory)
    actual = inventory_payload(
        root,
        run_id=expected["run_id"],
        generated_at_utc=expected["generated_at_utc"],
    )
    for field in ("file_count", "total_size_bytes", "payload_digest", "files"):
        if actual[field] != expected[field]:
            raise ContractError(
                f"payload does not match inventory.json: {field} differs"
            )


def provenance_gaps(run: dict[str, Any] | None) -> list[str]:
    if run is None:
        return ["missing run.json"]
    source = run["source"]
    gaps = []
    if source.get("git_commit") is None:
        gaps.append("unknown generating Git commit")
    if source.get("git_clean") is None:
        gaps.append("unknown clean-tree state")
    if source.get("lockfile") is None:
        gaps.append("unknown lockfile identity")
    return gaps


# Scoped removal of unused remote backups


def _remote_prune_graph(repo, catalog):
    local = validated_run_graph(repo / ".pingstore/runs")
    records = {identity: entry["record"] for identity, entry in catalog["runs"].items()}
    for identity, record in local.items():
        if identity in records and records[identity] != record:
            raise PingstoreError(f"local and remote records differ: {identity}")
        records[identity] = record
    return local, records


def _remote_prune_objects(catalog, remote, scope):
    scoped = {
        name: size
        for name, size in remote.items()
        if name.split("/", 1)[0].split("-", 1)[0] in scope
    }
    expected = {
        f"{identity}/{name}": item["bytes"]
        for identity, entry in catalog["runs"].items()
        if entry["record"]["experiment"] in scope
        for name, item in entry["files"].items()
    }
    actual = {
        name: size for name, size in scoped.items() if not name.endswith("/.protected")
    }
    if actual != expected:
        raise PingstoreError("scoped remote objects differ from the verified catalog")
    return scoped


def remote_prune_plan(repo: Path, destination: str, experiments: list[str]) -> dict:
    """Plan scoped deletion, retaining report, protected and unfinished ancestry."""
    repo = repo.resolve()
    scope = set(experiments)
    if not scope or any(not EXPERIMENT_RE.fullmatch(value) for value in scope):
        raise PingstoreError("remote pruning requires explicit valid experiments")
    catalog = read_catalog(destination)
    remote = inventory(destination)
    local, records = _remote_prune_graph(repo, catalog)
    selections, used_by = report_usage(repo, records)
    used = set(used_by)
    selected = {
        identity for identity in used if records[identity]["experiment"] in scope
    }
    if not selected or selected - catalog["runs"].keys():
        raise PingstoreError(
            "back up the complete current report runset before remote pruning"
        )
    protected = {
        identity
        for identity in records
        if (repo / ".pingstore/runs" / identity / ".protected").exists()
        or identity + "/.protected" in remote
    }
    pending = defaultdict(set)
    hidden = _hidden_inputs(repo / ".pingstore/runs", records, pending)
    outside = {
        identity
        for identity, record in records.items()
        if record["experiment"] not in scope
    }
    keep = ancestry(records, used | protected | set(pending) | outside)
    candidates = {
        identity
        for identity, entry in catalog["runs"].items()
        if entry["record"]["experiment"] in scope and identity not in keep
    }
    scoped_objects = _remote_prune_objects(catalog, remote, scope)
    revised = dict(
        catalog,
        runs={
            identity: entry
            for identity, entry in catalog["runs"].items()
            if identity not in candidates
        },
    )
    validate_catalog(revised)
    plan = dict(
        schema="pingstore.remote-prune-plan/v1",
        root=str(repo),
        destination=destination,
        experiments=sorted(scope),
        catalog=catalog,
        report_selections=selections,
        hidden=hidden,
        protected=sorted(protected),
        remote_objects=scoped_objects,
        local_controls={
            identity: file_sha256(repo / ".pingstore/runs" / identity / "run.json")
            for identity in local
        },
        keep=sorted(
            identity
            for identity in catalog["runs"]
            if identity in keep and records[identity]["experiment"] in scope
        ),
        prune=sorted(candidates),
    )
    return {**plan, "plan_hash": _plan_hash(plan)}


def apply_remote_prune(
    repo: Path, destination: str, experiments: list[str], expected_hash: str
) -> dict:
    """Publish a valid surviving index before deleting only the planned objects."""
    with operation_lock(repo.resolve() / ".pingstore", exclusive=True):
        plan = remote_prune_plan(repo, destination, experiments)
        if plan["plan_hash"] != expected_hash:
            raise PingstoreError("remote prune plan changed; run --dry-run again")
        if not plan["prune"]:
            return plan
        previous = plan["catalog"]
        removed = set(plan["prune"])
        revised = dict(
            previous,
            updated_at=datetime.now(timezone.utc).isoformat(),
            runs={
                identity: entry
                for identity, entry in previous["runs"].items()
                if identity not in removed
            },
        )
        replace_catalog(destination, previous, revised)
        paths = {
            f"{identity}/{name}"
            for identity in removed
            for name in previous["runs"][identity]["files"]
        }
        with tempfile.TemporaryDirectory(prefix="pingstore-remote-prune-") as temporary:
            names = file_list(Path(temporary), "delete.txt", paths)
            if read_catalog(destination) != revised:
                raise PingstoreError("catalog changed before object deletion")
            rclone("delete", destination, "--files-from-raw", names)
        after = inventory(destination)
        if any(name.split("/", 1)[0] in removed for name in after):
            raise PingstoreError(
                "remote prune left superseded objects; inspect cleanup plan"
            )
        for name, size in plan["remote_objects"].items():
            if name.split("/", 1)[0] not in removed and after.get(name) != size:
                raise PingstoreError("retained remote inventory changed during pruning")
        if read_catalog(destination) != revised:
            raise PingstoreError("catalog changed during remote pruning")
    return plan


def render_remote_prune_plan(plan: dict) -> str:
    size = sum(
        file["bytes"]
        for identity in plan["prune"]
        for file in plan["catalog"]["runs"][identity]["files"].values()
    )
    return "\n".join(
        [
            f"Destination: {plan['destination']}",
            f"Scope: {', '.join(plan['experiments'])}",
            f"Keep: {len(plan['keep'])} remote runs",
            f"Prune: {len(plan['prune'])} remote runs ({size / 2**20:.2f} MiB)",
            *plan["prune"],
            f"Plan: {plan['plan_hash']}",
        ]
    )
