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
from pathlib import Path
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
    if any(path.is_symlink() for path in (root, *root.parents)):
        raise PingstoreError("Pingstore operation paths must not use symlinks")
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

    paths = ("experiments", "libraries", "pyproject.toml", "uv.lock")
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


# Current runsets and the derived reporting contract. Scientific input selection
# continues using source_run; none of these reporting policies changes input pins.

INDEX_SCHEMA = "pingstore.index/v1"


def _completed_time(record):
    completed = record["execution"].get("completed_at")
    if not completed:
        return None
    try:
        timestamp = datetime.fromisoformat(completed.replace("Z", "+00:00"))
        if timestamp.utcoffset() is None:
            raise ValueError("timezone missing")
        return timestamp.timestamp()
    except (AttributeError, ValueError) as exc:
        raise PingstoreError(f"invalid completion date: {record['run_id']}") from exc


def _runset(records, identity):
    """Validate only this root's exact ancestry, including cycles and pin digests."""
    visiting, visited = set(), set()

    def visit(run_id):
        if run_id in visiting:
            raise PingstoreError(f"input cycle at {run_id}")
        if run_id in visited:
            return
        record = records.get(run_id)
        if record is None:
            raise PingstoreError(f"missing input {run_id}")
        visiting.add(run_id)
        for reference in record["inputs"].values():
            parent = records.get(reference["run_id"])
            if (
                parent is None
                or parent["payload_digest"] != reference["payload_digest"]
            ):
                raise PingstoreError(
                    f"{run_id}: missing or changed input {reference['run_id']}"
                )
            visit(reference["run_id"])
        visiting.remove(run_id)
        visited.add(run_id)

    visit(identity)
    return sorted(visited)


def _select_runsets(*, records, local, errors, completed):
    selections, runsets, selection_errors = {}, {}, {}
    experiments = sorted({identity.split("-", 1)[0] for identity in records | errors})
    for experiment in experiments:
        # A malformed local presentation blocks fallback even when no valid local
        # presentation is left. Hidden unfinished stages are not candidates.
        broken = [
            identity
            for identity in errors
            if identity.startswith(experiment + "-") and identity.endswith("-present")
        ]
        if broken:
            selection_errors[experiment] = "; ".join(
                f"{identity}: {errors[identity]}" for identity in sorted(broken)
            )
            continue
        candidates = [
            identity
            for identity, record in records.items()
            if record["experiment"] == experiment and record["stage"] == "present"
        ]
        local_candidates = [identity for identity in candidates if identity in local]
        if local_candidates:
            candidates = local_candidates
        if not candidates:
            continue
        identity = max(candidates, key=lambda run_id: (completed[run_id], run_id))
        try:
            members = _runset(records, identity)
            invalid = set(members) & errors.keys()
            if invalid:
                raise PingstoreError(
                    "invalid local evidence: " + ", ".join(sorted(invalid))
                )
        except PingstoreError as exc:
            selection_errors[experiment] = str(exc)
            continue
        selections[experiment] = identity
        runsets[experiment] = members
    return selections, runsets, selection_errors


def _report_record(record):
    """Keep the derived index small; scientific configuration stays in run.json."""
    if record is None:
        return None
    result = {
        key: record[key]
        for key in (
            "schema",
            "run_id",
            "experiment",
            "stage",
            "collection",
            "origin",
            "created_at",
            "inputs",
            "payload_digest",
        )
    }
    result["execution"] = {
        key: value
        for key, value in record["execution"].items()
        if key in {"started_at", "completed_at", "operation"}
    }
    if "scientific_execution" in record:
        result["scientific_execution"] = {
            key: value
            for key, value in record["scientific_execution"].items()
            if key in {"started_at", "completed_at", "origin"}
        }
    if "completed_hpc_import" in record:
        imported = record["completed_hpc_import"]
        result["completed_hpc_import"] = {
            "source_origin": imported["source_origin"],
            "source_execution": {
                key: value
                for key, value in imported["source_execution"].items()
                if key in {"started_at", "completed_at"}
            },
        }
    return result


def _index_rows(
    *,
    records,
    errors,
    local,
    remote,
    reasons,
    remote_known,
    source,
    local_errors,
    verified_local,
    publication_known,
    inventory_known,
    remote_errors,
    remote_present,
):
    local_holdings = local.keys() | local_errors.keys()
    rows = {}
    for identity in sorted(records.keys() | errors.keys()):
        record = records.get(identity)
        current = sorted(
            reason[8:] for reason in reasons[identity] if reason.startswith("current:")
        )
        row_reasons = sorted(reasons[identity])
        unresolved = []
        if identity in errors:
            unresolved.append(errors[identity])
        if not remote_known and source != "local":
            unresolved.append("R2 inventory unavailable")
        # Surviving deployed pages are not proved by backup presence or recency.
        if not publication_known:
            unresolved.append(
                "active and incoming publication evidence not established"
            )
        if source != "local" and not inventory_known:
            unresolved.append("authenticated R2 inventory snapshot not established")
        status = (
            "Current"
            if current and identity not in errors
            else "Required"
            if any(not reason.startswith("unresolved-") for reason in row_reasons)
            and identity not in errors
            else "Unused"
            if publication_known
            and inventory_known
            and remote_known
            and identity not in errors
            else "Unresolved"
        )
        rows[identity] = dict(
            record=_report_record(record),
            current=current,
            status=status,
            reasons=row_reasons,
            unresolved=unresolved,
            storage="Both"
            if identity in local_holdings and identity in remote_present
            else "R2"
            if identity in remote_present
            else "Local"
            if identity in local_holdings
            else "Unavailable"
            if inventory_known and identity in remote
            else "Unknown",
            verification=dict(
                local="byte-verified"
                if identity in verified_local and identity not in errors
                else "metadata-validated"
                if identity in local and identity not in errors
                else "invalid"
                if identity in local_holdings and identity in errors
                else "absent",
                r2="missing"
                if identity in remote_errors and identity not in remote_present
                else "inventory-unresolved"
                if identity in remote_errors
                else "catalog-verified"
                if identity in remote
                else "unknown"
                if identity in errors and identity not in local_holdings
                else "absent"
                if remote_known
                else "unknown",
            ),
        )
    return rows


def _index_record_set(*, local, remote, errors):
    records = {**remote, **local}
    for identity, record in records.items():
        validate_run(record)
        if record["run_id"] != identity:
            raise PingstoreError(f"record identity mismatch: {identity}")
        if (
            identity in local
            and identity in remote
            and local[identity] != remote[identity]
        ):
            errors[identity] = "local and R2 records differ"
    completed = {}
    for identity, record in records.items():
        timestamp = _completed_time(record)
        if timestamp is not None:
            completed[identity] = timestamp
    records = {
        identity: record
        for identity, record in records.items()
        if identity in completed
    }
    return records, completed


def _index_inventory_roots(
    *,
    inventory_state,
    operation,
    source,
    local,
    records,
    runsets,
    selection_errors,
    reasons,
    errors,
):
    inventory_state = inventory_state or dict(known=True, unresolved={}, protected=[])
    remote_errors = (
        dict(inventory_state.get("unresolved", {})) if source != "local" else {}
    )
    for experiment, members in runsets.items():
        invalid = (set(members) & remote_errors.keys()) - local.keys()
        if invalid:
            selection_errors[experiment] = (
                f"R2 inventory unresolved for {sorted(invalid)}"
            )
    for identity in inventory_state.get("protected", []):
        if identity in records:
            for member in _runset(records, identity):
                reasons[member].add(f"protection:r2:{identity}")
    if remote_errors:
        for identity in records:
            reasons[identity].add("unresolved-remote-holdings")
        errors.update(
            {
                identity: error
                for identity, error in remote_errors.items()
                if identity not in local
            }
        )
    if operation:
        label = f"unfinished-operation:{operation['kind']}:{operation['plan_hash']}"
        for identity in records:
            reasons[identity].add(label)
        for identity in operation.get("run_ids", []):
            if identity not in records:
                errors[identity] = label + "; copy presence is unresolved"
    return inventory_state, remote_errors


def resolve_run_index(
    local: dict[str, dict],
    remote: dict[str, dict],
    *,
    source="auto",
    reports=None,
    protections=None,
    unfinished=None,
    remote_known=True,
    local_errors=None,
    verified_local=None,
    publications=None,
    inventory_state=None,
    operation=None,
):
    """Resolve deterministic current runsets and explain retention from metadata.

    Callers authenticate payloads/catalogs before supplying records. Broken local
    copies are never replaced by remote evidence. Publication retention remains
    unresolved until verified active/incoming deployments are established;
    this reporting index is deliberately not a deletion authorization.
    """
    if source not in {"auto", "local", "remote"}:
        raise PingstoreError(f"invalid run source: {source}")
    reports = reports or {}
    protections = protections or {}
    unfinished = unfinished or {}
    local_errors = local_errors or {}
    remote_known = remote_known and source != "local"
    local = {} if source == "remote" else local
    remote = {} if source == "local" else remote
    local_errors = {} if source == "remote" else local_errors
    errors = dict(local_errors)
    records, completed = _index_record_set(local=local, remote=remote, errors=errors)
    selections, runsets, selection_errors = _select_runsets(
        records=records, local=local, errors=errors, completed=completed
    )
    reasons = defaultdict(set)
    for experiment, members in runsets.items():
        for member in members:
            reasons[member].add(f"current:{experiment}")
    report_selections = {}
    for article, metadata in sorted(reports.items()):
        # Current reporting never follows historical manual pins. Existing pins
        # remain visible for migration without selecting historical evidence.
        selected = {
            experiment: selections[experiment]
            for experiment in metadata["inputs"]
            if experiment in selections
        }
        report_selections[article] = selected
        for experiment in selected:
            for member in runsets[experiment]:
                reasons[member].add(f"report:{article}")
    for label, roots in (("protection", protections), ("unfinished", unfinished)):
        for identity, detail in sorted(roots.items()):
            try:
                members = _runset(records, identity)
            except PingstoreError as exc:
                errors[identity] = str(exc)
                continue
            for member in members:
                reasons[member].add(f"{label}:{detail}")
    publications = publications or empty_publications()
    _publication_reasons(publications, records, reasons, complete=False)
    inventory_state, remote_errors = _index_inventory_roots(
        inventory_state=inventory_state,
        operation=operation,
        source=source,
        local=local,
        records=records,
        runsets=runsets,
        selection_errors=selection_errors,
        reasons=reasons,
        errors=errors,
    )
    rows = _index_rows(
        records=records,
        errors=errors,
        local=local,
        remote=remote,
        reasons=reasons,
        local_errors=local_errors,
        verified_local=set(verified_local or ()),
        publication_known=publications["known"],
        inventory_known=inventory_state.get("known", False),
        remote_errors=remote_errors,
        remote_present=set(inventory_state.get("present", remote.keys())),
        remote_known=remote_known,
        source=source,
    )
    value = dict(
        schema=INDEX_SCHEMA,
        source=source,
        remote_known=remote_known,
        selections=selections,
        runsets=runsets,
        selection_errors=selection_errors,
        report_selections=report_selections,
        report_inputs={
            article: metadata["inputs"] for article, metadata in sorted(reports.items())
        },
        legacy_pins={
            article: metadata["runs"]
            for article, metadata in sorted(reports.items())
            if metadata.get("runs")
        },
        publication="known" if publications["known"] else "unresolved",
        publications=publications,
        operation={
            key: operation[key]
            for key in ("kind", "plan_hash", "run_ids")
            if key in operation
        }
        if operation
        else None,
        inventory_generation=inventory_state.get("generation"),
        publication_unavailable=sorted(
            {
                identity
                for manifest in (
                    [publications["active"]] if publications.get("active") else []
                )
                + list(publications["incoming"].values())
                for identity in manifest["runs"]
                if identity not in records
            }
        ),
        runs=rows,
    )
    return {**value, "generation": _plan_hash(value)}


def _local_index_signature(directory):
    """Read each entry's metadata once, without following payload symlinks."""
    signature = {}

    def visit(path, prefix):
        with os.scandir(path) as entries:
            for entry in entries:
                # Protection contents do not change scientific evidence. Invalid
                # marker types still enter the fingerprint and force validation.
                if entry.name == ".protected" and entry.is_file(follow_symlinks=False):
                    continue
                name = prefix + entry.name
                metadata = entry.stat(follow_symlinks=False)
                signature[name] = [
                    metadata.st_size,
                    metadata.st_mtime_ns,
                    metadata.st_ctime_ns,
                    metadata.st_ino,
                    metadata.st_dev,
                ]
                if entry.is_dir(follow_symlinks=False):
                    visit(entry.path, name + "/")

    visit(directory, "")
    return signature


def _local_index_records(runs, cache, *, verify):
    """Index metadata without reading arrays; explicit verification authenticates bytes."""
    records, errors, revised = {}, {}, {}
    if not runs.exists():
        return records, errors, revised
    if runs.is_symlink() or not runs.is_dir():
        raise PingstoreError(f"runs must be a real directory: {runs}")
    for directory in sorted(runs.iterdir()):
        identity = directory.name
        if identity.startswith("."):
            continue
        if not STAGE_ID_RE.fullmatch(identity):
            errors[identity] = "unsupported run identity"
            continue
        try:
            if directory.is_symlink() or not directory.is_dir():
                raise PingstoreError(f"run must be a real directory: {directory}")
            signature = _local_index_signature(directory)
            previous = cache.get(identity, {})
            manifest = directory / "run.json"
            if manifest.is_symlink() or not manifest.is_file():
                raise PingstoreError(f"run.json must be a regular file: {manifest}")
            record = validate_run(load_json(manifest))
            unchanged = (
                signature == previous.get("signature")
                and previous.get("record") == record
            )
            # A matching versioned fingerprint already passed canonical layout
            # validation. Recheck every pathname and metadata fingerprint on
            # every request; changed/new runs still use the full validator.
            if not unchanged or previous.get("layout_version") != 1:
                validate_layout(directory)
            if (
                record["run_id"] != identity
                or "data_root" in record
                or "export_root" in record
            ):
                raise PingstoreError(
                    f"invalid operational run identity/layout: {identity}"
                )
            if previous and not unchanged and not verify:
                raise PingstoreError(
                    "local files changed since indexing; inspect with status --verify"
                )
            verified = unchanged and previous.get("verified", False)
            if verify and not verified:
                record = validate_operational_run_directory(directory)
                verified = True
            records[identity] = record
            revised[identity] = dict(
                signature=signature, record=record, verified=verified, layout_version=1
            )
        except (OSError, ValueError) as exc:
            errors[identity] = str(exc)
            if identity in cache:
                revised[identity] = cache[identity]
    return records, errors, revised


def reporting_index(
    repo: Path,
    *,
    runs=None,
    catalog=None,
    source="auto",
    remote_known=True,
    verify=False,
    publications=None,
    operation=None,
    inventory_state=None,
):
    """Build reporting metadata; byte verification is an explicit offline operation."""
    repo = repo.resolve()
    runs = (runs or repo / ".pingstore/runs").absolute()
    if source == "remote" and catalog is None:
        raise PingstoreError("remote reporting requires a verified catalog")
    namespace = hashlib.sha256(str(runs).encode()).hexdigest()[:12]
    cache_path = runs.parent / f".report-index-cache-{namespace}.json"
    try:
        cache = load_json(cache_path)
    except (OSError, ValueError):
        cache = {}
    local, errors, revised = (
        ({}, {}, {})
        if source == "remote"
        else _local_index_records(runs, cache, verify=verify)
    )
    if source != "remote" and revised != cache:
        write_json_atomic(cache_path, revised)
    remote = {}
    if source != "local" and catalog is not None:
        validate_catalog(catalog, complete=False)
        remote = {
            identity: entry["record"] for identity, entry in catalog["runs"].items()
        }
    protections = {}
    for identity in local:
        marker = runs / identity / ".protected"
        if marker.exists():
            protections[identity] = identity
    pending = defaultdict(set)
    if source != "remote" and runs.is_dir():
        _hidden_inputs(runs, {**remote, **local}, pending)
    return resolve_run_index(
        local,
        remote,
        source=source,
        reports=articles(repo),
        protections=protections,
        unfinished={
            identity: ",".join(sorted(details)) for identity, details in pending.items()
        },
        remote_known=remote_known,
        local_errors=errors,
        verified_local={
            identity for identity, row in revised.items() if row.get("verified")
        },
        publications=publications,
        operation=operation,
        inventory_state=inventory_state
        or (catalog.get("inventory") if catalog else None)
        or dict(known=False, unresolved={}, protected=[]),
    )


def render_status(index):
    lines = [
        f"Generation: {index['generation']}",
        f"Source: {index['source']}",
        f"Inventory generation: {index['inventory_generation'] or 'unresolved'}",
        f"Publication: {index['publication']}",
        "Run\tEvidence\tStorage\tVerification\tReasons / unresolved",
    ]
    for identity, row in index["runs"].items():
        reasons = row["reasons"] + row["unresolved"]
        verification = ", ".join(
            f"{key}: {value}" for key, value in row["verification"].items()
        )
        lines.append(
            f"{identity}\t{row['status']}\t{row['storage']}\t{verification}\t{'; '.join(reasons)}"
        )
    for experiment, error in index["selection_errors"].items():
        lines.append(f"{experiment}\tUnresolved selection: {error}")
    return "\n".join(lines)


# Shared maintenance controls


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


def records_for_backup(source, destination, selected_ids=None):
    records = {
        path.name: validate_operational_run_directory(path)
        for path in sorted(source.iterdir())
        if not path.name.startswith(".")
        and (selected_ids is None or path.name in selected_ids)
    }
    missing_inputs = {
        reference["run_id"]
        for record in records.values()
        for reference in record["inputs"].values()
    } - records.keys()
    if not missing_inputs:
        for identity in records:
            _runset(records, identity)
        return records

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


def backup_plan(source, destination, selected_ids=None):
    records = records_for_backup(source, destination, selected_ids)
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
        records = records_for_backup(
            source, destination, {row["run_id"] for row in plan["runs"]}
        )
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
    try:
        return coordinated_backup(source, destination, dry_run=dry_run, confirm=confirm)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"pingstore backup: {exc}", file=sys.stderr)
        return 1


# Checksummed public catalogs

SCHEMA = "pingstore.catalog/v1"


def catalog_path(destination):
    return destination.rsplit("/", 1)[0] + "/catalog.json"


def read_catalog(destination, *, complete=True, allow_malformed=False):

    # Missing is allowed only when listing confirms this is the first catalog.
    parent = destination.rsplit("/", 1)[0]
    names = json.loads(rclone("lsjson", parent, "--files-only", capture=True))
    if not any(row["Path"] == "catalog.json" for row in names):
        return {"schema": SCHEMA, "runs": {}}
    value = json.loads(rclone("cat", catalog_path(destination), capture=True))
    if allow_malformed:
        if value.get("schema") != SCHEMA or not isinstance(value.get("runs"), dict):
            raise PingstoreError("invalid R2 catalog envelope")
    else:
        validate_catalog(value, complete=complete)
    return value


def _check_catalog_entries(*, value, complete=True):
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
        for reference in record["inputs"].values() if complete else ():
            parent = value["runs"].get(reference["run_id"])
            if (
                parent is None
                or parent["record"]["payload_digest"] != reference["payload_digest"]
            ):
                raise PingstoreError(f"catalog missing or changed ancestor: {identity}")


def validate_catalog(value, *, complete=True):
    if value.get("schema") != SCHEMA or not isinstance(value.get("runs"), dict):
        raise PingstoreError("invalid R2 catalog")
    _check_catalog_entries(value=value, complete=complete)
    if not complete:
        return

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
    if read_catalog(destination, complete=False) != previous:
        raise PingstoreError("catalog changed concurrently; retry backup")
    conditional_document(
        catalog_path(destination), previous, value, exists="updated_at" in previous
    )


def inventory_projection(objects, entries):
    expected = {
        f"{identity}/{name}": row["bytes"]
        for identity, entry in entries.items()
        for name, row in entry["files"].items()
    }
    protected = sorted(
        {name.split("/")[0] for name in objects if name.endswith("/.protected")}
    )
    unresolved = {}
    for name in objects.keys() | expected.keys():
        if name.endswith("/.protected"):
            continue
        if objects.get(name) != expected.get(name):
            unresolved[name.split("/")[0]] = (
                "authenticated object inventory differs from the catalog"
            )
    return dict(
        schema="pingstore.inventory/v1",
        known=True,
        generation=_plan_hash(objects),
        protected=protected,
        present=sorted({name.split("/")[0] for name in objects}),
        unresolved=unresolved,
    )


def _catalog_watermarks(source, previous, entries):
    counters = dict(previous.get("allocation_watermarks", {}))
    for experiment, number in allocation_watermarks(source.parent).items():
        counters[experiment] = max(counters.get(experiment, 0), number)
    for identity in entries:
        match = STAGE_ID_RE.fullmatch(identity)
        counters[match.group(1)] = max(
            counters.get(match.group(1), 0), int(match.group(2))
        )
    return counters


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
            if identity in plan.get("tombstones", {}):
                raise PingstoreError(
                    f"uncataloged objects would resurrect pruned evidence: {identity}"
                )
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
        counters = _catalog_watermarks(source, previous, entries)
        value = {
            "schema": SCHEMA,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "runs": entries,
            "allocation_watermarks": counters,
            "inventory": inventory_projection(remote, entries),
        }
        value["report_index"] = resolve_run_index(
            {},
            {identity: entry["record"] for identity, entry in entries.items()},
            source="remote",
            reports=articles(source.parent.parent),
            publications=plan.get("publications"),
            inventory_state=value["inventory"],
        )
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


# Coordinated remote maintenance. A lease never expires automatically: a stopped
# writer must resume its original journal, rather than racing a newer operation.

CONTROL_SCHEMA = "pingstore.control/v1"
UNIFIED_PLAN_SCHEMA = "pingstore.prune-plan/v6"
PUBLICATION_SCHEMA = "pingstore.publications/v1"


def empty_publications():
    return dict(schema=PUBLICATION_SCHEMA, known=False, active=None, incoming={})


def control_path(destination):
    return destination.rsplit("/", 1)[0] + "/control.json"


def read_control(destination):
    parent = destination.rsplit("/", 1)[0]
    names = json.loads(rclone("lsjson", parent, "--files-only", capture=True))
    if not any(row["Path"] == "control.json" for row in names):
        return dict(
            schema=CONTROL_SCHEMA,
            revision=0,
            operation=None,
            publications=empty_publications(),
            tombstones={},
        )
    value = json.loads(rclone("cat", control_path(destination), capture=True))
    if (
        value.get("schema") != CONTROL_SCHEMA
        or type(value.get("revision")) is not int
        or value["revision"] < 0
        or not isinstance(value.get("tombstones"), dict)
    ):
        raise PingstoreError("invalid remote coordination document")
    operation = value.get("operation")
    if operation is not None and (
        not isinstance(operation, dict)
        or operation.get("kind") not in {"backup", "prune"}
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", str(operation.get("plan_hash")))
        or not re.fullmatch(r"[0-9a-f]{64}", str(operation.get("token")))
    ):
        raise PingstoreError("invalid remote operation fence")
    validate_publications(value.get("publications"))
    return value


def _s3_client(remote):
    # Keep credentials in memory; never print the decrypted rclone configuration.
    import boto3

    config = json.loads(rclone("config", "dump", capture=True)).get(remote)
    if not config or config.get("type") != "s3":
        raise PingstoreError("conditional maintenance requires an S3/R2 rclone remote")
    endpoint = config.get("endpoint")
    if not endpoint or not endpoint.startswith("https://"):
        raise PingstoreError("conditional maintenance requires an HTTPS S3 endpoint")
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name=config.get("region") or "auto",
        aws_access_key_id=config.get("access_key_id") or None,
        aws_secret_access_key=config.get("secret_access_key") or None,
    )


def conditional_document(path, previous, value, *, exists):
    """Use the S3 SDK's atomic PutObject conditions, not rclone upload headers."""
    remote, object_path = path.split(":", 1)
    bucket, key = object_path.split("/", 1)
    client = _s3_client(remote)
    condition = {"IfNoneMatch": "*"}
    if exists:
        response = client.get_object(Bucket=bucket, Key=key)
        before = response["Body"].read()
        if json.loads(before) != previous:
            raise PingstoreError(f"remote document changed concurrently: {path}")
        condition = {"IfMatch": response["ETag"]}
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    try:
        client.put_object(
            Bucket=bucket,
            Key=key,
            Body=payload,
            ContentType="application/json",
            CacheControl="no-cache",
            **condition,
        )
    except Exception as exc:
        # SDK exceptions may contain signed URLs or configuration. Report only
        # the service's error code, leaving ambiguous outcomes for journal recovery.
        code = (
            getattr(exc, "response", {})
            .get("Error", {})
            .get("Code", type(exc).__name__)
        )
        raise PingstoreError(
            f"conditional remote publication failed ({code}): {path}"
        ) from None
    if rclone("cat", path, capture=True) != payload:
        raise PingstoreError(f"remote publication verification failed: {path}")


def replace_control(destination, previous, value):
    if read_control(destination) != previous:
        raise PingstoreError("remote coordination state changed concurrently")
    conditional_document(
        control_path(destination), previous, value, exists=previous["revision"] > 0
    )


def acquire_remote_operation(destination, previous, operation):
    if previous["operation"] is not None:
        raise PingstoreError(f"unfinished remote operation: {previous['operation']}")
    revised = dict(previous, revision=previous["revision"] + 1, operation=operation)
    replace_control(destination, previous, revised)
    return revised


def assert_remote_operation(destination, operation):
    current = read_control(destination)
    if current["operation"] != operation:
        raise PingstoreError("remote maintenance lease changed; deletion stopped")
    return current


def release_remote_operation(
    destination, operation, *, tombstones=None, publications=None
):
    current = assert_remote_operation(destination, operation)
    revised = dict(current, operation=None, revision=current["revision"] + 1)
    if tombstones is not None:
        revised["tombstones"] = tombstones
    if publications is not None:
        validate_publications(publications)
        revised["publications"] = publications
    replace_control(destination, current, revised)
    return revised


def validate_evidence_manifest(value, records=None):
    if (
        not isinstance(value, dict)
        or value.get("schema") != "pingstore.evidence/v1"
        or not isinstance(value.get("build"), dict)
        or not isinstance(value["build"].get("id"), str)
        or not re.fullmatch(r"[0-9a-f]{40}", str(value["build"].get("revision", "")))
        or not isinstance(value.get("runs"), dict)
        or not isinstance(value.get("selections"), dict)
    ):
        raise PingstoreError("invalid deployment evidence manifest")
    graph = {}
    for identity, entry in value["runs"].items():
        if not STAGE_ID_RE.fullmatch(identity) or not isinstance(entry, dict):
            raise PingstoreError("invalid deployed run identity")
        _validate_input_references(value=entry, run_id=identity)
        if not re.fullmatch(
            r"sha256:[a-f0-9]{64}", str(entry.get("payload_digest", ""))
        ):
            raise PingstoreError("invalid deployed payload digest")
        if records is not None:
            record = records.get(identity)
            if record is None or any(
                record[key] != entry[key] for key in ("payload_digest", "inputs")
            ):
                raise PingstoreError(
                    f"deployed evidence is missing or changed: {identity}"
                )
        graph[identity] = entry
    for identity in graph:
        _runset(graph, identity)
    for experiment, identity in value["selections"].items():
        if (
            not EXPERIMENT_RE.fullmatch(experiment)
            or identity not in graph
            or not identity.startswith(experiment + "-")
            or not identity.endswith("-present")
        ):
            raise PingstoreError("invalid deployed presentation selection")
    return value


def validate_publications(value):
    if (
        not isinstance(value, dict)
        or value.get("schema") != PUBLICATION_SCHEMA
        or type(value.get("known")) is not bool
        or not isinstance(value.get("incoming"), dict)
    ):
        raise PingstoreError("invalid publication state")
    manifests = list(value["incoming"].values())
    if value.get("active") is not None:
        manifests.append(value["active"])
    for manifest in manifests:
        validate_evidence_manifest(manifest)
    return value


def _publication_reasons(publications, records, reasons, *, complete=True):
    validate_publications(publications)
    manifests = (
        [("deployment", publications["active"])] if publications.get("active") else []
    )
    manifests += [
        ("incoming", manifest) for manifest in publications["incoming"].values()
    ]
    for label, manifest in manifests:
        validate_evidence_manifest(manifest, records if complete else None)
        for identity in manifest["runs"]:
            if identity not in records:
                continue
            if any(
                records[identity][key] != manifest["runs"][identity][key]
                for key in ("payload_digest", "inputs")
            ):
                raise PingstoreError(f"published evidence differs: {identity}")
            reasons[identity].add(f"{label}:{manifest['build']['id']}")
    if not publications["known"]:
        for identity in records:
            reasons[identity].add("unresolved-publication")


def _maintenance_files(directory):
    """Inventory only a bounded, real run directory; never follow a link."""
    if directory.is_symlink() or not directory.is_dir():
        raise PingstoreError(f"unsafe run directory: {directory}")
    rows = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise PingstoreError(f"unsupported run object: {path}")
        relative = path.relative_to(directory).as_posix()
        if not _safe_object_path(relative):
            raise PingstoreError(f"unsafe run object: {path}")
        if path.is_file():
            rows[relative] = dict(bytes=path.stat().st_size, sha256=file_sha256(path))
    return rows


def _safe_object_path(name):
    return (
        isinstance(name, str)
        and bool(name)
        and not any(c in name for c in "\\\r\n\0")
        and all(part not in {"", ".", ".."} for part in name.split("/"))
    )


def maintenance_local_snapshot(repo):
    runs = repo / ".pingstore/runs"
    records, objects, malformed = {}, {}, {}
    if not runs.exists():
        return dict(
            records=records,
            objects=objects,
            malformed=malformed,
            hidden=[],
            watermarks=allocation_watermarks(repo / ".pingstore"),
        )
    if runs.is_symlink() or not runs.is_dir():
        raise PingstoreError(f"unsafe runs root: {runs}")
    for directory in sorted(runs.iterdir()):
        if directory.name.startswith("."):
            continue
        if not STAGE_ID_RE.fullmatch(directory.name):
            raise PingstoreError(f"unbounded malformed identity: {directory.name}")
        marker = directory / ".protected"
        if marker.is_symlink() or (marker.exists() and not marker.is_file()):
            raise PingstoreError(f"unsafe protection marker: {marker}")
        objects[directory.name] = _maintenance_files(directory)
        try:
            records[directory.name] = validate_operational_run_directory(directory)
        except (OSError, ValueError) as exc:
            malformed[directory.name] = str(exc)
    return dict(
        records=records,
        objects=objects,
        malformed=malformed,
        hidden=_maintenance_hidden_snapshot(runs),
        watermarks=allocation_watermarks(repo / ".pingstore"),
    )


def _maintenance_hidden_snapshot(runs):
    """Read unfinished-stage controls without hashing completed payloads."""
    if runs.is_symlink() or not runs.is_dir():
        raise PingstoreError(f"unsafe runs root: {runs}")
    hidden = []
    for path in sorted(runs.iterdir()):
        if path.name.startswith(".") and path.is_dir():
            if path.is_symlink():
                raise PingstoreError(f"unsafe unfinished directory: {path}")
            state = dict(name=path.name, active=_live_writer(path))
            for name in ("run.json", ".reservation.json"):
                file = path / name
                if file.exists():
                    if file.is_symlink() or not file.is_file():
                        raise PingstoreError(f"unsafe unfinished document: {file}")
                    state[name] = dict(sha256=file_sha256(file), value=load_json(file))
            hidden.append(state)
    return hidden


def _maintenance_remote_snapshot(destination, repo):
    # Validate individual records/inventories, then retained ancestry separately.
    catalog = read_catalog(destination, complete=False, allow_malformed=True)
    valid, malformed = {}, {}
    for identity, entry in catalog["runs"].items():
        if not STAGE_ID_RE.fullmatch(identity):
            raise PingstoreError(f"unbounded catalog identity: {identity}")
        try:
            _check_catalog_entries(value={"runs": {identity: entry}}, complete=False)
            valid[identity] = entry
        except (KeyError, TypeError, ValueError) as exc:
            malformed[identity] = str(exc)
    objects = inventory(destination)
    for name, size in objects.items():
        if not _safe_object_path(name) or type(size) is not int or size < 0:
            raise PingstoreError(f"unsafe remote object: {name}")
        if not STAGE_ID_RE.fullmatch(name.split("/", 1)[0]) or "/" not in name:
            raise PingstoreError(f"unbounded remote object identity: {name}")
    control = read_control(destination)
    deployment = (
        _verify_active_publication(repo, control["publications"])
        if control["publications"]["known"]
        else None
    )
    return dict(
        catalog=catalog,
        objects=objects,
        control=control,
        deployment=deployment,
        valid=valid,
        malformed=malformed,
    )


def _retained_graph(records, roots, reasons, *, strict):
    """Unknown out-of-scope ancestry retains candidates with its exact uncertainty."""
    keep = set()
    for identity in sorted(roots):
        try:
            members = _runset(records, identity)
        except PingstoreError as exc:
            if identity in strict:
                raise PingstoreError(f"retained evidence {identity}: {exc}") from exc
            keep.add(identity)
            for candidate in records:
                reasons[candidate].add(f"unresolved-ancestry:{identity}:{exc}")
            keep.update(records)
            continue
        keep.update(members)
        for member in members:
            if member != identity:
                reasons[member].update(
                    f"required-by:{identity}:{reason}" for reason in reasons[identity]
                )
    return keep


def _maintenance_hidden_roots(local, records, reasons, strict):
    for state in local["hidden"]:
        record = state.get("run.json", {}).get("value", {})
        for reference in record.get("inputs", {}).values():
            identity = reference["run_id"]
            if records.get(identity, {}).get("payload_digest") != reference.get(
                "payload_digest"
            ):
                raise PingstoreError(
                    f"unfinished {state['name']}: missing or changed input {identity}"
                )
            reasons[identity].add(f"unfinished:{state['name']}")
            strict.add(identity)


def _retain_other_store_inputs(storage, local, remote, records, reasons, strict):
    if storage == "both":
        return
    outside = local if storage == "r2" else remote
    for identity in outside:
        try:
            members = _runset(records, identity)
        except PingstoreError as exc:
            for candidate in records:
                reasons[candidate].add(
                    f"unresolved-other-storage-ancestry:{identity}:{exc}"
                )
            continue
        for member in set(members) - outside.keys():
            reasons[member].add(f"required-by:{identity}:other-storage-copy")
            strict.add(member)


def _maintenance_roots(
    repo, local, remote, records, scope, stages, publications, storage
):
    reasons = defaultdict(set)
    strict = set()
    selections = {}
    declarations = {
        article: metadata["inputs"] for article, metadata in articles(repo).items()
    }
    declared = set(declarations) | {
        experiment for inputs in declarations.values() for experiment in inputs
    }
    # An explicitly scoped, removed experiment has no current-report root.
    # Shared inputs, protections, unfinished work and publication still retain it.
    retired = {
        experiment
        for experiment in scope or ()
        if experiment not in declared
        and not (repo / "experiments" / experiment).exists()
        and not (repo / "experiments" / f"{experiment}.py").exists()
    }
    contexts = [("auto", local["records"], remote), ("remote", {}, remote)]
    for label, local_records, remote_records in contexts:
        index = resolve_run_index(local_records, remote_records, source=label)
        for experiment, error in index["selection_errors"].items():
            if experiment in retired:
                continue
            if scope is None or experiment in scope:
                raise PingstoreError(
                    f"unresolved {label} current runset {experiment}: {error}"
                )
            for identity in records:
                reasons[identity].add(
                    f"unresolved-current:{label}:{experiment}:{error}"
                )
        selections[label] = index["selections"]
        for experiment, identity in index["selections"].items():
            if experiment in retired:
                continue
            reasons[identity].add(f"current:{label}:{experiment}")
            strict.add(identity)
    for identity, record in records.items():
        if scope is not None and record["experiment"] not in scope:
            reasons[identity].add("out-of-scope")
        if stages is not None and record["stage"] not in stages:
            reasons[identity].add("out-of-stage-scope")
    for identity, files in local["objects"].items():
        if ".protected" in files:
            reasons[identity].add(f"protection:{identity}")
            strict.add(identity)
    _maintenance_hidden_roots(local, records, reasons, strict)
    _retain_other_store_inputs(
        storage, local["records"], remote, records, reasons, strict
    )
    _publication_reasons(publications, records, reasons)
    for identity, labels in reasons.items():
        if any(label.startswith(("deployment:", "incoming:")) for label in labels):
            strict.add(identity)
    # Report declarations are part of the plan hash; reporting cannot select pins.
    keep = _retained_graph(records, set(reasons), reasons, strict=strict)
    return reasons, keep, selections, declarations, retired


def _prune_objects(
    *, storage, local, remote, remote_state, records, reasons, keep, opaque_remote
):
    if remote_state:
        for name in remote_state["objects"]:
            identity = name.split("/", 1)[0]
            if name.endswith("/.protected"):
                reasons[identity].add(f"remote-protection:{identity}")
                members = _runset(records, identity)
                keep.update(members)
                for member in members:
                    reasons[member].add(f"remote-protection:{identity}")
    objects = dict(local={}, r2={})
    if storage != "r2":
        for identity, files in local["objects"].items():
            if identity not in keep:
                objects["local"].update(
                    {f"{identity}/{name}": row for name, row in files.items()}
                )
    if storage != "local":
        for name, size in remote_state["objects"].items():
            identity = name.split("/", 1)[0]
            entry = remote.get(identity)
            if identity in opaque_remote:
                if identity not in keep:
                    objects["r2"][name] = dict(bytes=size, sha256=None)
                continue
            expected = entry["files"].get(name.split("/", 1)[1])
            if not name.endswith("/.protected") and (
                expected is None or expected["bytes"] != size
            ):
                raise PingstoreError(f"remote object differs from catalog: {name}")
            if identity not in keep:
                objects["r2"][name] = dict(bytes=size, sha256=expected["sha256"])
        expected_paths = {
            f"{identity}/{name}"
            for identity, entry in remote.items()
            for name in entry["files"]
        }
        missing = expected_paths - remote_state["objects"].keys()
        if missing:
            raise PingstoreError(f"catalog objects unavailable: {sorted(missing)}")
    # Remote protections also retain a local copy's ancestry in a both-store plan.
    objects["local"] = {
        name: row
        for name, row in objects["local"].items()
        if name.split("/")[0] not in keep
    }
    return objects


def _opaque_holdings(remote_state, remote, local, malformed_runs):
    opaque_remote = dict(remote_state.get("malformed", {})) if remote_state else {}
    if remote_state:
        for name in remote_state["objects"]:
            identity = name.split("/")[0]
            if identity not in remote:
                opaque_remote.setdefault(identity, "uncataloged remote holding")
    malformed_runs = set(malformed_runs or ())
    if any(not STAGE_ID_RE.fullmatch(identity) for identity in malformed_runs):
        raise PingstoreError("malformed cleanup requires bounded v4 run identities")
    unknown = malformed_runs - (local["malformed"].keys() | opaque_remote.keys())
    if unknown:
        raise PingstoreError(
            f"explicit malformed selection is not malformed: {sorted(unknown)}"
        )
    return opaque_remote, malformed_runs


def _retain_opaque(
    *,
    opaque_remote,
    local,
    malformed_runs,
    publications,
    keep,
    reasons,
    records,
    scope,
    stage_scope,
):
    opaque = {**opaque_remote, **local["malformed"]}
    for identity, error in opaque.items():
        if identity not in malformed_runs or not publications["known"]:
            keep.add(identity)
            reasons[identity].add(f"unresolved-holding:{error}")
            for member in records:
                reasons[member].add(f"unresolved-ancestry:{identity}")
                keep.add(member)
        elif (scope is not None and identity.split("-")[0] not in scope) or (
            stage_scope is not None
            and STAGE_ID_RE.fullmatch(identity).group(3) not in stage_scope
        ):
            raise PingstoreError(
                "explicit malformed selection is outside the prune scope"
            )
    return opaque


def unified_prune_plan(
    repo: Path,
    experiments=None,
    stages=None,
    *,
    storage="both",
    destination="r2:pinglab/pingstore/runs",
    publications=None,
    remote_snapshot=None,
    malformed_runs=None,
):
    """One exact plan for both stores; a dry-run writes neither store nor journals."""
    if storage not in {"local", "r2", "both"}:
        raise PingstoreError("storage must be local, r2 or both")
    repo = repo.resolve()
    local = maintenance_local_snapshot(repo)
    remote_state = remote_snapshot
    if remote_state is None:
        remote_state = (
            _maintenance_remote_snapshot(destination, repo)
            if storage != "local"
            else None
        )
    remote = (
        remote_state.get("valid", remote_state["catalog"]["runs"])
        if remote_state
        else {}
    )
    opaque_remote, malformed_runs = _opaque_holdings(
        remote_state, remote, local, malformed_runs
    )
    remote_records = {identity: entry["record"] for identity, entry in remote.items()}
    records = {**remote_records, **local["records"]}
    for identity in local["records"].keys() & remote_records.keys():
        if local["records"][identity] != remote_records[identity]:
            raise PingstoreError(f"local and R2 records differ: {identity}")
    scope = _experiment_scope(
        experiments,
        {
            **records,
            **{
                identity: dict(experiment=identity.split("-")[0])
                for identity in local["malformed"]
            },
        },
    )
    stage_scope = set(stages) if stages is not None else None
    if stage_scope is not None and (
        not stage_scope or stage_scope - {"compute", "analyse", "present"}
    ):
        raise PingstoreError("invalid or empty stage filter")
    if publications is None:
        publications = (
            remote_state["control"]["publications"]
            if remote_state
            else empty_publications()
        )
    if remote_state and remote_state["control"]["operation"]:
        raise PingstoreError(
            f"unfinished remote operation: {remote_state['control']['operation']}"
        )
    reasons, keep, selections, declarations, retired = _maintenance_roots(
        repo, local, remote_records, records, scope, stage_scope, publications, storage
    )
    opaque = _retain_opaque(
        opaque_remote=opaque_remote,
        local=local,
        malformed_runs=malformed_runs,
        publications=publications,
        keep=keep,
        reasons=reasons,
        records=records,
        scope=scope,
        stage_scope=stage_scope,
    )
    objects = _prune_objects(
        storage=storage,
        local=local,
        remote=remote,
        remote_state=remote_state,
        records=records,
        reasons=reasons,
        keep=keep,
        opaque_remote=opaque_remote,
    )
    catalog_removals = (
        sorted(
            identity
            for identity in remote_state["catalog"]["runs"]
            if identity not in keep
        )
        if remote_state and storage != "local"
        else []
    )
    target_ids = {
        name.split("/")[0]
        for objects_by_target in objects.values()
        for name in objects_by_target
    } | set(catalog_removals)
    rows = []
    for identity in sorted(records.keys() | opaque.keys()):
        if identity not in keep and identity not in target_ids:
            keep.add(identity)
            reasons[identity].add("not-in-selected-storage")
        record = records.get(identity)
        in_scope = (scope is None or identity.split("-")[0] in scope) and (
            stage_scope is None or (record and record["stage"] in stage_scope)
        )
        if not in_scope:
            keep.add(identity)
        rows.append(
            dict(
                run_id=identity,
                payload_digest=record["payload_digest"] if record else None,
                reasons=sorted(reasons[identity]) if identity in keep else ["unused"],
            )
        )
    # Bind exact snapshots, including all objects outside the selected scope.
    plan = dict(
        schema=UNIFIED_PLAN_SCHEMA,
        root=str(repo),
        storage=storage,
        destination=destination,
        experiments=sorted(scope) if scope is not None else None,
        stages=sorted(stage_scope) if stage_scope is not None else None,
        local=local,
        remote=remote_state,
        malformed_runs=sorted(malformed_runs),
        catalog_removals=catalog_removals,
        publications=publications,
        report_inputs=declarations,
        retired_experiments=sorted(retired),
        selections=selections,
        keep=[row for row in rows if row["run_id"] in keep],
        prune=[row for row in rows if row["run_id"] not in keep],
        objects=objects,
    )
    return {**plan, "plan_hash": _plan_hash(plan)}


def _prune_journal_path(repo, plan_hash):
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", str(plan_hash)):
        raise PingstoreError(
            "--confirm requires the complete sha256 plan hash from --dry-run"
        )
    return repo / ".pingstore/operations" / f"prune-{plan_hash[7:]}.json"


def _prune_scope(repo, experiments, stages, storage, destination, malformed_runs):
    return dict(
        root=str(repo.resolve()),
        experiments=sorted(set(experiments)) if experiments is not None else None,
        stages=sorted(set(stages)) if stages is not None else None,
        storage=storage,
        destination=destination,
        malformed_runs=sorted(set(malformed_runs or ())),
    )


def _journal_save(path, journal):
    write_json_atomic(path, journal)


def _expected_pruned_catalog(plan, updated_at):
    previous = plan["remote"]["catalog"]
    removed = set(plan["catalog_removals"])
    revised = dict(
        previous,
        runs={
            identity: entry
            for identity, entry in previous["runs"].items()
            if identity not in removed
        },
    )
    revised["updated_at"] = updated_at
    surviving_objects = {
        name: size
        for name, size in plan["remote"]["objects"].items()
        if name not in plan["objects"]["r2"]
    }
    revised["inventory"] = inventory_projection(surviving_objects, revised["runs"])
    revised["allocation_watermarks"] = _plan_watermarks(plan)
    revised["report_index"] = resolve_run_index(
        {},
        {identity: entry["record"] for identity, entry in revised["runs"].items()},
        source="remote",
        inventory_state=revised["inventory"],
        publications=plan["publications"],
        reports={
            article: {"inputs": inputs}
            for article, inputs in plan["report_inputs"].items()
        },
    )
    validate_catalog(revised)
    return revised


def _plan_watermarks(plan):
    counters = dict(plan["local"]["watermarks"])
    remote = plan["remote"]
    if remote:
        for experiment, count in (
            remote["catalog"].get("allocation_watermarks", {}).items()
        ):
            counters[experiment] = max(counters.get(experiment, 0), count)
    identities = set(plan["local"]["objects"])
    if remote:
        identities.update(remote["catalog"]["runs"])
        identities.update(name.split("/")[0] for name in remote["objects"])
    for identity in identities:
        match = STAGE_ID_RE.fullmatch(identity)
        counters[match.group(1)] = max(
            counters.get(match.group(1), 0), int(match.group(2))
        )
    return counters


def _local_retention_controls(plan, *, hidden=None):
    """Check mutable retention controls without rescanning the archive bytes."""
    repo = Path(plan["root"])
    if hidden is None:
        runs = repo / ".pingstore/runs"
        hidden = _maintenance_hidden_snapshot(runs) if runs.exists() else []
    if hidden != plan["local"]["hidden"]:
        raise PingstoreError(
            "unfinished work changed; original cleanup cannot continue"
        )
    if any(state["active"] for state in hidden):
        raise PingstoreError("active writer prevents pruning")
    declarations = {
        article: metadata["inputs"] for article, metadata in articles(repo).items()
    }
    if declarations != plan["report_inputs"]:
        raise PingstoreError("report inputs changed during cleanup")
    if any(
        (repo / "experiments" / experiment).exists()
        or (repo / "experiments" / f"{experiment}.py").exists()
        for experiment in plan.get("retired_experiments", [])
    ):
        raise PingstoreError("retired experiment source restored during cleanup")


def _local_plan_controls(plan):
    repo = Path(plan["root"])
    current = maintenance_local_snapshot(repo)
    original = plan["local"]
    _local_retention_controls(plan, hidden=current["hidden"])
    removed = {name.split("/")[0] for name in plan["objects"]["local"]}
    expected = {
        identity: files
        for identity, files in original["objects"].items()
        if identity not in removed
    }
    actual = {
        identity: files
        for identity, files in current["objects"].items()
        if identity not in removed
    }
    if actual != expected or current["records"].keys() - original["records"].keys():
        raise PingstoreError("retained local inventory or run selections changed")
    for identity in removed & current["objects"].keys():
        if current["objects"][identity] != original["objects"][identity]:
            raise PingstoreError(f"local deletion candidate changed: {identity}")
    return current


def _remote_plan_controls(plan, operation, *, revised):
    destination = plan["destination"]
    control = assert_remote_operation(destination, operation)
    original = plan["remote"]
    for key in ("publications", "tombstones"):
        if control[key] != original["control"][key]:
            raise PingstoreError(f"remote {key} changed during cleanup")
    if original["control"]["publications"]["known"]:
        _verify_active_publication(Path(plan["root"]), control["publications"])
    catalog = read_catalog(destination, complete=False)
    if catalog not in (original["catalog"], revised):
        raise PingstoreError("catalog changed during cleanup")
    actual = inventory(destination)
    expected = original["objects"]
    deletions = plan["objects"]["r2"]
    if actual.keys() - expected.keys():
        raise PingstoreError("remote objects were added during cleanup")
    for name, size in expected.items():
        if name not in deletions and actual.get(name) != size:
            raise PingstoreError(f"retained remote object changed: {name}")
        if name in actual and actual[name] != size:
            raise PingstoreError(f"remote deletion candidate changed: {name}")
    return catalog, actual


def _delete_remote_plan(plan, journal, path):
    destination = plan["destination"]
    revised = journal["revised_catalog"]
    catalog, actual = _remote_plan_controls(plan, journal["operation"], revised=revised)
    _local_plan_controls(plan)
    if catalog != revised:
        replace_catalog(destination, catalog, revised)
    journal["phase"] = "catalog-published"
    _journal_save(path, journal)
    remaining = set(plan["objects"]["r2"]) & actual.keys()
    try:
        if remaining:
            with tempfile.TemporaryDirectory(prefix="pingstore-delete-") as temporary:
                names = file_list(Path(temporary), "objects.txt", remaining)
                assert_remote_operation(destination, journal["operation"])
                rclone("delete", destination, "--files-from-raw", names)
    finally:
        after = inventory(destination)
        journal["deleted"]["r2"] = sorted(set(plan["objects"]["r2"]) - after.keys())
        _journal_save(path, journal)
    if set(plan["objects"]["r2"]) & after.keys():
        raise PingstoreError(
            "partial R2 cleanup: planned objects remain; resume the original hash"
        )
    _remote_plan_controls(plan, journal["operation"], revised=revised)
    journal["phase"] = "r2-deleted"
    _journal_save(path, journal)


def _delete_local_plan(plan, journal, path):
    repo = Path(plan["root"])
    runs = repo / ".pingstore/runs"
    quarantine = repo / ".pingstore/quarantine" / plan["plan_hash"][7:]
    if quarantine.is_symlink() or quarantine.parent.is_symlink():
        raise PingstoreError("cleanup quarantine must not use symlinks")
    identities = sorted({name.split("/")[0] for name in plan["objects"]["local"]})
    deleted = set(journal["deleted"]["local"])
    for identity in identities:
        source, staged = runs / identity, quarantine / identity
        if identity in deleted and not source.exists() and not staged.exists():
            continue
        # The operation lock excludes cooperative writers. Recheck mutable
        # retention controls and this target, not every unrelated payload.
        _local_retention_controls(plan)
        expected = plan["local"]["objects"][identity]
        newly_staged = source.exists()
        if newly_staged:
            if staged.exists():
                raise PingstoreError(f"ambiguous local quarantine: {identity}")
            quarantine.mkdir(parents=True, exist_ok=True)
            os.rename(source, staged)
        if staged.exists():
            actual = _maintenance_files(staged)
            if (newly_staged and actual != expected) or any(
                expected.get(name) != row for name, row in actual.items()
            ):
                raise PingstoreError(f"quarantined candidate changed: {identity}")
            shutil.rmtree(staged)
        deleted.add(identity)
        journal["deleted"]["local"] = sorted(deleted)
        journal["phase"] = "local-deleting"
        _journal_save(path, journal)
    _local_plan_controls(plan)
    journal["phase"] = "local-deleted"
    _journal_save(path, journal)


def _complete_prune(plan, journal, path):
    if plan["storage"] != "local":
        _remote_plan_controls(
            plan, journal["operation"], revised=journal["revised_catalog"]
        )
        tombstones = dict(plan["remote"]["control"]["tombstones"])
        removed = set(plan["catalog_removals"]) | {
            name.split("/")[0] for name in plan["objects"]["r2"]
        }
        for identity in removed:
            entry = plan["remote"]["catalog"]["runs"].get(identity)
            record = entry.get("record") if isinstance(entry, dict) else None
            tombstones[identity] = dict(
                plan_hash=plan["plan_hash"],
                payload_digest=record.get("payload_digest")
                if isinstance(record, dict)
                else None,
            )
        release_remote_operation(
            plan["destination"], journal["operation"], tombstones=tombstones
        )
    journal["phase"] = "complete"
    journal["error"] = None
    _journal_save(path, journal)
    return journal


def _prune_already_released(plan, control):
    """Recover an acknowledged-or-not release without taking another writer's lease."""
    if plan["storage"] == "local" or control["operation"] is not None:
        return False
    removed = set(plan["catalog_removals"])
    if any(
        control["tombstones"].get(identity, {}).get("plan_hash") != plan["plan_hash"]
        for identity in removed
    ):
        return False
    actual = inventory(plan["destination"])
    if set(plan["objects"]["r2"]) & actual.keys():
        return False
    root = Path(plan["root"]) / ".pingstore"
    for identity in {name.split("/")[0] for name in plan["objects"]["local"]}:
        if (root / "runs" / identity).exists() or (
            root / "quarantine" / plan["plan_hash"][7:] / identity
        ).exists():
            return False
    validate_catalog(read_catalog(plan["destination"]))
    _local_plan_controls(plan)
    return True


def _validate_prune_journal(journal, plan):
    operation = journal.get("operation", {})
    if (
        operation.get("kind") != "prune"
        or operation.get("plan_hash") != plan["plan_hash"]
        or not re.fullmatch(r"[a-f0-9]{64}", str(operation.get("token", "")))
    ):
        raise PingstoreError("cleanup journal operation identity is invalid")
    if plan["storage"] != "local" and journal.get(
        "revised_catalog"
    ) != _expected_pruned_catalog(plan, journal["prepared_at"]):
        raise PingstoreError(
            "cleanup journal surviving catalog differs from the verified plan"
        )
    if journal.get("phase") == "complete":
        if plan["storage"] != "local":
            if not _prune_already_released(plan, read_control(plan["destination"])):
                raise PingstoreError(
                    "completed cleanup inventory changed; inspect a fresh plan"
                )
        else:
            root = Path(plan["root"]) / ".pingstore/runs"
            if any(
                (root / name.split("/")[0]).exists()
                for name in plan["objects"]["local"]
            ):
                raise PingstoreError(
                    "completed local cleanup copies have reappeared; inspect a fresh plan"
                )


def apply_unified_prune(
    repo: Path,
    expected_hash,
    experiments=None,
    stages=None,
    *,
    storage="both",
    destination="r2:pinglab/pingstore/runs",
    publications=None,
    remote_snapshot=None,
    malformed_runs=None,
):
    """Confirm or resume the original verified plan, retaining partial progress."""
    import secrets

    repo = repo.resolve()
    path = _prune_journal_path(repo, expected_hash)
    with operation_lock(repo / ".pingstore", exclusive=True):
        if path.exists():
            journal = load_json(path)
            plan = journal["plan"]
            if (
                _plan_hash(
                    {key: value for key, value in plan.items() if key != "plan_hash"}
                )
                != expected_hash
            ):
                raise PingstoreError("cleanup journal plan checksum mismatch")
            expected_scope = _prune_scope(
                repo, experiments, stages, storage, destination, malformed_runs
            )
            if any(plan[key] != value for key, value in expected_scope.items()):
                raise PingstoreError(
                    "confirmation scope differs from original cleanup journal"
                )
            _validate_prune_journal(journal, plan)
            if journal["phase"] == "complete":
                return journal
        else:
            plan = unified_prune_plan(
                repo,
                experiments,
                stages,
                storage=storage,
                destination=destination,
                publications=publications,
                remote_snapshot=remote_snapshot,
                malformed_runs=malformed_runs,
            )
            if plan["plan_hash"] != expected_hash:
                raise PingstoreError(
                    f"prune plan changed: expected {expected_hash}, now {plan['plan_hash']}"
                )
            journal = dict(
                schema="pingstore.operation/v1",
                plan=plan,
                phase="prepared",
                prepared_at=utc_now(),
                operation=dict(
                    kind="prune",
                    plan_hash=expected_hash,
                    token=secrets.token_hex(32),
                    run_ids=sorted(plan["catalog_removals"]),
                ),
                deleted=dict(local=[], r2=[]),
                error=None,
            )
            if not any(plan["objects"].values()) and not plan["catalog_removals"]:
                return dict(journal, phase="complete")
            if storage != "local":
                journal["revised_catalog"] = _expected_pruned_catalog(
                    plan, journal["prepared_at"]
                )
            _journal_save(path, journal)
        try:
            _local_plan_controls(plan)
            persist_watermarks(repo / ".pingstore", _plan_watermarks(plan))
            if storage != "local":
                current = read_control(destination)
                if _prune_already_released(plan, current):
                    journal["phase"] = "complete"
                    journal["error"] = None
                    _journal_save(path, journal)
                    return journal
                if current["operation"] != journal["operation"]:
                    if current != plan["remote"]["control"]:
                        raise PingstoreError(
                            "remote controls changed before cleanup lease acquisition"
                        )
                    acquire_remote_operation(destination, current, journal["operation"])
                _delete_remote_plan(plan, journal, path)
            _delete_local_plan(plan, journal, path)
            return _complete_prune(plan, journal, path)
        except BaseException as exc:
            journal["error"] = f"{type(exc).__name__}: {exc}"
            _journal_save(path, journal)
            raise PingstoreError(
                f"cleanup incomplete ({journal['phase']}); journal {path}; resume --confirm {expected_hash}: {exc}"
            ) from exc


def render_unified_prune_plan(plan):
    lines = [
        f"Plan: {plan['plan_hash']}",
        f"Storage: {plan['storage']}",
        f"Scope: {', '.join(plan['experiments'] or []) or 'all experiments'}",
        f"Stages: {', '.join(plan['stages'] or []) or 'all stages'}",
        "",
        "KEEP",
    ]
    if plan.get("retired_experiments"):
        lines.insert(
            4, f"Retired experiments: {', '.join(plan['retired_experiments'])}"
        )
    lines.extend(
        f"{row['run_id']}\t{'; '.join(row['reasons'])}" for row in plan["keep"]
    )
    lines.extend(["", "PRUNE"])
    lines.extend(row["run_id"] for row in plan["prune"])
    for target, objects in plan["objects"].items():
        size = sum(row["bytes"] for row in objects.values())
        identities = {name.split("/")[0] for name in objects}
        lines.append(
            f"{target}: {len(identities)} runs, {len(objects)} objects, {size} bytes"
        )
        lines.extend(
            f"{target}\t{name}\t{row['bytes']}\t{row['sha256']}"
            for name, row in sorted(objects.items())
        )
    scope = (
        " ".join(
            f"--experiment {experiment}" for experiment in plan["experiments"] or []
        )
        or "--all"
    )
    stages = "".join(f" --stage {stage}" for stage in plan["stages"] or [])
    stages += "".join(
        f" --malformed-run {identity}" for identity in plan["malformed_runs"]
    )
    command = (
        f"uv run pingstore prune --root {shlex.quote(plan['root'])} {scope}{stages} "
        f"--storage {plan['storage']} --destination {shlex.quote(plan['destination'])} --confirm {plan['plan_hash']}"
    )
    lines.extend(["", f"Confirm with: {command}"])
    return "\n".join(lines)


def evidence_manifest(index, build_id, revision):
    if index["selection_errors"]:
        raise PingstoreError(
            f"cannot publish unresolved selections: {index['selection_errors']}"
        )
    identities = sorted(
        {identity for members in index["runsets"].values() for identity in members}
    )
    value = dict(
        schema="pingstore.evidence/v1",
        build=dict(id=build_id, revision=revision),
        source=index["source"],
        index_generation=index["generation"],
        selections=index["selections"],
        reports=index["report_selections"],
        unavailable_reports={
            article: sorted(set(inputs) - index["selections"].keys())
            for article, inputs in index["report_inputs"].items()
            if set(inputs) - index["selections"].keys()
        },
        runs={
            identity: {
                key: index["runs"][identity]["record"][key]
                for key in ("payload_digest", "inputs")
            }
            for identity in identities
        },
    )
    return validate_evidence_manifest(value)


def _github_repository(repo):
    remote = subprocess.check_output(
        ["git", "-C", str(repo), "remote", "get-url", "origin"], text=True
    ).strip()
    match = re.fullmatch(
        r"(?:https://github.com/|git@github.com:)([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+?)(?:\.git)?",
        remote,
    )
    if not match:
        raise PingstoreError("publication requires a verifiable GitHub origin")
    return match.group(1)


def _github_json(repository, endpoint):
    try:
        data = subprocess.check_output(
            ["gh", "api", f"repos/{repository}/{endpoint}"], stderr=subprocess.PIPE
        )
        return json.loads(data)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        raise PingstoreError(
            f"cannot establish GitHub publication state ({type(exc).__name__})"
        ) from None


def _github_pages(repository, endpoint, key=None):
    page = 1
    while True:
        suffix = f"&page={page}" if page > 1 else ""
        response = _github_json(repository, endpoint + suffix)
        rows = response[key] if key else response
        yield from rows
        if len(rows) < 100:
            return
        page += 1


def github_pages_state(repo):
    repository = _github_repository(repo)
    deployments = _github_pages(
        repository, "deployments?environment=github-pages&per_page=100"
    )
    active = None
    for deployment in deployments:
        statuses = _github_json(
            repository, f"deployments/{deployment['id']}/statuses?per_page=1"
        )
        if statuses and statuses[0]["state"] == "success":
            status = statuses[0]
            match = re.search(r"/actions/runs/(\d+)", status.get("log_url", ""))
            if not match:
                raise PingstoreError(
                    "successful Pages deployment has no verifiable workflow identity"
                )
            active = dict(
                deployment_id=deployment["id"],
                run_id=match.group(1),
                revision=deployment["sha"],
                url=status["environment_url"],
            )
            break
    workflows = _github_pages(
        repository, "actions/workflows/pages.yml/runs?per_page=100", "workflow_runs"
    )
    pending = sorted(
        str(run["id"]) for run in workflows if run["status"] != "completed"
    )
    return dict(repository=repository, active=active, pending=pending)


def public_json(url):
    request = Request(
        url,
        headers={"User-Agent": "Pinglab-Pingstore/1.0", "Cache-Control": "no-cache"},
    )
    with urlopen(request, timeout=30) as response:
        return json.load(response)


def _verify_legacy_artifact(state, manifest):
    with tempfile.TemporaryDirectory(prefix="pingstore-legacy-proof-") as temporary:
        subprocess.run(
            [
                "gh",
                "run",
                "download",
                state["active"]["run_id"],
                "--repo",
                state["repository"],
                "--name",
                "github-pages",
                "--dir",
                temporary,
            ],
            check=True,
            capture_output=True,
        )
        if (
            file_sha256(Path(temporary) / "artifact.tar")
            != manifest["bootstrap"]["artifact_sha256"]
        ):
            raise PingstoreError(
                "active legacy artifact changed from the bootstrapped evidence"
            )


def _verify_active_publication(repo, publications):
    """Never infer successful publication from backup availability or latest recency."""
    state = github_pages_state(repo)
    actual = state["active"]
    active = publications.get("active")
    if actual is None:
        if active is not None:
            raise PingstoreError(
                "recorded publication has no successful Pages deployment"
            )
        return state
    candidates = ([active] if active else []) + list(publications["incoming"].values())
    matching = [
        manifest
        for manifest in candidates
        if manifest["build"]["revision"] == actual["revision"]
        and manifest["build"]["id"].split("/")[0] == actual["run_id"]
    ]
    if not matching:
        raise PingstoreError(
            f"active deployment {actual['deployment_id']} is not reconciled with Pingstore publication evidence"
        )
    active = matching[0]
    if "bootstrap" in active:
        _verify_legacy_artifact(state, active)
    else:
        deployed = public_json(actual["url"].rstrip("/") + "/pingstore-evidence.json")
        if deployed not in matching:
            raise PingstoreError(
                "deployed manifest differs from recorded protected evidence"
            )
    registered = {key.split("/")[0] for key in publications["incoming"]}
    unmatched = set(state["pending"]) - registered
    if unmatched:
        raise PingstoreError(
            f"unresolved incoming Pages workflows: {sorted(unmatched)}"
        )
    return state


def update_publications(destination, previous, revised):
    """Publication metadata races on the same CAS fence as every storage writer."""
    validate_publications(revised)
    control = read_control(destination)
    if control["operation"] is not None:
        raise PingstoreError(
            "storage maintenance is active; publication registration must retry"
        )
    if control["publications"] != previous:
        raise PingstoreError("publication state changed concurrently")
    records = {
        identity: entry["record"]
        for identity, entry in read_catalog(destination)["runs"].items()
    }
    reasons = defaultdict(set)
    _publication_reasons(revised, records, reasons)
    replace_control(
        destination,
        control,
        dict(control, revision=control["revision"] + 1, publications=revised),
    )


def publication_prepare(repo: Path, destination, output: Path, *, build_id, revision):
    """Freeze a remote-only build and retain its exact inputs before rendering."""
    catalog = read_catalog(destination)
    public = public_json(
        os.environ.get(
            "PINGLAB_ASSET_BASE_URL", "https://assets.eoinmurray.info"
        ).rstrip("/")
        + "/pingstore/catalog.json"
    )
    if public != catalog:
        raise PingstoreError(
            "public and authenticated catalogs disagree before publication"
        )
    control = read_control(destination)
    index = reporting_index(
        repo, source="remote", catalog=catalog, publications=control["publications"]
    )
    manifest = evidence_manifest(index, build_id, revision)
    output.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output / "catalog.json", catalog)
    write_json_atomic(output / "index.json", index)
    write_json_atomic(output / "evidence.json", manifest)
    publication_retain(destination, manifest)
    return manifest


def publication_retain(destination, manifest):
    """Retain frozen evidence again before a deployment, including job retries."""
    catalog = read_catalog(destination)
    records = {identity: entry["record"] for identity, entry in catalog["runs"].items()}
    validate_evidence_manifest(manifest, records)
    previous = read_control(destination)["publications"]
    revised = dict(
        previous,
        incoming={**previous["incoming"], manifest["build"]["id"]: manifest},
    )
    update_publications(destination, previous, revised)
    return manifest


def publication_activate(repo: Path, destination, manifest):
    """Release old deployment roots only after GitHub and the deployed bytes agree."""
    validate_evidence_manifest(manifest)
    state = github_pages_state(repo)
    active = state["active"]
    if (
        active is None
        or active["revision"] != manifest["build"]["revision"]
        or active["run_id"] != manifest["build"]["id"].split("/")[0]
    ):
        raise PingstoreError(
            "the replacement is not the active successful Pages deployment"
        )
    deployed = public_json(active["url"].rstrip("/") + "/pingstore-evidence.json")
    if deployed != manifest:
        raise PingstoreError("the replacement evidence manifest is not deployed")
    previous = read_control(destination)["publications"]
    # A verified replacement also supersedes completed failed attempts. Pending
    # workflows keep their roots; deploy-only retries register them again first.
    pending = set(state["pending"]) - {active["run_id"]}
    incoming = {
        identity: evidence
        for identity, evidence in previous["incoming"].items()
        if identity.split("/")[0] in pending
    }
    revised = dict(previous, known=True, active=manifest, incoming=incoming)
    update_publications(destination, previous, revised)
    return revised


def publication_failed(repo: Path, destination, build_id):
    state = github_pages_state(repo)
    if state["active"] and state["active"]["run_id"] == build_id.split("/")[0]:
        raise PingstoreError(
            "this build is deployed; reconcile activation instead of releasing its evidence"
        )
    previous = read_control(destination)["publications"]
    incoming = dict(previous["incoming"])
    incoming.pop(build_id, None)
    revised = dict(previous, incoming=incoming)
    update_publications(destination, previous, revised)
    return revised


def bootstrap_publication(repo: Path, destination, output: Path):
    """Recover conservative legacy roots from the active authenticated Pages artifact."""
    import tarfile

    state = github_pages_state(repo)
    if state["pending"]:
        raise PingstoreError(
            f"legacy publication is unresolved while workflows run: {state['pending']}"
        )
    if state["active"] is None:
        previous = read_control(destination)["publications"]
        revised = dict(previous, known=True, active=None)
        update_publications(destination, previous, revised)
        return revised
    active = state["active"]
    catalog = read_catalog(destination)
    records = {identity: entry["record"] for identity, entry in catalog["runs"].items()}
    identities = set()
    with tempfile.TemporaryDirectory(prefix="pingstore-pages-bootstrap-") as temporary:
        subprocess.run(
            [
                "gh",
                "run",
                "download",
                active["run_id"],
                "--repo",
                state["repository"],
                "--name",
                "github-pages",
                "--dir",
                temporary,
            ],
            check=True,
            capture_output=True,
        )
        archive = Path(temporary) / "artifact.tar"
        artifact_digest = file_sha256(archive)
        with tarfile.open(archive) as artifact:
            for member in artifact:
                if not member.isfile() or not member.name.endswith(".html"):
                    continue
                if member.size > 32 * 1024**2:
                    raise PingstoreError(
                        "legacy HTML exceeds the bounded bootstrap reader"
                    )
                data = artifact.extractfile(member).read().decode("utf8")
                # This authored text archive is explicitly separate from current
                # exp124 evidence. Its historical Dataset rows are not live roots.
                if member.name.removeprefix("./") == "recovered/exp124/published.html":
                    data = re.sub(
                        r'<tr[^>]*class="[^"]*selected-run[^"]*"[^>]*>.*?</tr>',
                        "",
                        data,
                        flags=re.S,
                    )
                for row in re.findall(
                    r'<tr[^>]*class="[^"]*selected-run[^"]*"[^>]*>(.*?)</tr>',
                    data,
                    re.S,
                ):
                    identities.update(
                        match.group(0)
                        for match in re.finditer(
                            STAGE_ID_RE.pattern.removeprefix("^").removesuffix("$"), row
                        )
                    )
                identities.update(
                    re.findall(
                        r"/pingstore/runs/(exp\d{3}-r\d+-(?:compute(?:-phase-[a-z0-9-]+)?|analyse|present))/",
                        data,
                    )
                )
    # STAGE_ID_RE has capture groups; identity extraction must use full matches.
    if any(not isinstance(identity, str) for identity in identities):
        raise PingstoreError("legacy artifact run identity extraction failed")
    missing = identities - records.keys()
    if missing:
        raise PingstoreError(
            f"active legacy deployment references unavailable evidence: {sorted(missing)}"
        )
    kept = set()
    for identity in identities:
        kept.update(_runset(records, identity))
    manifest = dict(
        schema="pingstore.evidence/v1",
        build=dict(id=active["run_id"] + "/legacy", revision=active["revision"]),
        selections={},
        runs={
            identity: {
                key: records[identity][key] for key in ("payload_digest", "inputs")
            }
            for identity in sorted(kept)
        },
        bootstrap=dict(
            deployment_id=active["deployment_id"],
            source="authenticated-pages-artifact",
            artifact_sha256=artifact_digest,
            excluded_text_inventory=["recovered/exp124/published.html"],
        ),
    )
    validate_evidence_manifest(manifest, records)
    write_json_atomic(output, manifest)
    previous = read_control(destination)["publications"]
    revised = dict(previous, known=True, active=manifest)
    update_publications(destination, previous, revised)
    return revised


def _backup_selection(source, destination, control):
    local = {
        path.name: validate_run(load_json(path / "run.json"))
        for path in sorted(source.iterdir())
        if not path.name.startswith(".")
    }
    remote = {
        identity: entry["record"]
        for identity, entry in read_catalog(destination, complete=False)["runs"].items()
    }
    records = {**remote, **local}
    index = resolve_run_index(local, remote)
    if index["selection_errors"]:
        raise PingstoreError(
            f"unresolved current backup inputs: {index['selection_errors']}"
        )
    required = {
        identity for members in index["runsets"].values() for identity in members
    }
    reasons = defaultdict(set)
    if source.is_dir():
        _hidden_inputs(source, records, reasons)
    required.update(reasons)
    required.update(
        identity for identity in local if (source / identity / ".protected").exists()
    )
    selected, excluded = set(), {}
    tombstones = control["tombstones"]
    for identity in sorted(local):
        if identity in tombstones:
            if identity in required:
                raise PingstoreError(
                    f"current or protected evidence was pruned: {identity}; regenerate or explicitly recover its evidence"
                )
            excluded[identity] = "pruned history; resurrection refused"
            continue
        try:
            members = _runset(records, identity)
        except PingstoreError as exc:
            if identity in required:
                raise
            excluded[identity] = f"unused broken history: {exc}"
            continue
        deleted_inputs = set(members) & tombstones.keys()
        if deleted_inputs:
            raise PingstoreError(
                f"backup {identity} would resurrect pruned inputs: {sorted(deleted_inputs)}"
            )
        selected.update(set(members) & local.keys())
    return selected, excluded


def _coordinated_backup_plan(source, destination):
    control = read_control(destination)
    if control["operation"] is not None:
        raise PingstoreError(f"unfinished remote operation: {control['operation']}")
    selected, excluded = _backup_selection(source, destination, control)
    plan = backup_plan(source, destination, selected)
    plan["excluded"] = excluded
    tombstoned = {row["run_id"] for row in plan["runs"]} & control["tombstones"].keys()
    if tombstoned:
        raise PingstoreError(
            f"backup would resurrect pruned evidence: {sorted(tombstoned)}"
        )
    plan.pop("plan_hash")
    plan["control"] = control
    return {**plan, "plan_hash": _plan_hash(plan)}


def _backup_work_plan(source, destination, original):
    selected = {row["run_id"] for row in original["runs"]}
    work = backup_plan(source, destination, selected)
    if work["runs"] != original["runs"]:
        raise PingstoreError("original backup inputs changed; resumption refused")
    allowed = dict(original["remote_inventory"])
    for row in original["runs"]:
        allowed.update(row["files"])
    actual = work["remote_inventory"]
    if any(allowed.get(name) != size for name, size in actual.items()):
        raise PingstoreError("unexpected remote objects during backup resumption")
    if any(
        actual.get(name) != size for name, size in original["remote_inventory"].items()
    ):
        raise PingstoreError("pre-existing remote objects changed during backup")
    return work


def _backup_already_released(plan, control):
    if control["operation"] is not None:
        return False
    catalog = read_catalog(plan["destination"])
    for row in plan["runs"]:
        entry = catalog["runs"].get(row["run_id"])
        if entry is None or entry["record"]["payload_digest"] != row["payload_digest"]:
            return False
        if (
            entry["files"]["run.json"]["sha256"] != row["run_json_sha256"]
            or entry["files"]["README.md"]["sha256"] != row["readme_sha256"]
        ):
            return False
        expected = {
            name.removeprefix(row["run_id"] + "/"): size
            for name, size in row["files"].items()
        }
        if {name: file["bytes"] for name, file in entry["files"].items()} != expected:
            return False
    return True


def coordinated_backup(source, destination, *, dry_run, confirm):
    import secrets

    source = source.resolve()
    store = source.parent
    with operation_lock(store, exclusive=True):
        if dry_run:
            plan = _coordinated_backup_plan(source, destination)
            print(f"Destination: {destination}")
            print(
                f"New runs: {len(plan['new_runs'])}; existing runs: {len(plan['existing_runs'])}"
            )
            for row in plan["runs"]:
                print(
                    f"{row['run_id']}\t{sum(row['files'].values())} bytes\t{row['payload_digest']}"
                )
            for identity, reason in plan["excluded"].items():
                print(f"Not uploaded: {identity} ({reason})")
            print(f"Plan: {plan['plan_hash']}")
            return 0
        _prune_journal_path(
            store.parent, confirm
        )  # Validate the full hash before any remote access.
        path = store / "operations" / f"backup-{confirm[7:]}.json"
        if path.exists():
            journal = load_json(path)
            plan = journal["plan"]
            if (
                plan["source"] != str(source)
                or plan["destination"] != destination
                or _plan_hash(
                    {key: value for key, value in plan.items() if key != "plan_hash"}
                )
                != confirm
            ):
                raise PingstoreError("backup journal scope or checksum mismatch")
            if journal["phase"] == "complete":
                print("Backup already completed and verified.")
                return 0
        else:
            plan = _coordinated_backup_plan(source, destination)
            if plan["plan_hash"] != confirm:
                raise PingstoreError("backup plan changed; run --dry-run again")
            journal = dict(
                schema="pingstore.operation/v1",
                plan=plan,
                phase="prepared",
                error=None,
                operation=dict(
                    kind="backup",
                    plan_hash=confirm,
                    token=secrets.token_hex(32),
                    run_ids=sorted(row["run_id"] for row in plan["runs"]),
                ),
            )
            _journal_save(path, journal)
        try:
            current = read_control(destination)
            if journal["phase"] == "catalog-published" and _backup_already_released(
                plan, current
            ):
                journal["phase"] = "complete"
                journal["error"] = None
                _journal_save(path, journal)
                print(
                    "Backup completion and release verified from the surviving catalog."
                )
                return 0
            if current["operation"] != journal["operation"]:
                if current != plan["control"]:
                    raise PingstoreError(
                        "remote controls changed before backup lease acquisition"
                    )
                acquire_remote_operation(destination, current, journal["operation"])
            work = _backup_work_plan(source, destination, plan)
            work["tombstones"] = plan["control"]["tombstones"]
            work["publications"] = plan["control"]["publications"]
            assert_remote_operation(destination, journal["operation"])
            apply_backup(source, destination, work)
            journal["phase"] = "uploaded"
            _journal_save(path, journal)
            assert_remote_operation(destination, journal["operation"])
            publish_catalog(source, destination, work)
            journal["phase"] = "catalog-published"
            _journal_save(path, journal)
            release_remote_operation(destination, journal["operation"])
            journal["phase"] = "complete"
            journal["error"] = None
            _journal_save(path, journal)
            return 0
        except BaseException as exc:
            journal["error"] = f"{type(exc).__name__}: {exc}"
            _journal_save(path, journal)
            raise PingstoreError(
                f"backup incomplete ({journal['phase']}); journal {path}; resume --confirm {confirm}: {exc}"
            ) from exc


def public_maintenance_snapshot(asset_base):
    base = asset_base.rstrip("/")
    catalog = public_json(base + "/pingstore/catalog.json")
    validate_catalog(catalog, complete=False)
    try:
        control = public_json(base + "/pingstore/control.json")
        validate_publications(control["publications"])
    except OSError:
        control = dict(
            schema=CONTROL_SCHEMA,
            revision=0,
            operation=None,
            publications=empty_publications(),
            tombstones={},
        )
    objects = {
        f"{identity}/{name}": row["bytes"]
        for identity, entry in catalog["runs"].items()
        for name, row in entry["files"].items()
    }
    for identity in catalog.get("inventory", {}).get("protected", []):
        if not STAGE_ID_RE.fullmatch(identity):
            raise PingstoreError("invalid published protection identity")
        objects[f"{identity}/.protected"] = 0
    return dict(
        catalog=catalog,
        control=control,
        objects=objects,
        authenticated=False,
        deployment=None,
    )
