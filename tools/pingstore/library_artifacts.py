"""Preserve and validate native library evidence across export canonicalization."""

import tempfile
from pathlib import Path

from .contracts import PingstoreError, file_sha256, load_json


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


def validate_native_artifacts(export, mapping, *, snapshot_stages=None):
    """Authenticate native evidence, requiring pairs unless retention is explicit.

    snapshot_stages maps original execution directories to retained native stages.
    Store this policy in run.json so an intentionally single-stage capture remains
    distinguishable from a broken start/end pair.
    """
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
