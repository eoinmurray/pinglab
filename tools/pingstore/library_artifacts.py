"""Preserve and validate native library evidence across export canonicalization."""

import tempfile
from pathlib import Path

from .contracts import PingstoreError, file_sha256, load_json


def native_artifact_files(export):
    """Find complete bundles and snapshots, without relying on a bundle suffix."""
    files = set()
    for path in export.rglob("*.json"):
        if path.name not in {"manifest.json", "snapshot.json"}:
            continue
        metadata = load_json(path)
        if metadata.get("schema") == "snnlang.bundle/v1":
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


def validate_native_artifacts(export, mapping):
    """Authenticate preserved bundles and require complete start/end snapshots."""
    bundles, snapshots = [], {}
    for source, target in mapping.items():
        if Path(source).name not in {"manifest.json", "snapshot.json"}:
            continue
        metadata = load_json(export / target)
        if metadata.get("schema") == "snnlang.bundle/v1":
            bundles.append(Path(source).parent)
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
    for root, stages in snapshots.items():
        if stages != {"start", "end"}:
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
