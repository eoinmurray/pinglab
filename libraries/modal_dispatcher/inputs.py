"""Explicit immutable run transfers for Modal jobs."""

import json
import shutil
from pathlib import Path

from pingstore.core import run_id
from pingstore.library import PingstoreError, source_run, validate_run_directory


def input_path(reference):
    return f"inputs/{run_id(reference['run_id'])}"


def prepare_inputs(root, identities, *, include_ancestry=True):
    sources, active = {}, set()

    def visit(identity):
        if identity in active:
            raise PingstoreError("cycle in Modal source ancestry")
        if identity in sources:
            return sources[identity]
        active.add(identity)
        source = source_run(root, identity)
        for upstream in source.record["depends_on"] if include_ancestry else ():
            visit(upstream)
        active.remove(identity)
        sources[identity] = source
        return source

    return {
        role: visit(identity).reference for role, identity in identities.items()
    }, sources


def upload_inputs(volume, sources):
    for source in sources.values():
        path = input_path(source.reference)
        try:
            record = json.loads(b"".join(volume.read_file(f"{path}/run.json")))
        except FileNotFoundError:
            record = None
        if record is not None:
            if record != source.record:
                raise PingstoreError("cached Modal input metadata differs")
        else:
            with volume.batch_upload() as batch:
                # The completion record goes last so interrupted uploads aren't reused.
                for file in sorted(source.directory.rglob("*")):
                    if file.is_file() and file.name != "run.json":
                        batch.put_file(
                            file,
                            "/"
                            + path
                            + "/"
                            + file.relative_to(source.directory).as_posix(),
                        )
            with volume.batch_upload() as batch:
                batch.put_file(source.directory / "run.json", "/" + path + "/run.json")
        source.check_unchanged()


def restore_inputs(root, volume_root, references):
    root, volume_root = Path(root), Path(volume_root)
    (root / "runs").mkdir(parents=True, exist_ok=True)
    for reference in references:
        identity = run_id(reference["run_id"])
        destination = root / "runs" / identity
        if not destination.exists():
            staging = destination.with_name(f".{identity}.tmp")
            shutil.copytree(volume_root / input_path(reference), staging)
            record = validate_run_directory(staging)
            if record["id"] != identity:
                raise PingstoreError("transferred Modal input identity mismatch")
            staging.rename(destination)
        source_run(root, identity, reference=reference)
