"""Validated, checksum-addressed source-run transfer for shared Modal jobs."""

import json
import shutil
from pathlib import Path

from pingstore.library import STAGE_ID_RE, PingstoreError, source_run


def input_path(reference):
    identity, digest = reference["run_id"], reference["payload_digest"]
    if not STAGE_ID_RE.fullmatch(identity) or not digest.startswith("sha256:"):
        raise PingstoreError("invalid Modal input pin")
    checksum = digest[7:]
    if len(checksum) != 64 or any(c not in "0123456789abcdef" for c in checksum):
        raise PingstoreError("invalid Modal input checksum")
    return f"inputs/{identity}/{checksum}"


def prepare_inputs(root, identities):
    """Resolve explicit roles and authenticate their complete source ancestry."""
    sources, active = {}, set()

    def visit(identity, reference=None):
        if identity in active:
            raise PingstoreError("cycle in Modal source ancestry")
        if identity in sources:
            if reference is not None and sources[identity].reference != reference:
                raise PingstoreError("conflicting Modal input pins")
            return sources[identity]
        active.add(identity)
        source = source_run(root, identity, reference=reference)
        for pin in source.record["inputs"].values():
            visit(pin["run_id"], pin)
        active.remove(identity)
        sources[identity] = source
        return source

    inputs = {role: visit(identity).reference for role, identity in identities.items()}
    return inputs, sources


def upload_inputs(volume, sources):
    """Reuse only checksum-addressed inputs; workers revalidate every file."""
    for source in sources.values():
        path = input_path(source.reference)
        try:
            record = json.loads(b"".join(volume.read_file(f"{path}/run.json")))
        except FileNotFoundError:
            record = None
        if record is not None:
            if (
                record.get("run_id") != source.record["run_id"]
                or record.get("payload_digest") != source.reference["payload_digest"]
            ):
                raise PingstoreError("cached Modal input identifies another payload")
        else:
            print(f"Uploading source {source.record['run_id']}", flush=True)
            with volume.batch_upload() as batch:
                batch.put_directory(source.directory, "/" + path)
        source.check_unchanged()


def restore_inputs(root, volume_root, references):
    """Atomically install and authenticate byte-preserved completed source runs."""
    root, volume_root = Path(root), Path(volume_root)
    (root / "runs").mkdir(parents=True, exist_ok=True)
    for reference in references:
        identity = reference["run_id"]
        path = input_path(reference)
        destination = root / "runs" / identity
        if not destination.exists():
            staging = destination.with_name(f".{identity}.tmp")
            # A failed restoration must not be reused without validation.
            if staging.exists():
                raise PingstoreError(f"incomplete Modal input restoration: {identity}")
            shutil.copytree(volume_root / path, staging)
            from pingstore.library import validate_operational_run_directory

            record = validate_operational_run_directory(staging)
            if (
                record["run_id"] != identity
                or record["payload_digest"] != reference["payload_digest"]
            ):
                raise PingstoreError("transferred Modal input checksum mismatch")
            staging.rename(destination)
        source_run(root, identity, reference=reference)
