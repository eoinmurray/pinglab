"""Native SNNLab evidence for high-level requests and maintained PyTorch loops."""

from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import numpy as np
from pingstore.contracts import PingstoreError, file_sha256, load_json
from snnlab import lang
from snnlab.sim import ExecutionSpec, SnapshotSpec
from snnlab.sim.artifacts.snapshots import SnapshotSession


def save_bundle(bundle, directory):
    """Write and authenticate the complete authoring product, including assets."""
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(f"network bundle already exists: {directory}")
    bundle.write(directory)
    lang.load_bundle(directory)
    return directory


def snapshot_request(spec, bundle, directory):
    """Keep the authored bundle and ask SNNLab to snapshot the resolved execution."""
    directory = Path(directory)
    if spec.kind == "build":
        raise ValueError("build is not a simulation, training or inference execution")
    save_bundle(bundle, directory / "network.bundle")
    return replace(
        spec, snapshot=SnapshotSpec(directory, stage="both", parameters=True)
    )


@contextmanager
def model_execution(
    model, bundle, directory, *, kind="simulate", seed=0, protocol=None, settings=None
):
    """Snapshot one scientific execution, including all its batches or time chunks.

    SNNLab 0.17's request handlers own this same lifecycle. Isolate its internal
    SnapshotSession here until SNNLab exposes a public context for custom loops;
    never duplicate the library's metadata or parameter serialization.
    """
    directory = Path(directory)
    save_bundle(bundle, directory / "network.bundle")
    device = str(next(iter(model.parameter_map().values())).device)
    spec = ExecutionSpec(
        kind=kind,
        graph=bundle.graph,
        training=bundle.training,
        seed=seed,
        device=device,
        protocol=protocol or {},
        options=settings or {},
        snapshot=SnapshotSpec(directory, stage="both", parameters=True),
    )
    session = SnapshotSession(
        spec,
        model,
        kind=kind,
        protocol=spec.protocol,
        training=bundle.training,
        device_resolved=device,
    )
    session.capture("start")
    yield session
    session.capture("end")


def snapshot_parameters(source, *parts):
    """Read named arrays from a validated source's native snapshot inventory."""
    metadata = load_json(source.file(*parts, "snapshot.json"))
    if metadata.get("schema") != "snnlab.run-snapshot/v1" or not metadata.get(
        "parameters_included"
    ):
        raise PingstoreError("expected a native parameter snapshot")
    if not metadata["parameters"]:
        return {}
    path = source.file(*parts, "parameters.npz")
    if file_sha256(path) != metadata.get("parameter_file_sha256"):
        raise PingstoreError("snapshot parameter checksum mismatch")
    with np.load(path, allow_pickle=False) as arrays:
        values = {}
        for name, entry in metadata["parameters"].items():
            value = arrays[entry["key"]]
            if (
                list(value.shape) != entry["shape"]
                or str(value.dtype) != entry["storage_dtype"]
            ):
                raise PingstoreError(
                    f"snapshot parameter shape or dtype mismatch: {name}"
                )
            if entry["dtype"] == "bfloat16":
                raise PingstoreError("use SNNLab tensors to read bfloat16 parameters")
            values[name] = value.copy()
    return values
