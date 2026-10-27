"""Native SNNLab execution evidence for exp142.

Retain only the bundle, snapshot lifecycle and parameter readers used by this
experiment. SNNLab owns snapshot metadata and parameter serialization."""

from contextlib import contextmanager
from pathlib import Path

import numpy as np
from pingstore.library import PingstoreError, file_sha256, load_json
from snnlab import lang
from snnlab.sim import SnapshotSpec
from snnlab.sim.artifacts.snapshots import SnapshotSession


def save_bundle(bundle, directory, *, visualise=False):
    """Write and authenticate the complete authoring product, including assets."""
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(f"network bundle already exists: {directory}")
    bundle.write(directory, visualise=visualise)
    lang.load_bundle(directory)
    return directory


@contextmanager
def model_execution(
    model,
    bundle,
    directory,
    *,
    kind="simulate",
    seed=0,
    protocol=None,
    settings=None,
    snapshot_stages=("start",),
):
    """Snapshot one scientific execution, including all its batches or time chunks.

    Use SNNLab's native lifecycle for this experiment's custom loop. Simulation
    retains start evidence only; training explicitly requests end evidence.
    SNNLab owns metadata and parameter serialization.
    """
    from snnlab.sim.requests.spec import ExecutionSpec

    if snapshot_stages not in (("start",), ("end",), ("start", "end")):
        raise ValueError("snapshot_stages must be start, end, or start/end")
    stage = "both" if len(snapshot_stages) == 2 else snapshot_stages[0]
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
        snapshot=SnapshotSpec(
            directory if stage == "both" else directory / stage,
            stage=stage,
            parameters=True,
        ),
    )
    session = SnapshotSession(
        spec,
        model,
        kind=kind,
        protocol=spec.protocol,
        training=bundle.training,
        device_resolved=device,
    )
    if "start" in snapshot_stages:
        session.capture("start")
    yield session
    if "end" in snapshot_stages:
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
