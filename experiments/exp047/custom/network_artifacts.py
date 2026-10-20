"""Native SNNLab execution evidence for exp047.

Retain only the bundle, snapshot lifecycle and parameter readers used by this
experiment. SNNLab owns snapshot metadata and parameter serialization."""

from contextlib import contextmanager
from pathlib import Path

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
    snapshot_stages=("start", "end"),
):
    """Snapshot one scientific execution, including all its batches or time chunks.

    Use SNNLab's internal SnapshotSession for this experiment's custom loop;
    leave metadata and parameter serialization to the library. Callers may retain
    only start or end evidence; existing callers keep the start/end pair.
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
