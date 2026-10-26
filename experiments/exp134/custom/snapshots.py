"""Capture native evidence for the frozen membrane intervention loop.

The compute coordinator calls this adapter after installing checkpoint weights.
SNNLab owns snapshot metadata and parameter serialization.
"""

from pathlib import Path

from snnlab.sim import SnapshotSpec
from snnlab.sim.artifacts.snapshots import SnapshotSession
from snnlab.sim.requests.spec import ExecutionSpec


def snapshot_start(
    model,
    bundle,
    directory,
    cfg,
):
    """Capture the resolved network and frozen weights before the custom loop."""
    device = str(next(iter(model.parameter_map().values())).device)
    spec = ExecutionSpec(
        kind="simulate",
        graph=bundle.graph,
        training=bundle.training,
        seed=cfg["model_seed"],
        device=device,
        protocol={},
        **cfg["performance"],
        snapshot=SnapshotSpec(
            path=Path(directory) / "start",
            stage="start",
            parameters=True,
        ),
    )
    session = SnapshotSession(
        spec,
        model,
        kind="simulate",
        protocol=spec.protocol,
        training=bundle.training,
        device_resolved=device,
    )
    session.capture("start")
