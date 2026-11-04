"""Native snapshot lifecycle for this experiment's custom training loop."""

from snnlab.sim import SnapshotSpec
from snnlab.sim.artifacts.snapshots import SnapshotSession
from snnlab.sim.requests.spec import ExecutionSpec


def capture(model, bundle, directory, cfg, *, kind, stage, epoch=None):
    device = str(next(iter(model.parameter_map().values())).device)
    spec = ExecutionSpec(
        kind=kind,
        graph=bundle.graph,
        training=bundle.training,
        seed=cfg["seed"],
        device=device,
        compile_mode=None,
        protocol={"timing": {"duration_ms": cfg["duration_ms"]}},
        options=cfg,
        snapshot=SnapshotSpec(directory / stage, stage=stage, parameters=True),
    )
    session = SnapshotSession(
        spec,
        model,
        kind=kind,
        protocol=spec.protocol,
        training=bundle.training,
        device_resolved=device,
    )
    session.capture(stage, epoch=epoch)
