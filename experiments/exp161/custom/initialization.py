"""Native snapshot lifecycle for an initialized, unadvanced reference network."""

from snnlab.sim import SnapshotSpec
from snnlab.sim.artifacts.snapshots import SnapshotSession
from snnlab.sim.requests.spec import ExecutionSpec


def capture_start(bundle, initialized, cfg, path):
    spec = ExecutionSpec(
        kind="simulate",
        graph=bundle.graph,
        seed=cfg["seed"],
        device="cpu",
        compile_mode=None,
        protocol=dict(timing=dict(duration_ms=0.0), operation="initialization_only"),
        snapshot=SnapshotSpec(path, stage="start", parameters=True),
    )
    session = SnapshotSession(
        spec,
        initialized.model,
        kind="infer",
        protocol=spec.protocol,
        device_resolved="cpu",
        initialization=initialized.metrics["initialization"],
        autapses=initialized.metrics["autapses"],
        initialization_only=True,
        completed_steps=0,
    )
    session.capture("start")
