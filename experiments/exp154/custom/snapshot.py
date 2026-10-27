"""Capture native SNNLab evidence for this experiment training loop.

Only this adapter touches SNNLab's internal snapshot lifecycle. Graph and
parameter metadata are serialized by SNNLab, rather than a study-owned format.
Snapshots are inspectable evidence, not resumable training checkpoints.
"""

from snnlab.sim import SnapshotSpec
from snnlab.sim.artifacts.snapshots import SnapshotSession
from snnlab.sim.requests.spec import ExecutionSpec


def capture(model, bundle, directory, cfg, seed, *, stage, selection):
    protocol = dict(
        timing=dict(dt_ms=cfg["dt_ms"], duration_ms=cfg["presentation_ms"]),
        task=cfg["protocol"],
        activity_reset="each presentation; learned weights persist",
        selection=selection,
    )
    spec = ExecutionSpec(
        kind="train",
        graph=bundle.graph,
        seed=seed,
        device=cfg["device"],
        protocol=protocol,
        snapshot=SnapshotSpec(directory, stage=stage, parameters=True),
    )
    session = SnapshotSession(
        spec,
        model,
        kind="train",
        protocol=protocol,
        device_resolved=cfg["device"],
        custom_loop="exp154",
    )
    session.capture(stage)
