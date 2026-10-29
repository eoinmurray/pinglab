"""Capture native SNNLab start evidence for the custom ROS streaming loop.

SNNLab owns graph metadata and parameter serialization. The adapter accesses the
prepared stream's private executor because its public API has no snapshot option.
"""

from pathlib import Path


def capture_start(stream, bundle, directory, cfg):
    from snnlab.sim import SnapshotSpec
    from snnlab.sim.artifacts.snapshots import SnapshotSession
    from snnlab.sim.requests.spec import ExecutionSpec

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    bundle.write(directory / "network.bundle", visualise=False)
    # Both sides share this architecture; one diagram is sufficient.
    if directory.name == "flexor":
        bundle.visualise(directory / "circuit.svg", view="circuit")
    spec = ExecutionSpec(
        kind="simulate",
        graph=bundle.graph,
        training=bundle.training,
        seed=cfg["seed"],
        device="cpu",
        protocol={
            "timing": {"dt_ms": cfg["dt_ms"], "duration_ms": cfg["duration"] * 1000}
        },
        snapshot=SnapshotSpec(directory / "start", stage="start", parameters=True),
    )
    lifecycle = SnapshotSession(
        spec,
        stream._executor,
        kind="simulate",
        protocol=spec.protocol,
        training=bundle.training,
        device_resolved="cpu",
        prepared_stream=stream.info,
    )
    lifecycle.capture("start")
