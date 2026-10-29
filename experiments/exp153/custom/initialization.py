"""Retain native start evidence for an initialized, unadvanced reference network.

The public build API initializes weights without simulating spikes but has no
snapshot option. This adapter uses SNNLab's native capture lifecycle; it does not
reimplement graph metadata or parameter serialization.
"""

from snnlab.sim import SnapshotSpec
from snnlab.sim.artifacts.snapshots import SnapshotSession
from snnlab.sim.requests.spec import ExecutionSpec


def capture_start(bundle, initialized, cfg, directory):
    spec = ExecutionSpec(
        kind="simulate",
        graph=bundle.graph,
        seed=cfg["network"]["network_seed"],
        device="cpu",
        compile_mode=None,
        protocol=dict(timing=dict(duration_ms=0.0), operation="initialization_only"),
        snapshot=SnapshotSpec(directory, stage="start", parameters=True),
    )
    lifecycle = SnapshotSession(
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
    lifecycle.capture("start")
