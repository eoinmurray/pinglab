"""Run the native graph in float64 and capture its native start snapshot.

The installed simulate() API has no dtype selector. This small adapter uses
GraphExecutor.to(float64), with SNNLab's own snapshot lifecycle rather than
reimplementing metadata or parameter serialization. No custom time loop is used.
"""

import time

import torch
from snnlab.sim import DenseArrayBinding, GraphExecutor, SnapshotSpec, plan_graph
from snnlab.sim.artifacts.snapshots import SnapshotSession
from snnlab.sim.requests.spec import ExecutionSpec


def execute(bundle, drive, cfg, *, snapshot_path=None):
    model = GraphExecutor(
        plan_graph(bundle.graph),
        seed=cfg["seed"],
        extension_validation="strict",
        compile_mode=cfg["performance"]["compile_mode"],
        compile_block_steps=cfg["performance"]["compile_block_steps"],
        projection_mode=cfg["performance"]["projection_mode"],
        precision=cfg["performance"]["precision"],
        cuda_graphs=cfg["performance"]["cuda_graphs"],
    ).to(device=cfg["device"], dtype=torch.float64)
    if snapshot_path is not None:
        # Internal request type is confined to the native snapshot adapter.
        spec = ExecutionSpec(
            kind="simulate",
            graph=bundle.graph,
            seed=cfg["seed"],
            device=cfg["device"],
            input_bindings=(DenseArrayBinding("drive", drive),),
            extension_validation="strict",
            compile_mode=cfg["performance"]["compile_mode"],
            compile_block_steps=cfg["performance"]["compile_block_steps"],
            projection_mode=cfg["performance"]["projection_mode"],
            precision=cfg["performance"]["precision"],
            cuda_graphs=cfg["performance"]["cuda_graphs"],
            snapshot=SnapshotSpec(snapshot_path, stage="start", parameters=True),
        )
        session = SnapshotSession(
            spec,
            model,
            kind="simulate",
            protocol={"duration_ms": cfg["duration_ms"]},
            device_resolved=cfg["device"],
            dtype=cfg["dtype"],
            initialization=model.initialization_metadata,
        )
        session.capture("start")
    started = time.perf_counter()
    with torch.no_grad():
        result = model({"drive": drive}, diagnostics=False)
    elapsed = time.perf_counter() - started
    return result, elapsed
