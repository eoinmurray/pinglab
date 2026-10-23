"""Execute one native HCO graph in float64 with a native start snapshot.

simulate() has no dtype selector in SNNLab 0.24. This adapter confines the
internal snapshot lifecycle to one module and delegates all time stepping,
state continuation and parameter serialization to the native graph engine.
"""

import time

import torch
from snnlab.sim import DenseArrayBinding, GraphExecutor, SnapshotSpec, plan_graph
from snnlab.sim.artifacts.snapshots import SnapshotSession
from snnlab.sim.requests.spec import ExecutionSpec


def executor(bundle, cfg):
    performance = cfg["performance"]
    return GraphExecutor(
        plan_graph(bundle.graph),
        seed=cfg["seed"],
        extension_validation="strict",
        compile_mode=performance["compile_mode"],
        compile_block_steps=performance["compile_block_steps"],
        projection_mode=performance["projection_mode"],
        precision=performance["precision"],
        cuda_graphs=performance["cuda_graphs"],
    ).to(device=cfg["device"], dtype=torch.float64)


def execute(bundle, clock, cfg, *, snapshot_path=None):
    model = executor(bundle, cfg)
    if snapshot_path is not None:
        spec = ExecutionSpec(
            kind="simulate",
            graph=bundle.graph,
            seed=cfg["seed"],
            device=cfg["device"],
            input_bindings=(DenseArrayBinding("clock", clock),),
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
        result = model({"clock": clock}, diagnostics=False)
    return result, time.perf_counter() - started
