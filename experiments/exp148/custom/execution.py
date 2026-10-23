"""Continue one native graph while scaling only its external-to-I weights.

The installed simulator has no time-varying projection-weight binding. This
adapter retains native runtime state across calls, updates one parameter between
calls, and captures the native simulation-start snapshot without serialization
of custom model state.
"""

import hashlib
import time

import numpy as np
import torch
from experiments.exp148.recipe import drive_gain, execution_bounds
from snnlab.sim import DenseArrayBinding, GraphExecutor, SnapshotSpec, plan_graph
from snnlab.sim.artifacts.snapshots import SnapshotSession
from snnlab.sim.requests.spec import ExecutionSpec


def executor(bundle, cfg):
    performance = cfg["performance"]
    return GraphExecutor(
        plan_graph(bundle.graph),
        seed=cfg["seed"],
        compile_mode=performance["compile_mode"],
        precision=performance["precision"],
        cuda_graphs=performance["cuda_graphs"],
    ).to(cfg["device"])


def weight_hashes(parameters):
    return {
        name: hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()
        for name, value in parameters.items()
    }


def execute(bundle, external, cfg, *, snapshot_path=None, progress=False):
    model = executor(bundle, cfg)
    parameters = model.parameter_map()
    parameter = cfg["control"]["parameter"]
    original = parameters[parameter].detach().clone()
    hashes = weight_hashes(parameters)
    if snapshot_path is not None:
        spec = ExecutionSpec(
            kind="simulate",
            graph=bundle.graph,
            seed=cfg["seed"],
            device=cfg["device"],
            input_bindings=(DenseArrayBinding("X", external),),
            compile_mode=cfg["performance"]["compile_mode"],
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
            dtype="float32",
            initialization=model.initialization_metadata,
        )
        session.capture("start")
    events = {population: [[], []] for population in ("E", "I")}
    state = None
    started = time.monotonic()
    bounds = execution_bounds(cfg)
    with torch.no_grad():
        for start, end in zip(bounds[:-1], bounds[1:]):
            parameters[parameter].copy_(
                original * drive_gain(start * cfg["dt_ms"], cfg)
            )
            result = model(
                {"X": external[start:end]},
                diagnostics=False,
                runtime_state=state,
                retain_parameters=False,
                retain_final_state=False,
            )
            state = result.runtime_state
            for population in events:
                times, cells = np.nonzero(
                    result.outputs[population + "_spikes"][:, 0].cpu().numpy()
                )
                # Preserve the scratch probe's zero-based sample-index convention.
                events[population][0].append((times + start) * cfg["dt_ms"])
                events[population][1].append(cells)
            if progress and end % round(cfg["execution_block_ms"] / cfg["dt_ms"]) == 0:
                print(
                    f"{end * cfg['dt_ms']:.0f}/{cfg['duration_ms']:g} ms simulated; "
                    f"elapsed {time.monotonic() - started:.1f}s",
                    flush=True,
                )
    elapsed = time.monotonic() - started
    final_hashes = weight_hashes(parameters)
    if any(
        final_hashes[name] != digest
        for name, digest in hashes.items()
        if name != parameter
    ):
        raise ValueError("a weight outside the scheduled projection changed")
    torch.testing.assert_close(
        parameters[parameter], original * cfg["control"]["final_gain"], rtol=0, atol=0
    )
    arrays = {
        population + "_" + quantity: np.concatenate(values)
        for population, pair in events.items()
        for quantity, values in zip(("time_ms", "cell"), pair)
    }
    audit = dict(
        initial_weight_sha256=hashes,
        final_weight_sha256=final_hashes,
        unchanged_other_weights=True,
        final_gain_verified=True,
        execution_seconds=elapsed,
        executor_calls=len(bounds) - 1,
        timing_scope="All native calls and control/event collection; excludes build and snapshot",
        spike_time_convention="zero-based output sample index multiplied by dt_ms",
    )
    return arrays, audit, result
