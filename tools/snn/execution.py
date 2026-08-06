"""Typed execution seam and graph-native forward executor.

The bundle is data.  This module intentionally has no dependency on snnlang.
Legacy requests continue to route through the existing CLI handlers; graph
requests are planned once and execute a fixed vectorised schedule per step.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
import tracemalloc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Mapping

import models as M
import numpy as np
import torch
from bundle import load_graph_bundle
from torch import nn

ExecutorName = Literal["legacy", "graph"]
RequestKind = Literal["build", "simulate", "train", "infer"]


@dataclass(frozen=True)
class ExecutionSpec:
    kind: RequestKind
    executor: ExecutorName = "legacy"
    bundle: Path | None = None
    graph: Mapping[str, Any] | None = None
    inputs: Mapping[str, torch.Tensor] = field(default_factory=dict)
    seed: int = 0
    device: str = "cpu"
    record: bool = True
    checkpoint: Path | None = None
    runtime_state: GraphRuntimeState | None = None
    options: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    executor: ExecutorName
    outputs: dict[str, torch.Tensor] = field(default_factory=dict)
    recordings: dict[str, torch.Tensor] = field(default_factory=dict)
    parameters: dict[str, torch.Tensor] = field(default_factory=dict)
    final_state: dict[str, torch.Tensor] = field(default_factory=dict)
    runtime_state: GraphRuntimeState | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    model: nn.Module | None = None


@dataclass(frozen=True)
class CapabilityIssue:
    element: str
    capability: str
    message: str


GRAPH_CAPABILITIES_V1 = {
    "schema": "tools/snn.capabilities/v1",
    "neurons": {"coba_lif", "leaky_integrator"},
    "synapses": {"ampa", "gaba", "leaky_integrator"},
    "operations": {"reduce_mean"},
    "connections": {"feedforward", "recurrent", "feedback"},
    "recordings": {"spikes", "voltage"},
    "delays": "integer_steps",
    "training": False,
}


def graph_capability_issues(graph: Mapping[str, Any]) -> list[CapabilityIssue]:
    """Return precise graph-executor capability failures."""
    issues: list[CapabilityIssue] = []
    neuron_capabilities: set[str] = {"coba_lif", "leaky_integrator"}
    synapse_capabilities: set[str] = {"ampa", "gaba", "leaky_integrator"}
    operation_capabilities: set[str] = {"reduce_mean"}
    connection_capabilities: set[str] = {"feedforward", "recurrent", "feedback"}
    for pop in graph.get("populations", []):
        kind = pop.get("neuron", {}).get("kind")
        if kind not in neuron_capabilities:
            issues.append(CapabilityIssue(pop["id"], f"neuron:{kind}", "unsupported neuron kind"))
    for projection in graph.get("projections", []):
        synapse = projection.get("synapse", {}).get("kind")
        if synapse not in synapse_capabilities:
            issues.append(CapabilityIssue(projection["id"], f"synapse:{synapse}", "unsupported synapse kind"))
        connection = projection.get("connection")
        if connection not in connection_capabilities:
            issues.append(CapabilityIssue(projection["id"], f"connection:{connection}", "unsupported connection kind"))
    for operation in graph.get("operations", []):
        kind = operation.get("kind")
        if kind not in operation_capabilities:
            issues.append(CapabilityIssue(operation["id"], f"operation:{kind}", "unsupported operation kind"))
    return issues


@dataclass(frozen=True)
class PlannedProjection:
    id: str
    source: str
    target: str
    polarity: str
    decay: float
    delay_steps: int
    parameter: str


@dataclass(frozen=True)
class GraphPlan:
    graph: Mapping[str, Any]
    dt_ms: float
    populations: tuple[Mapping[str, Any], ...]
    projections: tuple[PlannedProjection, ...]
    observables: tuple[Mapping[str, Any], ...]
    outputs: tuple[Mapping[str, Any], ...]


RUNTIME_STATE_SCHEMA = "tools/snn.graph-runtime-state/v1"


@dataclass
class GraphRuntimeState:
    """Complete dynamic state required to continue one graph trajectory.

    Static parameters are deliberately excluded: weight checkpoints and runtime
    state are orthogonal, allowing a mature state to branch across compatible
    parameterisations of the same graph structure.
    """

    signature: str
    compatibility: dict[str, Any]
    completed_steps: int
    voltages: dict[str, torch.Tensor]
    refractory: dict[str, torch.Tensor]
    conductances: dict[str, torch.Tensor]
    population_histories: dict[str, torch.Tensor]
    input_histories: dict[str, torch.Tensor]

    def detached(self, *, device: str | torch.device = "cpu") -> GraphRuntimeState:
        def moved(values: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            return {name: value.detach().to(device).clone() for name, value in values.items()}

        return GraphRuntimeState(
            signature=self.signature,
            compatibility=self.compatibility,
            completed_steps=self.completed_steps,
            voltages=moved(self.voltages),
            refractory=moved(self.refractory),
            conductances=moved(self.conductances),
            population_histories=moved(self.population_histories),
            input_histories=moved(self.input_histories),
        )


def runtime_state_compatibility(plan: GraphPlan) -> dict[str, Any]:
    """Describe state-layout and dynamical semantics, excluding parameter values."""
    parameters = {row["id"]: row for row in plan.graph.get("parameters", [])}
    return {
        "schema": RUNTIME_STATE_SCHEMA,
        "dt_ms": plan.dt_ms,
        "populations": [
            {"id": row["id"], "size": row["size"], "neuron": row["neuron"]}
            for row in plan.populations
        ],
        "inputs": [
            {"id": row["id"], "shape": row["shape"], "signal_type": row.get("signal_type")}
            for row in plan.graph.get("inputs", [])
        ],
        "projections": [
            {
                "id": row.id,
                "source": row.source,
                "target": row.target,
                "polarity": row.polarity,
                "synapse": next(
                    item["synapse"] for item in plan.graph.get("projections", []) if item["id"] == row.id
                ),
                "delay_steps": row.delay_steps,
                "parameter": row.parameter,
                "parameter_shape": parameters[row.parameter]["shape"],
            }
            for row in plan.projections
        ],
    }


def runtime_state_signature(plan: GraphPlan) -> str:
    payload = runtime_state_compatibility(plan)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _compatibility_mismatch(expected: Any, actual: Any, path: str = "graph") -> str | None:
    if type(expected) is not type(actual):
        return f"{path} expected {expected!r}, got {actual!r}"
    if isinstance(expected, dict):
        if set(expected) != set(actual):
            return f"{path} keys expected {sorted(expected)}, got {sorted(actual)}"
        for key in expected:
            mismatch = _compatibility_mismatch(expected[key], actual[key], f"{path}.{key}")
            if mismatch:
                return mismatch
    elif isinstance(expected, list):
        if len(expected) != len(actual):
            return f"{path} length expected {len(expected)}, got {len(actual)}"
        for index, (left, right) in enumerate(zip(expected, actual)):
            mismatch = _compatibility_mismatch(left, right, f"{path}[{index}]")
            if mismatch:
                return mismatch
    elif expected != actual:
        return f"{path} expected {expected!r}, got {actual!r}"
    return None


def _file_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def save_runtime_state(path: str | Path, state: GraphRuntimeState) -> Path:
    """Atomically publish a portable JSON/NPZ graph-runtime state directory."""
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    groups = {
        "voltages": state.voltages,
        "refractory": state.refractory,
        "conductances": state.conductances,
        "population_histories": state.population_histories,
        "input_histories": state.input_histories,
    }
    arrays: dict[str, np.ndarray] = {}
    tensors: list[dict[str, Any]] = []
    for group_name, values in groups.items():
        for name in sorted(values):
            key = f"tensor_{len(arrays):04d}"
            value = values[name].detach().cpu().contiguous()
            arrays[key] = value.numpy()
            tensors.append({
                "group": group_name,
                "name": name,
                "key": key,
                "shape": list(value.shape),
                "dtype": str(value.dtype).removeprefix("torch."),
            })
    fd, temporary_name = tempfile.mkstemp(prefix=".tensors-", suffix=".npz", dir=root)
    os.close(fd)
    temporary_tensors = Path(temporary_name)
    try:
        np.savez_compressed(temporary_tensors, **arrays)
        tensors_digest = _file_digest(temporary_tensors)
        os.replace(temporary_tensors, root / "tensors.npz")
    finally:
        temporary_tensors.unlink(missing_ok=True)
    manifest = {
        "schema": RUNTIME_STATE_SCHEMA,
        "schema_version": 1,
        "signature": state.signature,
        "compatibility": state.compatibility,
        "completed_steps": state.completed_steps,
        "tensors_file": "tensors.npz",
        "tensors_digest": tensors_digest,
        "tensors": tensors,
    }
    fd, temporary_name = tempfile.mkstemp(prefix=".manifest-", suffix=".json", dir=root)
    temporary_manifest = Path(temporary_name)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(manifest, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
        os.replace(temporary_manifest, root / "manifest.json")
    finally:
        temporary_manifest.unlink(missing_ok=True)
    return root


def load_runtime_state(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> GraphRuntimeState:
    """Load and authenticate a portable graph-runtime state artifact."""
    root = Path(path)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema") != RUNTIME_STATE_SCHEMA or manifest.get("schema_version") != 1:
        raise ValueError(f"unsupported runtime-state schema: {manifest.get('schema')}")
    tensors_path = root / manifest.get("tensors_file", "tensors.npz")
    actual_digest = _file_digest(tensors_path)
    if actual_digest != manifest.get("tensors_digest"):
        raise ValueError(
            f"runtime-state tensors digest expected {manifest.get('tensors_digest')}, got {actual_digest}"
        )
    groups: dict[str, dict[str, torch.Tensor]] = {
        "voltages": {},
        "refractory": {},
        "conductances": {},
        "population_histories": {},
        "input_histories": {},
    }
    with np.load(tensors_path, allow_pickle=False) as archive:
        expected_keys = {row["key"] for row in manifest["tensors"]}
        if set(archive.files) != expected_keys:
            raise ValueError(
                f"runtime-state tensor keys expected {sorted(expected_keys)}, got {sorted(archive.files)}"
            )
        for row in manifest["tensors"]:
            array = archive[row["key"]]
            if list(array.shape) != row["shape"] or str(array.dtype) != row["dtype"]:
                raise ValueError(
                    f"runtime-state tensor {row['group']}.{row['name']} metadata does not match tensors.npz"
                )
            groups[row["group"]][row["name"]] = torch.from_numpy(array.copy()).to(device)
    return GraphRuntimeState(
        signature=manifest["signature"],
        compatibility=manifest["compatibility"],
        completed_steps=int(manifest["completed_steps"]),
        voltages=groups["voltages"],
        refractory=groups["refractory"],
        conductances=groups["conductances"],
        population_histories=groups["population_histories"],
        input_histories=groups["input_histories"],
    )


class DelayBuffer:
    """Fixed causal delay used by recurrent and feedback projections."""

    def __init__(self, delay_steps: int, prototype: torch.Tensor):
        if delay_steps < 1:
            raise ValueError("causal delay buffer requires at least one step")
        self.delay_steps = delay_steps
        self._values = [torch.zeros_like(prototype) for _ in range(delay_steps)]

    def read(self) -> torch.Tensor:
        return self._values[0]

    def push(self, value: torch.Tensor) -> None:
        self._values.append(value)
        self._values.pop(0)

    def export(self) -> torch.Tensor:
        return torch.stack([value.detach().clone() for value in self._values])

    @classmethod
    def restore(cls, values: torch.Tensor) -> DelayBuffer:
        if values.ndim < 2 or values.shape[0] < 1:
            raise ValueError("delay history must have shape [delay, batch, ...]")
        result = cls(int(values.shape[0]), values[0])
        result._values = [value.detach().clone() for value in values.unbind(0)]
        return result


def plan_graph(graph: Mapping[str, Any]) -> GraphPlan:
    issues = graph_capability_issues(graph)
    if issues:
        detail = "; ".join(f"{x.element} requires {x.capability}: {x.message}" for x in issues)
        raise ValueError(f"graph executor capability failure: {detail}")
    dt = float(graph["timebase"]["dt"]["value"])
    if dt <= 0:
        raise ValueError("graph timebase dt must be positive")
    planned = []
    for row in graph.get("projections", []):
        delay = row.get("delay")
        delay_ms = 0.0 if delay is None else float(delay["value"])
        raw_steps = delay_ms / dt
        steps = int(round(raw_steps))
        if not math.isclose(raw_steps, steps, abs_tol=1e-9):
            raise ValueError(f"{row['id']}: delay {delay_ms} ms is not an integer number of dt={dt} ms steps")
        source_owner = row["source"].partition(".")[0]
        # Recurrent/feedback edges are causal even when the author declares
        # zero additional delay. Feedforward population edges may consume the
        # current-step source after topological scheduling.
        if source_owner in {p["id"] for p in graph.get("populations", [])} and row.get("connection") != "feedforward":
            steps = max(1, steps)
        tau = float(row["synapse"]["tau"]["value"])
        decay = 0.0 if row["synapse"]["kind"] == "leaky_integrator" else math.exp(-dt / tau)
        planned.append(PlannedProjection(
            id=row["id"], source=row["source"], target=row["target"],
            polarity=row["polarity"], decay=decay,
            delay_steps=steps, parameter=row["parameters"][0],
        ))
    populations = list(graph.get("populations", []))
    population_ids = {p["id"] for p in populations}
    zero_edges = [
        (p.source.partition(".")[0], p.target.partition(".")[0])
        for p in planned
        if p.delay_steps == 0 and p.source.partition(".")[0] in population_ids
    ]
    ordered: list[Mapping[str, Any]] = []
    remaining = {p["id"]: p for p in populations}
    while remaining:
        ready = sorted(
            name for name in remaining
            if not any(dst == name and src in remaining for src, dst in zero_edges)
        )
        if not ready:
            raise ValueError("zero-delay population projections form an algebraic cycle")
        for name in ready:
            ordered.append(remaining.pop(name))
    return GraphPlan(
        graph=graph, dt_ms=dt,
        populations=tuple(ordered),
        projections=tuple(planned),
        observables=tuple(graph.get("observables", [])),
        outputs=tuple(graph.get("outputs", [])),
    )


class GraphExecutor(nn.Module):
    """Dense graph executor whose graph topology is lowered before simulation."""

    def __init__(self, plan: GraphPlan, *, seed: int = 0):
        super().__init__()
        self.plan = plan
        torch.manual_seed(seed)
        rows = {row["id"]: row for row in plan.graph.get("parameters", [])}
        self.weights = nn.ParameterDict()
        pop_ids = {p["id"] for p in plan.populations}
        def init_priority(projection: PlannedProjection) -> tuple[int, str]:
            source = projection.source.partition(".")[0]
            target = projection.target.partition(".")[0]
            target_kind = next(p["neuron"]["kind"] for p in plan.populations if p["id"] == target)
            if source not in pop_ids:
                return (0, projection.id)
            if target_kind == "leaky_integrator":
                return (1, projection.id)
            if projection.polarity == "excitatory":
                return (2, projection.id)
            return (3, projection.id)
        realised: dict[str, torch.Tensor] = {}
        for projection in sorted(plan.projections, key=init_priority):
            row = rows[projection.parameter]
            init = row["initializer"]
            shape = tuple(reversed(row["shape"]))  # runtime is [source, target]
            if init["kind"] == "normal":
                value = torch.randn(*shape).mul_(float(init["std"])).add_(float(init["mean"])).clamp_(min=0)
            elif init["kind"] == "constant":
                value = torch.full(shape, float(init["value"]))
            else:
                raise ValueError(f"{projection.parameter}: unsupported initializer {init['kind']}")
            value = value / shape[0]
            realised[projection.parameter] = value
        for projection in plan.projections:
            self.weights[projection.parameter.replace(".", "__")] = nn.Parameter(realised[projection.parameter], requires_grad=False)

    def parameter_map(self) -> dict[str, torch.Tensor]:
        return {name.replace("__", "."): value for name, value in self.weights.items()}

    def forward(
        self,
        inputs: Mapping[str, torch.Tensor],
        *,
        record: bool = True,
        runtime_state: GraphRuntimeState | None = None,
    ) -> ExecutionResult:
        if not inputs:
            raise ValueError("graph execution requires at least one input tensor")
        first = next(iter(inputs.values()))
        if first.ndim == 2:
            inputs = {k: v.unsqueeze(1) for k, v in inputs.items()}
            first = next(iter(inputs.values()))
        steps, batch = first.shape[:2]
        device = first.device
        parameter_dtype = next(iter(self.weights.values())).dtype
        for name, value in inputs.items():
            if value.shape[:2] != (steps, batch):
                raise ValueError(
                    f"input {name} leading shape expected {(steps, batch)}, got {tuple(value.shape[:2])}"
                )
            if value.device != device:
                raise ValueError(f"input {name} device expected {device}, got {value.device}")
            if value.dtype != parameter_dtype:
                raise ValueError(f"input {name} dtype expected {parameter_dtype}, got {value.dtype}")
        populations = {p["id"]: p for p in self.plan.populations}
        population_history_lengths = {
            name: max((p.delay_steps for p in self.plan.projections if p.source.startswith(name + ".")), default=1)
            for name in populations
        }
        input_history_lengths = {
            row["id"]: max(
                (p.delay_steps for p in self.plan.projections if p.source.partition(".")[0] == row["id"]),
                default=0,
            )
            for row in self.plan.graph.get("inputs", [])
        }
        expected_compatibility = runtime_state_compatibility(self.plan)
        expected_signature = runtime_state_signature(self.plan)
        if runtime_state is None:
            voltage = {
                name: (
                    torch.zeros((batch, p["size"]), device=device)
                    if p["neuron"]["kind"] == "leaky_integrator"
                    else torch.full((batch, p["size"]), M.E_L, device=device)
                )
                for name, p in populations.items()
            }
            refractory = {
                name: torch.zeros((batch, p["size"]), dtype=torch.long, device=device)
                for name, p in populations.items()
            }
            spikes = {name: torch.zeros((batch, p["size"]), device=device) for name, p in populations.items()}
            conductance = {
                (p.id, p.polarity): torch.zeros(
                    (batch, populations[p.target.partition('.')[0]]["size"]), device=device
                )
                for p in self.plan.projections
            }
            histories = {
                name: DelayBuffer(population_history_lengths[name], value) for name, value in spikes.items()
            }
            input_histories = {
                name: torch.zeros((length, *inputs[name].shape[1:]), device=device, dtype=inputs[name].dtype)
                for name, length in input_history_lengths.items() if length > 0
            }
            completed_steps = 0
        else:
            if runtime_state.signature != expected_signature:
                detail = _compatibility_mismatch(expected_compatibility, runtime_state.compatibility)
                raise ValueError(
                    "runtime state is incompatible with graph plan: "
                    + (detail or f"signature expected {expected_signature}, got {runtime_state.signature}")
                )
            def restore_group(
                label: str,
                values: Mapping[str, torch.Tensor],
                shapes: Mapping[str, tuple[int, ...]],
            ) -> dict[str, torch.Tensor]:
                if set(values) != set(shapes):
                    raise ValueError(
                        f"runtime state {label} keys expected {sorted(shapes)}, got {sorted(values)}"
                    )
                restored = {}
                for name, expected_shape in shapes.items():
                    value = values[name]
                    if tuple(value.shape) != expected_shape:
                        raise ValueError(
                            f"runtime state {label}.{name} shape expected {expected_shape}, got {tuple(value.shape)}"
                        )
                    restored[name] = value.detach().to(device).clone()
                return restored

            pop_shapes = {name: (batch, int(row["size"])) for name, row in populations.items()}
            voltage = restore_group("voltages", runtime_state.voltages, pop_shapes)
            for name, value in voltage.items():
                if value.dtype != parameter_dtype:
                    raise ValueError(
                        f"runtime state voltages.{name} dtype expected {parameter_dtype}, got {value.dtype}"
                    )
            refractory = restore_group("refractory", runtime_state.refractory, pop_shapes)
            for name, value in refractory.items():
                if value.dtype != torch.long:
                    raise ValueError(f"runtime state refractory.{name} dtype expected torch.int64, got {value.dtype}")
            conductance_by_id = restore_group(
                "conductances",
                runtime_state.conductances,
                {
                    p.id: (batch, int(populations[p.target.partition('.')[0]]["size"]))
                    for p in self.plan.projections
                },
            )
            conductance = {(p.id, p.polarity): conductance_by_id[p.id] for p in self.plan.projections}
            for name, value in conductance_by_id.items():
                if value.dtype != parameter_dtype:
                    raise ValueError(
                        f"runtime state conductances.{name} dtype expected {parameter_dtype}, got {value.dtype}"
                    )
            population_history_values = restore_group(
                "population_histories",
                runtime_state.population_histories,
                {
                    name: (population_history_lengths[name], batch, int(row["size"]))
                    for name, row in populations.items()
                },
            )
            histories = {
                name: DelayBuffer.restore(value) for name, value in population_history_values.items()
            }
            for name, value in population_history_values.items():
                if value.dtype != parameter_dtype:
                    raise ValueError(
                        f"runtime state population_histories.{name} dtype expected {parameter_dtype}, got {value.dtype}"
                    )
            spikes = {name: histories[name]._values[-1].clone() for name in populations}
            input_histories = restore_group(
                "input_histories",
                runtime_state.input_histories,
                {
                    name: (length, *inputs[name].shape[1:])
                    for name, length in input_history_lengths.items() if length > 0
                },
            )
            for name, value in input_histories.items():
                if value.dtype != inputs[name].dtype:
                    raise ValueError(
                        f"runtime state input_histories.{name} dtype expected {inputs[name].dtype}, got {value.dtype}"
                    )
            completed_steps = int(runtime_state.completed_steps)
        recordings: dict[str, list[torch.Tensor]] = {o["id"]: [] for o in self.plan.observables}
        state_recordings: dict[str, list[torch.Tensor]] = {f"{name}.voltage": [] for name in populations}
        projection_recordings: dict[str, list[torch.Tensor]] = {f"{p.id}.conductance": [] for p in self.plan.projections}
        integrator_sum: dict[str, torch.Tensor] = {}

        for t in range(steps):
            new_spikes: dict[str, torch.Tensor] = {}
            for pop in self.plan.populations:
                name = pop["id"]
                incoming = {"excitatory": torch.zeros_like(voltage[name]), "inhibitory": torch.zeros_like(voltage[name])}
                for projection in self.plan.projections:
                    if projection.target.partition(".")[0] != name:
                        continue
                    key = (projection.id, projection.polarity)
                    source_owner = projection.source.partition(".")[0]
                    if source_owner in populations:
                        if projection.delay_steps == 0:
                            source = new_spikes[source_owner]
                        else:
                            history = histories[source_owner]._values
                            source = history[-projection.delay_steps]
                    else:
                        source_t = t - projection.delay_steps
                        source = (
                            inputs[source_owner][source_t]
                            if source_t >= 0
                            else input_histories[source_owner][source_t]
                        )
                    drive = source @ self.weights[projection.parameter.replace(".", "__")]
                    conductance[key] = conductance[key] * projection.decay + drive
                    incoming[projection.polarity] += conductance[key]
                neuron = pop["neuron"]
                if neuron["kind"] == "leaky_integrator":
                    beta = math.exp(-self.plan.dt_ms / float(neuron["tau"]["value"]))
                    voltage[name] = beta * voltage[name] + (1.0 - beta) / self.plan.dt_ms * incoming["excitatory"]
                    new_spikes[name] = torch.zeros_like(spikes[name])
                    integrator_sum[name] = integrator_sum.get(name, torch.zeros_like(voltage[name])) + voltage[name]
                    threshold = neuron.get("soft_reset_threshold")
                    if threshold is not None:
                        reset = M.fast_sigmoid_spike(
                            voltage[name] - float(threshold),
                            float(neuron.get("surrogate_slope", M.SURROGATE_SLOPE)),
                        )
                        voltage[name] = voltage[name] - reset * float(threshold)
                    continue
                tau_mem = float(neuron["tau_mem"]["value"])
                c_m = float(neuron.get("capacitance_nf", 1.0 if tau_mem >= 15 else 0.5))
                g_l = float(neuron.get("leak_us", c_m / tau_mem))
                ref_steps = int(neuron.get("refractory_steps", max(1, round((M.ref_ms_E if tau_mem >= 15 else M.ref_ms_I) / self.plan.dt_ms))))
                dampen = float(neuron.get("voltage_grad_dampen", M.V_GRAD_DAMPEN))
                voltage[name], new_spikes[name], refractory[name] = M.lif_step_expeuler(
                    voltage[name], refractory[name], incoming["excitatory"],
                    incoming["inhibitory"], c_m, g_l, ref_steps,
                    M.spike_biophysical, dt_override=self.plan.dt_ms,
                    v_grad_dampen=dampen,
                )
            spikes = new_spikes
            for name in populations:
                histories[name].push(spikes[name])
            if record:
                for observable in self.plan.observables:
                    owner, _, port = observable["signal"].partition(".")
                    recordings[observable["id"]].append((spikes if port == "spikes" else voltage)[owner].detach().clone())
                for name in populations:
                    state_recordings[f"{name}.voltage"].append(voltage[name].detach().clone())
                for projection in self.plan.projections:
                    projection_recordings[f"{projection.id}.conductance"].append(
                        conductance[(projection.id, projection.polarity)].detach().clone()
                    )

        outputs: dict[str, torch.Tensor] = {}
        operations = {o["id"]: o for o in self.plan.graph.get("operations", [])}
        for output in self.plan.outputs:
            op_id = output["signal"].partition(".")[0]
            op = operations[op_id]
            source_owner = op["sources"][0].partition(".")[0]
            if op["kind"] == "reduce_mean":
                outputs[output["id"]] = integrator_sum[source_owner] / steps
        packed = {k: torch.stack(v) for k, v in recordings.items() if v}
        packed.update({k: torch.stack(v) for k, v in state_recordings.items() if v})
        packed.update({k: torch.stack(v) for k, v in projection_recordings.items() if v})
        next_input_histories = {
            name: torch.cat((history, inputs[name]), dim=0)[-history.shape[0]:].detach().clone()
            for name, history in input_histories.items()
        }
        next_runtime_state = GraphRuntimeState(
            signature=expected_signature,
            compatibility=expected_compatibility,
            completed_steps=completed_steps + steps,
            voltages={name: value.detach().clone() for name, value in voltage.items()},
            refractory={name: value.detach().clone() for name, value in refractory.items()},
            conductances={p.id: conductance[(p.id, p.polarity)].detach().clone() for p in self.plan.projections},
            population_histories={name: history.export() for name, history in histories.items()},
            input_histories=next_input_histories,
        )
        return ExecutionResult(
            executor="graph", outputs=outputs, recordings=packed,
            parameters={k: v.detach().clone() for k, v in self.parameter_map().items()},
            final_state={f"{k}.voltage": v.detach().clone() for k, v in voltage.items()},
            runtime_state=next_runtime_state,
            model=self,
        )


def build(spec: ExecutionSpec) -> ExecutionResult:
    if spec.executor == "legacy":
        return ExecutionResult(executor="legacy", metrics={"request": "build", "routing": "legacy"})
    graph = spec.graph
    if graph is None and spec.bundle is not None:
        _, graph = load_graph_bundle(spec.bundle)
    if graph is None:
        raise ValueError("graph execution requires graph data or a bundle")
    started = time.perf_counter()
    model = GraphExecutor(plan_graph(graph), seed=spec.seed).to(spec.device)
    return ExecutionResult(executor="graph", model=model, parameters=model.parameter_map(), metrics={"build_s": time.perf_counter() - started})


def simulate(spec: ExecutionSpec, *, runtime_state: GraphRuntimeState | None = None) -> ExecutionResult:
    if spec.executor != "graph":
        return ExecutionResult(executor="legacy", metrics={"request": "simulate", "routing": "legacy"})
    built = build(spec)
    assert isinstance(built.model, GraphExecutor)
    if spec.checkpoint:
        built.model.load_state_dict(torch.load(spec.checkpoint, map_location=spec.device, weights_only=True))
    tracemalloc.start()
    started = time.perf_counter()
    result = built.model(
        {k: v.to(spec.device) for k, v in spec.inputs.items()},
        record=spec.record,
        runtime_state=runtime_state if runtime_state is not None else spec.runtime_state,
    )
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    result.metrics.update({"simulate_s": elapsed, "peak_python_bytes": peak, **built.metrics})
    if result.runtime_state is not None:
        result.metrics.update({
            "runtime_state_schema": RUNTIME_STATE_SCHEMA,
            "runtime_state_signature": result.runtime_state.signature,
            "completed_steps": result.runtime_state.completed_steps,
        })
    return result


def train(spec: ExecutionSpec) -> ExecutionResult:
    if spec.executor == "graph":
        raise NotImplementedError("graph training requires capability training:v1 (Milestone 6)")
    return ExecutionResult(executor="legacy", metrics={"request": "train", "routing": "legacy"})


def infer(spec: ExecutionSpec) -> ExecutionResult:
    return simulate(spec) if spec.executor == "graph" else ExecutionResult(executor="legacy", metrics={"request": "infer", "routing": "legacy"})


def execution_spec_from_args(args: Any, *, kind: RequestKind | None = None) -> ExecutionSpec:
    """Compatibility adapter: resolved CLI arguments become one typed request."""
    resolved_kind = kind or ("infer" if getattr(args, "infer", False) else args.mode)
    if resolved_kind == "sim":
        resolved_kind = "simulate"
    return ExecutionSpec(
        kind=resolved_kind,
        executor=getattr(args, "executor", "legacy"),
        bundle=Path(args.bundle) if getattr(args, "bundle", None) else None,
        seed=int(getattr(args, "seed", 0) or 0),
        device="cpu",
        checkpoint=(Path(args.load_weights) if getattr(args, "load_weights", None) else None),
        options={key: value for key, value in vars(args).items() if key not in {"bundle", "executor"}},
    )


def execute_request(
    spec: ExecutionSpec,
    *,
    legacy: Callable[[], ExecutionResult] | None = None,
) -> ExecutionResult:
    """Dispatch one typed request; the CLI supplies its unchanged legacy body."""
    if spec.executor == "legacy":
        if legacy is None:
            raise ValueError("legacy execution requires the registered legacy request body")
        return legacy()
    handlers = {"build": build, "simulate": simulate, "train": train, "infer": infer}
    return handlers[spec.kind](spec)
