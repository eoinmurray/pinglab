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
from numbers import Integral
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

import models as M
import numpy as np
import torch
from bundle import load_graph_bundle
from torch import nn

ExecutorName = Literal["legacy", "graph"]
RequestKind = Literal["build", "simulate", "train", "infer"]
RecordingProfile = Literal["full", "observables", "none"]


DENSE_ARRAY_BINDING_SCHEMA = "tools/snn.dense-array-binding/v1"
EVENT_STREAM_BINDING_SCHEMA = "tools/snn.event-stream-binding/v1"
MIXED_INPUT_BINDING_SCHEMA = "tools/snn.mixed-input-bindings/v1"
POISSON_INPUT_BINDING_SCHEMA = "tools/snn.poisson-input-binding/v1"
EXECUTION_PROTOCOL_SCHEMA = "tools/snn.execution-protocol/v1"


@dataclass(frozen=True)
class DenseArrayBinding:
    """One concrete dense tensor resolved against a named graph input."""

    input_id: str
    value: torch.Tensor
    source: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EventStreamBinding:
    """Sparse binary spike events resolved against one named graph input."""

    input_id: str
    steps: torch.Tensor
    batches: torch.Tensor
    channels: torch.Tensor
    steps_count: int
    batch_size: int
    source: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PoissonInputBinding:
    """Generated Bernoulli-discretised Poisson spikes for one graph input."""

    input_id: str
    steps_count: int
    batch_size: int
    rates_hz: Sequence[float]
    seed: int
    categorical: bool = False


@dataclass(frozen=True)
class ResolvedDenseInputs:
    tensors: Mapping[str, torch.Tensor]
    protocol: Mapping[str, Any]


@dataclass(frozen=True)
class ExecutionSpec:
    kind: RequestKind
    executor: ExecutorName = "legacy"
    bundle: Path | None = None
    graph: Mapping[str, Any] | None = None
    inputs: Mapping[str, torch.Tensor] = field(default_factory=dict)
    input_bindings: Sequence[DenseArrayBinding] = field(default_factory=tuple)
    event_bindings: Sequence[EventStreamBinding] = field(default_factory=tuple)
    poisson_bindings: Sequence[PoissonInputBinding] = field(default_factory=tuple)
    protocol: Mapping[str, Any] = field(default_factory=dict)
    seed: int = 0
    device: str = "auto"
    recording: RecordingProfile = "full"
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
    "operations": {
        "linear",
        "reduce_mean",
        "reduce_sum",
        "select_final",
        "duration_normalise",
        "cumulative_sum",
    },
    "connections": {"feedforward", "recurrent", "feedback"},
    "recordings": {"spikes", "voltage"},
    "delays": "integer_steps",
    "training": False,
}


def load_dense_array_bindings(
    path: str | Path, graph: Mapping[str, Any]
) -> tuple[DenseArrayBinding, ...]:
    """Load a replayable NPY/NPZ file without weakening named-input semantics."""
    source_path = Path(path)
    digest = "sha256:" + hashlib.sha256(source_path.read_bytes()).hexdigest()
    loaded = np.load(source_path, allow_pickle=False)
    input_ids = [row["id"] for row in graph.get("inputs", [])]
    source_base = {
        "kind": "file",
        "path": str(source_path),
        "digest": digest,
    }
    if isinstance(loaded, np.ndarray):
        if len(input_ids) != 1:
            raise ValueError(
                "a dense NPY can bind only a graph with exactly one input; "
                f"graph inputs are {input_ids}"
            )
        return (
            DenseArrayBinding(
                input_ids[0],
                torch.as_tensor(loaded),
                {**source_base, "array": None},
            ),
        )
    try:
        arrays = {key: loaded[key] for key in loaded.files}
    finally:
        loaded.close()
    if set(arrays) == {"input_spikes"} and len(input_ids) == 1:
        arrays = {input_ids[0]: arrays["input_spikes"]}
        keys = {input_ids[0]: "input_spikes"}
    else:
        keys = {key: key for key in arrays}
    return tuple(
        DenseArrayBinding(
            input_id,
            torch.as_tensor(value),
            {**source_base, "array": keys[input_id]},
        )
        for input_id, value in arrays.items()
    )


def load_event_stream_bindings(
    path: str | Path, graph: Mapping[str, Any]
) -> tuple[EventStreamBinding, ...]:
    """Load named sparse spike coordinates from a replayable NPZ file."""
    source_path = Path(path)
    if source_path.suffix.lower() != ".npz":
        raise ValueError("event-stream replay requires an NPZ file")
    digest = "sha256:" + hashlib.sha256(source_path.read_bytes()).hexdigest()
    loaded = np.load(source_path, allow_pickle=False)
    try:
        arrays = {key: loaded[key] for key in loaded.files}
    finally:
        loaded.close()
    input_ids = [row["id"] for row in graph.get("inputs", [])]
    fields = ("steps", "batches", "channels", "steps_count", "batch_size")
    plain = set(fields)
    use_plain = len(input_ids) == 1 and set(arrays) == plain
    expected = (
        plain
        if use_plain
        else {f"{input_id}.{field}" for input_id in input_ids for field in fields}
    )
    if set(arrays) != expected:
        raise ValueError(
            "event-stream NPZ keys do not match graph inputs; "
            f"expected={sorted(expected)}, got={sorted(arrays)}"
        )
    source_base = {"kind": "file", "path": str(source_path), "digest": digest}
    bindings = []
    for input_id in input_ids:

        def key(field: str) -> str:
            return field if use_plain else f"{input_id}.{field}"

        steps_count_value = np.asarray(arrays[key("steps_count")])
        batch_size_value = np.asarray(arrays[key("batch_size")])
        if steps_count_value.size != 1 or batch_size_value.size != 1:
            raise ValueError(
                f"event input {input_id} steps_count and batch_size must be scalars"
            )
        if not np.issubdtype(steps_count_value.dtype, np.integer) or not np.issubdtype(
            batch_size_value.dtype, np.integer
        ):
            raise ValueError(
                f"event input {input_id} steps_count and batch_size must use integer dtypes"
            )
        bindings.append(
            EventStreamBinding(
                input_id=input_id,
                steps=torch.as_tensor(arrays[key("steps")]),
                batches=torch.as_tensor(arrays[key("batches")]),
                channels=torch.as_tensor(arrays[key("channels")]),
                steps_count=int(steps_count_value.item()),
                batch_size=int(batch_size_value.item()),
                source={
                    **source_base,
                    "arrays": {field: key(field) for field in fields},
                },
            )
        )
    return tuple(bindings)


def resolve_dense_array_bindings(
    graph: Mapping[str, Any],
    *,
    bindings: Sequence[DenseArrayBinding] = (),
    inputs: Mapping[str, torch.Tensor] | None = None,
    device: str | torch.device = "cpu",
    seed: int = 0,
    protocol: Mapping[str, Any] | None = None,
) -> ResolvedDenseInputs:
    """Validate dense arrays, resolve symbolic axes, and freeze run provenance."""
    if bindings and inputs:
        raise ValueError("provide dense input bindings or input tensors, not both")
    if not bindings:
        bindings = tuple(
            DenseArrayBinding(name, value, {"kind": "memory"})
            for name, value in (inputs or {}).items()
        )
    specs = {row["id"]: row for row in graph.get("inputs", [])}
    by_name: dict[str, DenseArrayBinding] = {}
    for binding in bindings:
        if binding.input_id in by_name:
            raise ValueError(f"duplicate dense binding for input {binding.input_id}")
        by_name[binding.input_id] = binding
    if set(by_name) != set(specs):
        missing = sorted(set(specs) - set(by_name))
        unexpected = sorted(set(by_name) - set(specs))
        raise ValueError(
            f"dense input ids do not match graph inputs; missing={missing}, unexpected={unexpected}"
        )

    resolved: dict[str, torch.Tensor] = {}
    rows: list[dict[str, Any]] = []
    leading_shape: tuple[int, int] | None = None
    masks: list[str] = []
    for input_id in sorted(specs):
        spec = specs[input_id]
        binding = by_name[input_id]
        value = binding.value
        declared = spec.get("shape", [])
        if len(declared) < 2 or declared[:2] != ["time", "batch"]:
            raise ValueError(
                f"input {input_id} dense binding requires declared shape beginning with ['time', 'batch']"
            )
        if value.ndim == len(declared) - 1 and declared[1] == "batch":
            value = value.unsqueeze(1)
        if value.ndim != len(declared):
            raise ValueError(
                f"input {input_id} rank expected {len(declared)}, got {value.ndim}"
            )
        expected_tail = tuple(int(axis) for axis in declared[2:])
        if tuple(value.shape[2:]) != expected_tail:
            raise ValueError(
                f"input {input_id} trailing shape expected {expected_tail}, got {tuple(value.shape[2:])}"
            )
        current_leading = (int(value.shape[0]), int(value.shape[1]))
        if leading_shape is None:
            leading_shape = current_leading
        elif current_leading != leading_shape:
            raise ValueError(
                f"input {input_id} leading shape expected {leading_shape}, got {current_leading}"
            )
        signal_type = spec.get("signal_type")
        if signal_type == "mask":
            if value.dtype != torch.bool:
                if not torch.all((value == 0) | (value == 1)):
                    raise ValueError(
                        f"input {input_id} mask values must be boolean or zero/one"
                    )
                value = value.bool()
            masks.append(input_id)
        elif not (value.is_floating_point() or value.dtype == torch.bool):
            value = value.float()
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"input {input_id} contains non-finite values")
        if signal_type == "spikes" and not torch.all((value == 0) | (value == 1)):
            raise ValueError(
                f"input {input_id} spike values must be boolean or zero/one"
            )
        if signal_type != "mask":
            value = value.float()
        value = value.to(device)
        resolved[input_id] = value
        rows.append(
            {
                "input_id": input_id,
                "representation": "dense_array",
                "shape": list(value.shape),
                "dtype": str(value.dtype).removeprefix("torch."),
                "signal_type": signal_type,
                "unit": spec.get("unit"),
                "source": dict(binding.source),
            }
        )
    assert leading_shape is not None
    dt_ms = float(graph["timebase"]["dt"]["value"])
    supplied = dict(protocol or {})
    dataset = dict(supplied.pop("dataset", {}))
    reserved = {
        "schema",
        "binding_schema",
        "representation",
        "inputs",
        "timing",
        "masks",
        "seeds",
    }
    if reserved & supplied.keys():
        raise ValueError(
            f"execution protocol cannot override reserved fields {sorted(reserved & supplied.keys())}"
        )
    dataset.setdefault("identity", None)
    dataset.setdefault("split", None)
    dataset.setdefault("sample_cap", leading_shape[1])
    dataset.setdefault("batch_size", leading_shape[1])
    dataset.setdefault("shuffle", None)
    execution_protocol = {
        "schema": EXECUTION_PROTOCOL_SCHEMA,
        "binding_schema": DENSE_ARRAY_BINDING_SCHEMA,
        "representation": "dense_array",
        "inputs": rows,
        "dataset": dataset,
        "timing": {
            "dt_ms": dt_ms,
            "steps": leading_shape[0],
            "duration_ms": leading_shape[0] * dt_ms,
        },
        "masks": masks,
        "seeds": {"execution": int(seed)},
        **supplied,
    }
    try:
        json.dumps(execution_protocol, sort_keys=True)
    except TypeError as exc:
        raise ValueError(
            f"execution protocol must be JSON-serializable: {exc}"
        ) from exc
    return ResolvedDenseInputs(tensors=resolved, protocol=execution_protocol)


def resolve_event_stream_bindings(
    graph: Mapping[str, Any],
    *,
    bindings: Sequence[EventStreamBinding],
    device: str | torch.device = "cpu",
    seed: int = 0,
    protocol: Mapping[str, Any] | None = None,
) -> ResolvedDenseInputs:
    """Validate sparse spike coordinates and materialize binary graph inputs."""
    specs = {row["id"]: row for row in graph.get("inputs", [])}
    by_name: dict[str, EventStreamBinding] = {}
    for binding in bindings:
        if binding.input_id in by_name:
            raise ValueError(f"duplicate event binding for input {binding.input_id}")
        by_name[binding.input_id] = binding
    if set(by_name) != set(specs):
        missing = sorted(set(specs) - set(by_name))
        unexpected = sorted(set(by_name) - set(specs))
        raise ValueError(
            f"event input ids do not match graph inputs; missing={missing}, unexpected={unexpected}"
        )

    resolved: dict[str, torch.Tensor] = {}
    rows: list[dict[str, Any]] = []
    leading_shape: tuple[int, int] | None = None
    integer_dtypes = {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }
    for input_id in sorted(specs):
        spec = specs[input_id]
        binding = by_name[input_id]
        declared = spec.get("shape", [])
        if declared[:2] != ["time", "batch"] or len(declared) != 3:
            raise ValueError(
                f"input {input_id} event binding requires shape ['time', 'batch', channels]"
            )
        if spec.get("signal_type") != "spikes":
            raise ValueError(
                f"input {input_id} event binding requires signal_type spikes"
            )
        channels_count = int(declared[2])
        if (
            not isinstance(binding.steps_count, Integral)
            or isinstance(binding.steps_count, bool)
            or not isinstance(binding.batch_size, Integral)
            or isinstance(binding.batch_size, bool)
        ):
            raise ValueError(
                f"event input {input_id} steps_count and batch_size must be integers"
            )
        steps_count = int(binding.steps_count)
        batch_size = int(binding.batch_size)
        if steps_count <= 0 or batch_size <= 0:
            raise ValueError(
                f"event input {input_id} steps_count and batch_size must be positive"
            )
        coordinates = (binding.steps, binding.batches, binding.channels)
        if any(value.ndim != 1 for value in coordinates):
            raise ValueError(
                f"event input {input_id} coordinates must be one-dimensional"
            )
        lengths = {int(value.numel()) for value in coordinates}
        if len(lengths) != 1:
            raise ValueError(f"event input {input_id} coordinate lengths must match")
        if any(value.dtype not in integer_dtypes for value in coordinates):
            raise ValueError(
                f"event input {input_id} coordinates must use integer dtypes"
            )
        steps = binding.steps.to(dtype=torch.int64, device="cpu")
        batches = binding.batches.to(dtype=torch.int64, device="cpu")
        channels = binding.channels.to(dtype=torch.int64, device="cpu")
        bounds = (
            ("step", steps, steps_count),
            ("batch", batches, batch_size),
            ("channel", channels, channels_count),
        )
        for label, values, upper in bounds:
            if torch.any(values < 0) or torch.any(values >= upper):
                raise ValueError(
                    f"event input {input_id} {label} coordinates must be in [0, {upper})"
                )
        flat = (steps * batch_size + batches) * channels_count + channels
        if flat.numel() > 1:
            differences = flat[1:] - flat[:-1]
            if torch.any(differences < 0):
                raise ValueError(
                    f"event input {input_id} coordinates must be ordered by step, batch, channel"
                )
            if torch.any(differences == 0):
                raise ValueError(
                    f"event input {input_id} contains duplicate coordinates"
                )
        current_leading = (steps_count, batch_size)
        if leading_shape is None:
            leading_shape = current_leading
        elif current_leading != leading_shape:
            raise ValueError(
                f"event input {input_id} leading shape expected {leading_shape}, got {current_leading}"
            )
        value = torch.zeros(
            (steps_count, batch_size, channels_count),
            dtype=torch.float32,
            device=device,
        )
        if flat.numel():
            value[
                steps.to(device=device),
                batches.to(device=device),
                channels.to(device=device),
            ] = 1.0
        resolved[input_id] = value
        rows.append(
            {
                "input_id": input_id,
                "representation": "event_stream",
                "shape": list(value.shape),
                "dtype": "float32",
                "signal_type": "spikes",
                "unit": spec.get("unit"),
                "event_count": int(flat.numel()),
                "source": dict(binding.source),
            }
        )
    if leading_shape is None:
        raise ValueError("graph execution requires at least one event input binding")
    dt_ms = float(graph["timebase"]["dt"]["value"])
    supplied = dict(protocol or {})
    dataset = dict(supplied.pop("dataset", {}))
    reserved = {
        "schema",
        "binding_schema",
        "representation",
        "inputs",
        "dataset",
        "timing",
        "masks",
        "seeds",
        "resolution",
    }
    if reserved & supplied.keys():
        raise ValueError(
            f"execution protocol cannot override reserved fields {sorted(reserved & supplied.keys())}"
        )
    dataset.setdefault("identity", None)
    dataset.setdefault("split", None)
    dataset.setdefault("sample_cap", leading_shape[1])
    dataset.setdefault("batch_size", leading_shape[1])
    dataset.setdefault("shuffle", None)
    execution_protocol = {
        "schema": EXECUTION_PROTOCOL_SCHEMA,
        "binding_schema": EVENT_STREAM_BINDING_SCHEMA,
        "representation": "event_stream",
        "inputs": rows,
        "dataset": dataset,
        "timing": {
            "dt_ms": dt_ms,
            "steps": leading_shape[0],
            "duration_ms": leading_shape[0] * dt_ms,
        },
        "masks": [],
        "seeds": {"execution": int(seed)},
        "resolution": {
            "coordinates": "zero_based_integer_steps",
            "ordering": "step,batch,channel",
            "duplicates": "reject",
            "materialization": "binary_dense",
        },
        **supplied,
    }
    try:
        json.dumps(execution_protocol, sort_keys=True)
    except TypeError as exc:
        raise ValueError(
            f"execution protocol must be JSON-serializable: {exc}"
        ) from exc
    return ResolvedDenseInputs(tensors=resolved, protocol=execution_protocol)


def resolve_input_bindings(
    graph: Mapping[str, Any],
    *,
    dense_bindings: Sequence[DenseArrayBinding] = (),
    event_bindings: Sequence[EventStreamBinding] = (),
    poisson_bindings: Sequence[PoissonInputBinding] = (),
    inputs: Mapping[str, torch.Tensor] | None = None,
    device: str | torch.device = "cpu",
    seed: int = 0,
    protocol: Mapping[str, Any] | None = None,
) -> ResolvedDenseInputs:
    """Resolve dense, event-stream, generated-Poisson, or mixed graph inputs."""
    if dense_bindings and inputs:
        raise ValueError("provide dense input bindings or input tensors, not both")
    if inputs:
        dense_bindings = tuple(
            DenseArrayBinding(name, value, {"kind": "memory"})
            for name, value in inputs.items()
        )
    dense_ids = {binding.input_id for binding in dense_bindings}
    event_ids = {binding.input_id for binding in event_bindings}
    poisson_ids = {binding.input_id for binding in poisson_bindings}
    overlap = sorted(
        (dense_ids & event_ids) | (dense_ids & poisson_ids) | (event_ids & poisson_ids)
    )
    if overlap:
        raise ValueError(
            f"graph inputs cannot have dense and event bindings: {overlap}"
        )
    graph_ids = {row["id"] for row in graph.get("inputs", [])}
    if dense_ids | event_ids | poisson_ids != graph_ids:
        missing = sorted(graph_ids - dense_ids - event_ids - poisson_ids)
        unexpected = sorted((dense_ids | event_ids | poisson_ids) - graph_ids)
        raise ValueError(
            f"input ids do not match graph inputs; missing={missing}, unexpected={unexpected}"
        )
    if poisson_bindings:
        if dense_bindings or event_bindings:
            raise ValueError(
                "Poisson bindings cannot yet be mixed with replay bindings"
            )
        return resolve_poisson_input_bindings(
            graph,
            bindings=poisson_bindings,
            device=device,
            seed=seed,
            protocol=protocol,
        )
    if not event_bindings:
        return resolve_dense_array_bindings(
            graph,
            bindings=dense_bindings,
            device=device,
            seed=seed,
            protocol=protocol,
        )
    if not dense_bindings:
        return resolve_event_stream_bindings(
            graph,
            bindings=event_bindings,
            device=device,
            seed=seed,
            protocol=protocol,
        )

    def graph_with_inputs(input_ids: set[str]) -> dict[str, Any]:
        return {
            **graph,
            "inputs": [
                row for row in graph.get("inputs", []) if row["id"] in input_ids
            ],
        }

    dense = resolve_dense_array_bindings(
        graph_with_inputs(dense_ids),
        bindings=dense_bindings,
        device=device,
        seed=seed,
        protocol=protocol,
    )
    events = resolve_event_stream_bindings(
        graph_with_inputs(event_ids),
        bindings=event_bindings,
        device=device,
        seed=seed,
        protocol=protocol,
    )
    if dense.protocol["timing"] != events.protocol["timing"]:
        raise ValueError(
            "dense and event input bindings must resolve to the same timestep, duration, and batch shape"
        )
    execution_protocol = {
        "schema": EXECUTION_PROTOCOL_SCHEMA,
        "binding_schema": MIXED_INPUT_BINDING_SCHEMA,
        "representation": "mixed",
        "inputs": sorted(
            [*dense.protocol["inputs"], *events.protocol["inputs"]],
            key=lambda row: row["input_id"],
        ),
        "dataset": dense.protocol["dataset"],
        "timing": dense.protocol["timing"],
        "masks": dense.protocol["masks"],
        "seeds": dense.protocol["seeds"],
        "resolution": {"event_stream": events.protocol["resolution"]},
        **{
            key: value
            for key, value in dense.protocol.items()
            if key
            not in {
                "schema",
                "binding_schema",
                "representation",
                "inputs",
                "dataset",
                "timing",
                "masks",
                "seeds",
                "resolution",
            }
        },
    }
    return ResolvedDenseInputs(
        tensors={**dense.tensors, **events.tensors}, protocol=execution_protocol
    )


def resolve_poisson_input_bindings(
    graph: Mapping[str, Any],
    *,
    bindings: Sequence[PoissonInputBinding],
    device: str | torch.device = "cpu",
    seed: int = 0,
    protocol: Mapping[str, Any] | None = None,
) -> ResolvedDenseInputs:
    """Generate reproducible fixed or per-presentation categorical Poisson spikes."""
    specs = {row["id"]: row for row in graph.get("inputs", [])}
    by_name = {binding.input_id: binding for binding in bindings}
    if len(by_name) != len(bindings):
        raise ValueError("duplicate Poisson binding for a graph input")
    if set(by_name) != set(specs):
        missing = sorted(set(specs) - set(by_name))
        unexpected = sorted(set(by_name) - set(specs))
        raise ValueError(
            f"Poisson input ids do not match graph inputs; missing={missing}, unexpected={unexpected}"
        )
    dt_ms = float(graph["timebase"]["dt"]["value"])
    tensors: dict[str, torch.Tensor] = {}
    rows: list[dict[str, Any]] = []
    leading_shape: tuple[int, int] | None = None
    for input_id in sorted(specs):
        spec = specs[input_id]
        binding = by_name[input_id]
        declared = spec.get("shape", [])
        if declared[:2] != ["time", "batch"] or len(declared) != 3:
            raise ValueError(
                f"input {input_id} Poisson binding requires shape ['time', 'batch', channels]"
            )
        if spec.get("signal_type") != "spikes":
            raise ValueError(
                f"input {input_id} Poisson binding requires signal_type spikes"
            )
        if binding.steps_count <= 0 or binding.batch_size <= 0:
            raise ValueError(
                f"Poisson input {input_id} steps_count and batch_size must be positive"
            )
        rates = tuple(float(rate) for rate in binding.rates_hz)
        if not rates or any(not math.isfinite(rate) or rate < 0 for rate in rates):
            raise ValueError(
                f"Poisson input {input_id} rates must be finite and non-negative"
            )
        if not binding.categorical and len(rates) != 1:
            raise ValueError(
                f"fixed-rate Poisson input {input_id} requires exactly one rate"
            )
        if max(rates) * dt_ms / 1000.0 > 1.0:
            raise ValueError(
                f"Poisson input {input_id} rate times dt exceeds probability one"
            )
        current_leading = (int(binding.steps_count), int(binding.batch_size))
        if leading_shape is None:
            leading_shape = current_leading
        elif current_leading != leading_shape:
            raise ValueError(
                f"Poisson input {input_id} leading shape expected {leading_shape}, got {current_leading}"
            )
        generator = torch.Generator(device="cpu").manual_seed(int(binding.seed))
        if binding.categorical:
            indices = torch.randint(
                len(rates), (binding.batch_size,), generator=generator
            )
            realized = torch.tensor(rates, dtype=torch.float32)[indices]
        else:
            realized = torch.full((binding.batch_size,), rates[0], dtype=torch.float32)
        probability = realized.reshape(1, -1, 1) * dt_ms / 1000.0
        value = (
            (
                torch.rand(
                    binding.steps_count,
                    binding.batch_size,
                    int(declared[2]),
                    generator=generator,
                )
                < probability
            )
            .float()
            .to(device)
        )
        tensors[input_id] = value
        rows.append(
            {
                "input_id": input_id,
                "representation": "poisson",
                "shape": list(value.shape),
                "dtype": "float32",
                "signal_type": "spikes",
                "unit": spec.get("unit"),
                "protocol": "categorical_rate" if binding.categorical else "fixed_rate",
                "rates_hz": list(rates),
                "realized_rates_hz": realized.tolist(),
                "seed": int(binding.seed),
                "selection": "uniform_independent_per_presentation"
                if binding.categorical
                else "constant",
            }
        )
    assert leading_shape is not None
    supplied = dict(protocol or {})
    dataset = dict(supplied.pop("dataset", {}))
    if {
        "schema",
        "binding_schema",
        "representation",
        "inputs",
        "timing",
        "seeds",
    } & supplied.keys():
        raise ValueError("execution protocol cannot override reserved Poisson fields")
    dataset.setdefault("identity", None)
    dataset.setdefault("split", None)
    dataset.setdefault("sample_cap", leading_shape[1])
    dataset.setdefault("batch_size", leading_shape[1])
    dataset.setdefault("shuffle", None)
    execution_protocol = {
        "schema": EXECUTION_PROTOCOL_SCHEMA,
        "binding_schema": POISSON_INPUT_BINDING_SCHEMA,
        "representation": "poisson",
        "inputs": rows,
        "dataset": dataset,
        "timing": {
            "dt_ms": dt_ms,
            "steps": leading_shape[0],
            "duration_ms": leading_shape[0] * dt_ms,
        },
        "masks": [],
        "seeds": {
            "execution": int(seed),
            "poisson": {row["input_id"]: row["seed"] for row in rows},
        },
        "resolution": {
            "distribution": "Bernoulli discretization of homogeneous Poisson",
            "rate_selection": "per_presentation",
        },
        **supplied,
    }
    json.dumps(execution_protocol, sort_keys=True)
    return ResolvedDenseInputs(tensors=tensors, protocol=execution_protocol)


def graph_capability_issues(graph: Mapping[str, Any]) -> list[CapabilityIssue]:
    """Return precise graph-executor capability failures."""
    issues: list[CapabilityIssue] = []
    neuron_capabilities: set[str] = {"coba_lif", "leaky_integrator"}
    synapse_capabilities: set[str] = {"ampa", "gaba", "leaky_integrator"}
    operation_capabilities: set[str] = {
        "linear",
        "reduce_mean",
        "reduce_sum",
        "select_final",
        "duration_normalise",
        "cumulative_sum",
    }
    connection_capabilities: set[str] = {"feedforward", "recurrent", "feedback"}
    for pop in graph.get("populations", []):
        kind = pop.get("neuron", {}).get("kind")
        if kind not in neuron_capabilities:
            issues.append(
                CapabilityIssue(pop["id"], f"neuron:{kind}", "unsupported neuron kind")
            )
    for projection in graph.get("projections", []):
        synapse = projection.get("synapse", {}).get("kind")
        if synapse not in synapse_capabilities:
            issues.append(
                CapabilityIssue(
                    projection["id"], f"synapse:{synapse}", "unsupported synapse kind"
                )
            )
        connection = projection.get("connection")
        if connection not in connection_capabilities:
            issues.append(
                CapabilityIssue(
                    projection["id"],
                    f"connection:{connection}",
                    "unsupported connection kind",
                )
            )
    for operation in graph.get("operations", []):
        kind = operation.get("kind")
        if kind not in operation_capabilities:
            issues.append(
                CapabilityIssue(
                    operation["id"], f"operation:{kind}", "unsupported operation kind"
                )
            )
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
    enabled: bool


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
            return {
                name: value.detach().to(device).clone()
                for name, value in values.items()
            }

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
            {
                "id": row["id"],
                "shape": row["shape"],
                "signal_type": row.get("signal_type"),
            }
            for row in plan.graph.get("inputs", [])
        ],
        "projections": [
            {
                "id": row.id,
                "source": row.source,
                "target": row.target,
                "polarity": row.polarity,
                "synapse": next(
                    item["synapse"]
                    for item in plan.graph.get("projections", [])
                    if item["id"] == row.id
                ),
                "delay_steps": row.delay_steps,
                "parameter": row.parameter,
                "parameter_shape": parameters[row.parameter]["shape"],
                "enabled": row.enabled,
            }
            for row in plan.projections
        ],
    }


def runtime_state_signature(plan: GraphPlan) -> str:
    payload = runtime_state_compatibility(plan)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _compatibility_mismatch(
    expected: Any, actual: Any, path: str = "graph"
) -> str | None:
    if type(expected) is not type(actual):
        return f"{path} expected {expected!r}, got {actual!r}"
    if isinstance(expected, dict):
        if set(expected) != set(actual):
            return f"{path} keys expected {sorted(expected)}, got {sorted(actual)}"
        for key in expected:
            mismatch = _compatibility_mismatch(
                expected[key], actual[key], f"{path}.{key}"
            )
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
            tensors.append(
                {
                    "group": group_name,
                    "name": name,
                    "key": key,
                    "shape": list(value.shape),
                    "dtype": str(value.dtype).removeprefix("torch."),
                }
            )
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
    if (
        manifest.get("schema") != RUNTIME_STATE_SCHEMA
        or manifest.get("schema_version") != 1
    ):
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
            groups[row["group"]][row["name"]] = torch.from_numpy(array.copy()).to(
                device
            )
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
        detail = "; ".join(
            f"{x.element} requires {x.capability}: {x.message}" for x in issues
        )
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
            raise ValueError(
                f"{row['id']}: delay {delay_ms} ms is not an integer number of dt={dt} ms steps"
            )
        source_owner = row["source"].partition(".")[0]
        # Recurrent/feedback edges are causal even when the author declares
        # zero additional delay. Feedforward population edges may consume the
        # current-step source after topological scheduling.
        if (
            source_owner in {p["id"] for p in graph.get("populations", [])}
            and row.get("connection") != "feedforward"
        ):
            steps = max(1, steps)
        tau = float(row["synapse"]["tau"]["value"])
        decay = (
            0.0 if row["synapse"]["kind"] == "leaky_integrator" else math.exp(-dt / tau)
        )
        planned.append(
            PlannedProjection(
                id=row["id"],
                source=row["source"],
                target=row["target"],
                polarity=row["polarity"],
                decay=decay,
                delay_steps=steps,
                parameter=row["parameters"][0],
                enabled=row.get("enabled", True),
            )
        )
    populations = list(graph.get("populations", []))
    population_ids = {p["id"] for p in populations}
    zero_edges = [
        (p.source.partition(".")[0], p.target.partition(".")[0])
        for p in planned
        if p.enabled
        and p.delay_steps == 0
        and p.source.partition(".")[0] in population_ids
    ]
    ordered: list[Mapping[str, Any]] = []
    remaining = {p["id"]: p for p in populations}
    while remaining:
        ready = sorted(
            name
            for name in remaining
            if not any(dst == name and src in remaining for src, dst in zero_edges)
        )
        if not ready:
            raise ValueError(
                "zero-delay population projections form an algebraic cycle"
            )
        for name in ready:
            ordered.append(remaining.pop(name))
    return GraphPlan(
        graph=graph,
        dt_ms=dt,
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

        def initialise(
            row: Mapping[str, Any],
            *,
            runtime_shape: tuple[int, ...],
            scale_by_fanin: bool,
        ) -> torch.Tensor:
            init = row["initializer"]
            if init["kind"] == "normal":
                value = (
                    torch.randn(*runtime_shape)
                    .mul_(float(init["std"]))
                    .add_(float(init["mean"]))
                    .clamp_(min=0)
                )
            elif init["kind"] == "constant":
                value = torch.full(runtime_shape, float(init["value"]))
            else:
                raise ValueError(f"{row['id']}: unsupported initializer {init['kind']}")
            if scale_by_fanin:
                value = value / runtime_shape[0]
            return value

        def init_priority(projection: PlannedProjection) -> tuple[int, str]:
            source = projection.source.partition(".")[0]
            target = projection.target.partition(".")[0]
            target_kind = next(
                p["neuron"]["kind"] for p in plan.populations if p["id"] == target
            )
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
            shape = tuple(reversed(row["shape"]))  # runtime is [source, target]
            realised[projection.parameter] = initialise(
                row, runtime_shape=shape, scale_by_fanin=True
            )
        for operation in plan.graph.get("operations", []):
            if operation.get("kind") != "linear":
                continue
            for parameter in operation.get("parameters", []):
                if parameter in realised:
                    continue
                row = rows[parameter]
                shape = tuple(reversed(row["shape"]))  # runtime is [source, target]
                realised[parameter] = initialise(
                    row, runtime_shape=shape, scale_by_fanin=False
                )
        for projection in plan.projections:
            self.weights[projection.parameter.replace(".", "__")] = nn.Parameter(
                realised[projection.parameter], requires_grad=False
            )
        for operation in plan.graph.get("operations", []):
            for parameter in operation.get("parameters", []):
                self.weights[parameter.replace(".", "__")] = nn.Parameter(
                    realised[parameter], requires_grad=False
                )

    def parameter_map(self) -> dict[str, torch.Tensor]:
        return {name.replace("__", "."): value for name, value in self.weights.items()}

    def forward(
        self,
        inputs: Mapping[str, torch.Tensor],
        *,
        record: bool | RecordingProfile = True,
        runtime_state: GraphRuntimeState | None = None,
    ) -> ExecutionResult:
        recording: RecordingProfile = (
            "full" if record is True else "none" if record is False else record
        )
        if recording not in {"full", "observables", "none"}:
            raise ValueError(
                f"recording profile expected full, observables, or none; got {recording!r}"
            )
        if not inputs:
            raise ValueError("graph execution requires at least one input tensor")
        first = next(iter(inputs.values()))
        steps, batch = first.shape[:2]
        device = first.device
        parameter_dtype = (
            next(iter(self.weights.values())).dtype if self.weights else first.dtype
        )
        input_specs = {row["id"]: row for row in self.plan.graph.get("inputs", [])}
        for name, value in inputs.items():
            if value.shape[:2] != (steps, batch):
                raise ValueError(
                    f"input {name} leading shape expected {(steps, batch)}, got {tuple(value.shape[:2])}"
                )
            if value.device != device:
                raise ValueError(
                    f"input {name} device expected {device}, got {value.device}"
                )
            is_mask = input_specs.get(name, {}).get("signal_type") == "mask"
            if value.dtype != parameter_dtype and not (
                is_mask and value.dtype == torch.bool
            ):
                raise ValueError(
                    f"input {name} dtype expected {parameter_dtype}, got {value.dtype}"
                )
        populations = {p["id"]: p for p in self.plan.populations}
        population_history_lengths = {
            name: max(
                (
                    p.delay_steps
                    for p in self.plan.projections
                    if p.source.startswith(name + ".")
                ),
                default=1,
            )
            for name in populations
        }
        input_history_lengths = {
            row["id"]: max(
                (
                    p.delay_steps
                    for p in self.plan.projections
                    if p.source.partition(".")[0] == row["id"]
                ),
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
            spikes = {
                name: torch.zeros((batch, p["size"]), device=device)
                for name, p in populations.items()
            }
            conductance = {
                (p.id, p.polarity): torch.zeros(
                    (batch, populations[p.target.partition(".")[0]]["size"]),
                    device=device,
                )
                for p in self.plan.projections
            }
            histories = {
                name: DelayBuffer(population_history_lengths[name], value)
                for name, value in spikes.items()
            }
            input_histories = {
                name: torch.zeros(
                    (length, *inputs[name].shape[1:]),
                    device=device,
                    dtype=inputs[name].dtype,
                )
                for name, length in input_history_lengths.items()
                if length > 0
            }
            completed_steps = 0
        else:
            if runtime_state.signature != expected_signature:
                detail = _compatibility_mismatch(
                    expected_compatibility, runtime_state.compatibility
                )
                raise ValueError(
                    "runtime state is incompatible with graph plan: "
                    + (
                        detail
                        or f"signature expected {expected_signature}, got {runtime_state.signature}"
                    )
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

            pop_shapes = {
                name: (batch, int(row["size"])) for name, row in populations.items()
            }
            voltage = restore_group("voltages", runtime_state.voltages, pop_shapes)
            for name, value in voltage.items():
                if value.dtype != parameter_dtype:
                    raise ValueError(
                        f"runtime state voltages.{name} dtype expected {parameter_dtype}, got {value.dtype}"
                    )
            refractory = restore_group(
                "refractory", runtime_state.refractory, pop_shapes
            )
            for name, value in refractory.items():
                if value.dtype != torch.long:
                    raise ValueError(
                        f"runtime state refractory.{name} dtype expected torch.int64, got {value.dtype}"
                    )
            conductance_by_id = restore_group(
                "conductances",
                runtime_state.conductances,
                {
                    p.id: (batch, int(populations[p.target.partition(".")[0]]["size"]))
                    for p in self.plan.projections
                },
            )
            conductance = {
                (p.id, p.polarity): conductance_by_id[p.id]
                for p in self.plan.projections
            }
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
                name: DelayBuffer.restore(value)
                for name, value in population_history_values.items()
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
                    for name, length in input_history_lengths.items()
                    if length > 0
                },
            )
            for name, value in input_histories.items():
                if value.dtype != inputs[name].dtype:
                    raise ValueError(
                        f"runtime state input_histories.{name} dtype expected {inputs[name].dtype}, got {value.dtype}"
                    )
            completed_steps = int(runtime_state.completed_steps)
        recordings: dict[str, list[torch.Tensor]] = {
            o["id"]: [] for o in self.plan.observables
        }
        state_recordings: dict[str, list[torch.Tensor]] = {
            f"{name}.voltage": [] for name in populations
        }
        projection_recordings: dict[str, list[torch.Tensor]] = {
            f"{p.id}.conductance": [] for p in self.plan.projections
        }
        integrator_sum: dict[str, torch.Tensor] = {}
        spike_traces: dict[str, list[torch.Tensor]] = {name: [] for name in populations}
        voltage_traces: dict[str, list[torch.Tensor]] = {
            name: [] for name in populations
        }

        for t in range(steps):
            new_spikes: dict[str, torch.Tensor] = {}
            for pop in self.plan.populations:
                name = pop["id"]
                incoming = {
                    "excitatory": torch.zeros_like(voltage[name]),
                    "inhibitory": torch.zeros_like(voltage[name]),
                }
                for projection in self.plan.projections:
                    if projection.target.partition(".")[0] != name:
                        continue
                    key = (projection.id, projection.polarity)
                    if not projection.enabled:
                        conductance[key].zero_()
                        continue
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
                    drive = (
                        source @ self.weights[projection.parameter.replace(".", "__")]
                    )
                    conductance[key] = conductance[key] * projection.decay + drive
                    incoming[projection.polarity] += conductance[key]
                neuron = pop["neuron"]
                if neuron["kind"] == "leaky_integrator":
                    beta = math.exp(-self.plan.dt_ms / float(neuron["tau"]["value"]))
                    voltage[name] = (
                        beta * voltage[name]
                        + (1.0 - beta) / self.plan.dt_ms * incoming["excitatory"]
                    )
                    new_spikes[name] = torch.zeros_like(spikes[name])
                    integrator_sum[name] = (
                        integrator_sum.get(name, torch.zeros_like(voltage[name]))
                        + voltage[name]
                    )
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
                ref_steps = int(
                    neuron.get(
                        "refractory_steps",
                        max(
                            1,
                            round(
                                (M.ref_ms_E if tau_mem >= 15 else M.ref_ms_I)
                                / self.plan.dt_ms
                            ),
                        ),
                    )
                )
                dampen = float(neuron.get("voltage_grad_dampen", M.V_GRAD_DAMPEN))
                voltage[name], new_spikes[name], refractory[name] = M.lif_step_expeuler(
                    voltage[name],
                    refractory[name],
                    incoming["excitatory"],
                    incoming["inhibitory"],
                    c_m,
                    g_l,
                    ref_steps,
                    M.spike_biophysical,
                    dt_override=self.plan.dt_ms,
                    v_grad_dampen=dampen,
                )
            spikes = new_spikes
            for name in populations:
                spike_traces[name].append(spikes[name])
                voltage_traces[name].append(voltage[name])
            for name in populations:
                histories[name].push(spikes[name])
            if recording != "none":
                for observable in self.plan.observables:
                    owner, _, port = observable["signal"].partition(".")
                    recordings[observable["id"]].append(
                        (spikes if port == "spikes" else voltage)[owner]
                        .detach()
                        .clone()
                    )
            if recording == "full":
                for name in populations:
                    state_recordings[f"{name}.voltage"].append(
                        voltage[name].detach().clone()
                    )
                for projection in self.plan.projections:
                    projection_recordings[f"{projection.id}.conductance"].append(
                        conductance[(projection.id, projection.polarity)]
                        .detach()
                        .clone()
                    )

        outputs: dict[str, torch.Tensor] = {}
        signal_values: dict[str, torch.Tensor] = {
            f"{name}.value": value for name, value in inputs.items()
        }
        for name in populations:
            signal_values[f"{name}.spikes"] = torch.stack(spike_traces[name])
            signal_values[f"{name}.voltage"] = torch.stack(voltage_traces[name])

        def time_mask(
            mask: torch.Tensor, *, target: torch.Tensor, op_id: str
        ) -> torch.Tensor:
            if mask.shape[:2] != target.shape[:2]:
                raise ValueError(
                    f"{op_id}: valid-time mask leading shape expected {tuple(target.shape[:2])}, got {tuple(mask.shape[:2])}"
                )
            if mask.ndim != 2:
                raise ValueError(
                    f"{op_id}: valid-time mask must have shape [time, batch]"
                )
            mask_value = mask.to(device=target.device, dtype=target.dtype)
            return mask_value.reshape(
                mask_value.shape[0], mask_value.shape[1], *([1] * (target.ndim - 2))
            )

        def reduce_time(
            source: torch.Tensor, *, kind: str, mask: torch.Tensor | None, op_id: str
        ) -> torch.Tensor:
            if mask is None:
                return source.sum(dim=0) if kind == "reduce_sum" else source.mean(dim=0)
            weights = time_mask(mask, target=source, op_id=op_id)
            numerator = (source * weights).sum(dim=0)
            if kind == "reduce_sum":
                return numerator
            counts = weights.sum(dim=0)
            if torch.any(counts <= 0):
                raise ValueError(
                    f"{op_id}: valid-time mask contains an empty reduction window"
                )
            return numerator / counts

        remaining_ops = list(self.plan.graph.get("operations", []))
        while remaining_ops:
            ready_index = next(
                (
                    index
                    for index, op in enumerate(remaining_ops)
                    if all(source in signal_values for source in op["sources"])
                ),
                None,
            )
            if ready_index is None:
                unresolved = {
                    op["id"]: [
                        source
                        for source in op["sources"]
                        if source not in signal_values
                    ]
                    for op in remaining_ops
                }
                raise ValueError(f"operation dependencies are unresolved: {unresolved}")
            op = remaining_ops.pop(ready_index)
            sources = [signal_values[source] for source in op["sources"]]
            kind = op["kind"]
            if kind == "linear":
                parameter = op["parameters"][0].replace(".", "__")
                signal_values[f"{op['id']}.value"] = (
                    sources[0] @ self.weights[parameter]
                )
            elif kind in {"reduce_mean", "reduce_sum"}:
                mask_name = op.get("config", {}).get("mask")
                mask = signal_values.get(mask_name) if mask_name else None
                source_id = op["sources"][0]
                owner, _, port = source_id.partition(".")
                if (
                    kind == "reduce_mean"
                    and mask is None
                    and port == "voltage"
                    and owner in integrator_sum
                ):
                    signal_values[f"{op['id']}.value"] = integrator_sum[owner] / steps
                else:
                    signal_values[f"{op['id']}.value"] = reduce_time(
                        sources[0], kind=kind, mask=mask, op_id=op["id"]
                    )
            elif kind == "select_final":
                signal_values[f"{op['id']}.value"] = sources[0][-1]
            elif kind == "duration_normalise":
                config = op.get("config", {})
                mask_name = config.get("mask")
                if mask_name:
                    mask = signal_values[mask_name]
                    if mask.ndim != 2:
                        raise ValueError(
                            f"{op['id']}: valid-time mask must have shape [time, batch]"
                        )
                    mask_seconds = mask.to(
                        device=sources[0].device, dtype=sources[0].dtype
                    ).sum(dim=0) * (self.plan.dt_ms / 1000.0)
                    mask_seconds = mask_seconds.reshape(
                        mask_seconds.shape[0], *([1] * (sources[0].ndim - 1))
                    )
                    if torch.any(mask_seconds <= 0):
                        raise ValueError(
                            f"{op['id']}: valid-time mask contains zero valid duration"
                        )
                    signal_values[f"{op['id']}.value"] = sources[0] / mask_seconds
                else:
                    duration_s = float(config["duration"])
                    if duration_s <= 0:
                        raise ValueError(
                            f"{op['id']}: spike-rate duration must be positive seconds"
                        )
                    signal_values[f"{op['id']}.value"] = sources[0] / duration_s
            elif kind == "cumulative_sum":
                signal_values[f"{op['id']}.value"] = sources[0].cumsum(dim=0)
            else:
                raise ValueError(f"{op['id']}: unsupported operation {kind}")
        for output in self.plan.outputs:
            outputs[output["id"]] = signal_values[output["signal"]]
        packed = {k: torch.stack(v) for k, v in recordings.items() if v}
        packed.update({k: torch.stack(v) for k, v in state_recordings.items() if v})
        packed.update(
            {k: torch.stack(v) for k, v in projection_recordings.items() if v}
        )
        next_input_histories = {
            name: torch.cat((history, inputs[name]), dim=0)[-history.shape[0] :]
            .detach()
            .clone()
            for name, history in input_histories.items()
        }
        next_runtime_state = GraphRuntimeState(
            signature=expected_signature,
            compatibility=expected_compatibility,
            completed_steps=completed_steps + steps,
            voltages={name: value.detach().clone() for name, value in voltage.items()},
            refractory={
                name: value.detach().clone() for name, value in refractory.items()
            },
            conductances={
                p.id: conductance[(p.id, p.polarity)].detach().clone()
                for p in self.plan.projections
            },
            population_histories={
                name: history.export() for name, history in histories.items()
            },
            input_histories=next_input_histories,
        )
        return ExecutionResult(
            executor="graph",
            outputs=outputs,
            recordings=packed,
            parameters={k: v.detach().clone() for k, v in self.parameter_map().items()},
            final_state={
                f"{k}.voltage": v.detach().clone() for k, v in voltage.items()
            },
            runtime_state=next_runtime_state,
            model=self,
        )


def build(spec: ExecutionSpec) -> ExecutionResult:
    if spec.executor == "legacy":
        return ExecutionResult(
            executor="legacy", metrics={"request": "build", "routing": "legacy"}
        )
    graph = spec.graph
    if graph is None and spec.bundle is not None:
        _, graph = load_graph_bundle(spec.bundle)
    if graph is None:
        raise ValueError("graph execution requires graph data or a bundle")
    device = resolve_device(spec.device)
    started = time.perf_counter()
    model = GraphExecutor(plan_graph(graph), seed=spec.seed).to(device)
    return ExecutionResult(
        executor="graph",
        model=model,
        parameters=model.parameter_map(),
        metrics={"build_s": time.perf_counter() - started},
    )


def simulate(
    spec: ExecutionSpec, *, runtime_state: GraphRuntimeState | None = None
) -> ExecutionResult:
    if spec.executor != "graph":
        return ExecutionResult(
            executor="legacy", metrics={"request": "simulate", "routing": "legacy"}
        )
    built = build(spec)
    assert isinstance(built.model, GraphExecutor)
    device = resolve_device(spec.device)
    if spec.checkpoint:
        built.model.load_state_dict(
            torch.load(spec.checkpoint, map_location=device, weights_only=True)
        )
    tracemalloc.start()
    started = time.perf_counter()
    resolved_inputs = resolve_input_bindings(
        built.model.plan.graph,
        dense_bindings=spec.input_bindings,
        event_bindings=spec.event_bindings,
        poisson_bindings=spec.poisson_bindings,
        inputs=spec.inputs,
        device=device,
        seed=spec.seed,
        protocol=spec.protocol,
    )
    result = built.model(
        resolved_inputs.tensors,
        record=spec.recording,
        runtime_state=runtime_state
        if runtime_state is not None
        else spec.runtime_state,
    )
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    result.metrics.update(
        {
            "simulate_s": elapsed,
            "peak_python_bytes": peak,
            "device": device,
            "recording": spec.recording,
            "execution_protocol": resolved_inputs.protocol,
            **built.metrics,
        }
    )
    if result.runtime_state is not None:
        result.metrics.update(
            {
                "runtime_state_schema": RUNTIME_STATE_SCHEMA,
                "runtime_state_signature": result.runtime_state.signature,
                "completed_steps": result.runtime_state.completed_steps,
            }
        )
    return result


def train(spec: ExecutionSpec) -> ExecutionResult:
    if spec.executor == "graph":
        raise NotImplementedError(
            "graph training requires capability training:v1 (Milestone 6)"
        )
    return ExecutionResult(
        executor="legacy", metrics={"request": "train", "routing": "legacy"}
    )


def infer(spec: ExecutionSpec) -> ExecutionResult:
    return (
        simulate(spec)
        if spec.executor == "graph"
        else ExecutionResult(
            executor="legacy", metrics={"request": "infer", "routing": "legacy"}
        )
    )


def execution_spec_from_args(
    args: Any, *, kind: RequestKind | None = None
) -> ExecutionSpec:
    """Compatibility adapter: resolved CLI arguments become one typed request."""
    resolved_kind = kind or ("infer" if getattr(args, "infer", False) else args.mode)
    if resolved_kind == "sim":
        resolved_kind = "simulate"
    return ExecutionSpec(
        kind=resolved_kind,
        executor=getattr(args, "executor", "legacy"),
        bundle=Path(args.bundle) if getattr(args, "bundle", None) else None,
        seed=int(getattr(args, "seed", 0) or 0),
        device=resolve_device(getattr(args, "device", "auto")),
        recording=getattr(args, "recording", "full"),
        checkpoint=(
            Path(args.load_weights) if getattr(args, "load_weights", None) else None
        ),
        options={
            key: value
            for key, value in vars(args).items()
            if key not in {"bundle", "executor"}
        },
    )


def resolve_device(requested: str | torch.device = "auto") -> str:
    """Resolve an explicit device or select the fastest available accelerator."""
    name = str(requested).lower()
    if name == "auto":
        forced = os.environ.get("PINGLAB_DEVICE")
        if forced:
            return resolve_device(forced)
        if torch.cuda.is_available():
            return "cuda"
        # Graph execution launches several small kernels from Python per timestep.
        # On the representative 800E/200I graph MPS is slower than CPU, so keep it
        # available explicitly without selecting it automatically.
        return "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but torch.cuda.is_available() is false")
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError(
            "MPS was requested but torch.backends.mps.is_available() is false"
        )
    if (
        name != "cpu"
        and name != "cuda"
        and name != "mps"
        and not name.startswith("cuda:")
    ):
        raise ValueError(
            f"device expected auto, cpu, cuda, cuda:N, or mps; got {requested!r}"
        )
    if name.startswith("cuda:") and not torch.cuda.is_available():
        raise ValueError(f"{name} was requested but torch.cuda.is_available() is false")
    return name


def execute_request(
    spec: ExecutionSpec,
    *,
    legacy: Callable[[], ExecutionResult] | None = None,
) -> ExecutionResult:
    """Dispatch one typed request; the CLI supplies its unchanged legacy body."""
    if spec.executor == "legacy":
        if legacy is None:
            raise ValueError(
                "legacy execution requires the registered legacy request body"
            )
        return legacy()
    handlers = {"build": build, "simulate": simulate, "train": train, "infer": infer}
    return handlers[spec.kind](spec)
