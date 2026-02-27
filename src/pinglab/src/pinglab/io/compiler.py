from __future__ import annotations

import argparse
import copy
import hashlib
import json
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any, Literal

import numpy as np
from rich.console import Console
from rich.table import Table

from pinglab.io.external_spike_train import external_spike_train
from pinglab.io.oscillating import oscillating
from pinglab.io.pulse import add_pulse_to_input, add_pulse_train_to_input
from pinglab.io.ramp import ramp
from pinglab.io.weights import (
    ResolvedWeightMatrices,
    resolve_weight_matrices_from_full,
    to_torch_weights,
)
from pinglab.service.service import DEFAULT_CONFIG, DEFAULT_WEIGHTS

ALLOWED_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "sim",
        "nodes",
        "edges",
        "inputs",
        "execution",
        "biophysics",
        "constraints",
        "meta",
    }
)
ALLOWED_SIM_KEYS = frozenset({"dt_ms", "T_ms", "seed", "neuron_model"})
ALLOWED_EXECUTION_KEYS = frozenset({"performance_mode", "max_spikes", "burn_in_ms"})
ALLOWED_CONSTRAINT_KEYS = frozenset({"nonnegative_weights", "nonnegative_input"})
ALLOWED_NODE_KEYS = frozenset({"id", "kind", "type", "size"})
ALLOWED_EDGE_KEYS = frozenset(
    {"id", "from", "to", "kind", "w", "tau_ms", "delay_ms", "clamp_min", "target", "seed", "enabled"}
)
ALLOWED_INPUT_SPEC_KEYS = frozenset(
    {
        "mode",
        "seed",
        "mean",
        "std",
        "noise_std",
        "noise_std_E",
        "noise_std_I",
        "I_E_base",
        "I_I_base",
        "I_E_start",
        "I_E_end",
        "I_I_start",
        "I_I_end",
        "input_population",
        "sine_freq_hz",
        "sine_amp",
        "sine_y_offset",
        "sine_phase",
        "sine_phase_offset_i",
        "lambda0_hz",
        "mod_depth",
        "envelope_freq_hz",
        "phase_rad",
        "w_in",
        "tau_in_ms",
        "pulse_t_ms",
        "pulse_width_ms",
        "pulse_interval_ms",
        "pulse_amp_E",
        "pulse_amp_I",
    }
)


@dataclass(frozen=True)
class RuntimeBundle:
    config: Any
    external_input: Any
    weights: ResolvedWeightMatrices
    model: Any
    backend: str = "pytorch"
    device: str | None = None

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


def _require_dict(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _require_list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array")
    return value


def _reject_unknown_keys(payload: dict[str, Any], *, name: str, allowed: frozenset[str]) -> None:
    extras = sorted(set(payload.keys()) - set(allowed))
    if extras:
        extras_csv = ", ".join(extras)
        raise ValueError(f"{name} has unknown field(s): {extras_csv}")


def compile_graph(spec: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Compile a pinglab graph spec to a normalized runtime plan.

    This is intentionally lightweight: it validates core shape, resolves
    population index ranges, and normalizes edge records for downstream
    matrix compilation and simulation wiring.
    """
    if spec is None:
        raise ValueError("spec is required")
    _reject_unknown_keys(spec, name="spec", allowed=ALLOWED_TOP_LEVEL_KEYS)

    schema_version = str(spec.get("schema_version", ""))
    if schema_version != "pinglab-graph.v1":
        raise ValueError("schema_version must be 'pinglab-graph.v1'")

    sim = _require_dict(spec.get("sim"), "sim")
    _reject_unknown_keys(sim, name="sim", allowed=ALLOWED_SIM_KEYS)
    nodes = _require_list(spec.get("nodes"), "nodes")
    edges = _require_list(spec.get("edges"), "edges")
    inputs = _require_dict(spec.get("inputs", {}), "inputs")
    execution = _require_dict(spec.get("execution", {}), "execution")
    _reject_unknown_keys(execution, name="execution", allowed=ALLOWED_EXECUTION_KEYS)
    biophysics = _require_dict(spec.get("biophysics", {}), "biophysics")
    allowed_biophysics_keys = frozenset(
        str(k)
        for k in DEFAULT_CONFIG.model_dump().keys()
        if str(k)
        not in {"dt", "T", "seed", "neuron_model", "N_E", "N_I", "delay_ee", "delay_ei", "delay_ie", "delay_ii", "instruments"}
    )
    _reject_unknown_keys(biophysics, name="biophysics", allowed=allowed_biophysics_keys)
    constraints = _require_dict(spec.get("constraints", {}), "constraints")
    _reject_unknown_keys(constraints, name="constraints", allowed=ALLOWED_CONSTRAINT_KEYS)
    meta = _require_dict(spec.get("meta", {}), "meta")

    node_by_id: dict[str, dict[str, Any]] = {}
    population_nodes: list[dict[str, Any]] = []
    input_nodes: list[dict[str, Any]] = []
    for raw_node in nodes:
        node = _require_dict(raw_node, "node")
        _reject_unknown_keys(node, name="node", allowed=ALLOWED_NODE_KEYS)
        node_id = str(node.get("id", ""))
        if not node_id:
            raise ValueError("node.id is required")
        if node_id in node_by_id:
            raise ValueError(f"duplicate node id: {node_id}")
        kind = str(node.get("kind", ""))
        if kind not in {"population", "input"}:
            raise ValueError(f"node '{node_id}' has invalid kind '{kind}'")
        size = int(node.get("size", 0))
        if kind == "population" and size <= 0:
            raise ValueError(f"population node '{node_id}' must have size > 0")
        if kind == "input" and size != 0:
            raise ValueError(f"input node '{node_id}' must have size = 0")
        node_by_id[node_id] = node
        if kind == "population":
            population_nodes.append(node)
        else:
            input_nodes.append(node)

    population_index: dict[str, dict[str, Any]] = {}
    cursor = 0
    n_e = 0
    n_i = 0
    for node in population_nodes:
        node_id = str(node["id"])
        size = int(node["size"])
        pop_type = str(node.get("type", ""))
        if pop_type == "E":
            n_e += size
        elif pop_type == "I":
            n_i += size
        population_index[node_id] = {
            "type": pop_type,
            "size": size,
            "start": cursor,
            "stop": cursor + size,
        }
        cursor += size

    normalized_edges: list[dict[str, Any]] = []
    for raw_edge in edges:
        edge = _require_dict(raw_edge, "edge")
        _reject_unknown_keys(edge, name="edge", allowed=ALLOWED_EDGE_KEYS)
        edge_id = str(edge.get("id", ""))
        if not edge_id:
            raise ValueError("edge.id is required")
        src_id = str(edge.get("from", ""))
        dst_id = str(edge.get("to", ""))
        if src_id not in node_by_id:
            raise ValueError(f"edge '{edge_id}' has unknown source node '{src_id}'")
        if dst_id not in node_by_id:
            raise ValueError(f"edge '{edge_id}' has unknown target node '{dst_id}'")
        src = node_by_id[src_id]
        dst = node_by_id[dst_id]
        enabled = bool(edge.get("enabled", True))
        normalized_edges.append(
            {
                "id": edge_id,
                "kind": str(edge.get("kind", "")),
                "enabled": enabled,
                "from": src_id,
                "to": dst_id,
                "from_kind": str(src.get("kind", "")),
                "to_kind": str(dst.get("kind", "")),
                "from_type": str(src.get("type", "")),
                "to_type": str(dst.get("type", "")),
                "from_range": population_index.get(src_id),
                "to_range": population_index.get(dst_id),
                "params": {
                    "w": edge.get("w"),
                    "tau_ms": edge.get("tau_ms"),
                    "delay_ms": edge.get("delay_ms"),
                    "clamp_min": edge.get("clamp_min"),
                    "target": edge.get("target"),
                    "seed": edge.get("seed"),
                },
            }
        )

    input_node_ids = {str(node["id"]) for node in input_nodes}
    for input_id, input_spec in inputs.items():
        if str(input_id) not in input_node_ids:
            raise ValueError(f"inputs has unknown input node '{input_id}'")
        input_spec_obj = _require_dict(input_spec, f"inputs.{input_id}")
        _reject_unknown_keys(
            input_spec_obj,
            name=f"inputs.{input_id}",
            allowed=ALLOWED_INPUT_SPEC_KEYS,
        )

    missing_input_specs = [str(node["id"]) for node in input_nodes if str(node["id"]) not in inputs]

    return {
        "schema_version": schema_version,
        "sim": sim,
        "execution": execution,
        "biophysics": biophysics,
        "constraints": constraints,
        "meta": meta,
        "totals": {
            "N_E": n_e,
            "N_I": n_i,
            "N_total": n_e + n_i,
        },
        "population_index": population_index,
        "input_nodes": [str(node["id"]) for node in input_nodes],
        "input_specs": inputs,
        "edges": normalized_edges,
        "warnings": {
            "missing_input_specs": missing_input_specs,
        },
    }


def _stable_seed(base_seed: int, key: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


def _sample_distribution(
    rng: np.random.RandomState,
    dist_params: dict[str, float],
    shape: tuple[int, ...],
) -> np.ndarray:
    mean = float(dist_params.get("mean", 0.0))
    std = float(dist_params.get("std", 1.0))
    return rng.normal(loc=mean, scale=std, size=shape)


def _resolve_input_edge_gains(
    params: dict[str, Any],
    *,
    count: int,
    seed: int,
    nonnegative_input: bool,
) -> np.ndarray:
    if count <= 0:
        return np.zeros(0, dtype=float)
    w = params.get("w")
    if isinstance(w, dict):
        dist_params = {
            str(k): float(v)
            for k, v in w.items()
        }
        gains = _sample_distribution(
            np.random.RandomState(seed),
            dist_params,
            (count,),
        ).astype(float, copy=False)
    else:
        gain = float(params.get("w", 1.0))
        gains = np.full(count, gain, dtype=float)
    if nonnegative_input:
        np.maximum(gains, 0.0, out=gains)
    return gains


def _resolve_target_indices(
    target: dict[str, Any] | None,
    pop_start: int,
    pop_stop: int,
    *,
    seed: int,
) -> np.ndarray:
    size = max(0, pop_stop - pop_start)
    pop_ids = np.arange(pop_start, pop_stop, dtype=int)
    if size == 0:
        return pop_ids
    if not isinstance(target, dict):
        return pop_ids

    mode = str(target.get("mode", "all"))
    if mode == "all":
        return pop_ids
    if mode == "indices":
        raw = target.get("indices")
        if not isinstance(raw, list):
            return np.array([], dtype=int)
        keep: list[int] = []
        for idx in raw:
            idx_int = int(idx)
            if 0 <= idx_int < size:
                keep.append(pop_start + idx_int)
        return np.array(sorted(set(keep)), dtype=int)
    if mode == "subset":
        fraction = float(target.get("fraction", 1.0))
        fraction = max(0.0, min(1.0, fraction))
        count = int(np.floor(fraction * size))
        if count <= 0:
            return np.array([], dtype=int)
        if count >= size:
            return pop_ids
        strategy = str(target.get("strategy", "random"))
        if strategy == "first":
            return pop_ids[:count]
        rng = np.random.RandomState(seed)
        picks = rng.choice(pop_ids, size=count, replace=False)
        return np.sort(picks.astype(int, copy=False))
    return pop_ids


def _build_population_order(plan: dict[str, Any]) -> tuple[list[dict[str, Any]], int, int]:
    pop_index: dict[str, dict[str, Any]] = plan["population_index"]
    e_entries: list[dict[str, Any]] = []
    i_entries: list[dict[str, Any]] = []
    for node_id, meta in pop_index.items():
        entry = {
            "id": node_id,
            "type": str(meta["type"]),
            "size": int(meta["size"]),
        }
        if entry["type"] == "E":
            e_entries.append(entry)
        elif entry["type"] == "I":
            i_entries.append(entry)
    ordered = e_entries + i_entries
    cursor = 0
    for entry in ordered:
        entry["start"] = cursor
        entry["stop"] = cursor + entry["size"]
        cursor += entry["size"]
    n_e = sum(int(entry["size"]) for entry in e_entries)
    n_i = sum(int(entry["size"]) for entry in i_entries)
    return ordered, n_e, n_i


def _build_external_input_for_node(
    *,
    mode: str,
    n_e: int,
    n_i: int,
    dt: float,
    num_steps: int,
    seed: int,
    input_spec: dict[str, Any],
) -> np.ndarray:
    mean = float(input_spec.get("mean", input_spec.get("I_E_base", 0.0)))
    noise_std = float(input_spec.get("std", input_spec.get("noise_std", 0.0)))
    noise_std_e = float(input_spec.get("noise_std_E", noise_std))
    noise_std_i = float(input_spec.get("noise_std_I", noise_std))
    i_e_base = float(input_spec.get("I_E_base", mean))
    i_i_base = float(input_spec.get("I_I_base", mean))

    if mode == "tonic":
        mode = "ramp"
        input_spec = {
            **input_spec,
            "I_E_start": i_e_base,
            "I_E_end": i_e_base,
            "I_I_start": i_i_base,
            "I_I_end": i_i_base,
        }

    if mode == "ramp":
        return ramp(
            N_E=n_e,
            N_I=n_i,
            I_E_start=float(input_spec.get("I_E_start", i_e_base)),
            I_E_end=float(input_spec.get("I_E_end", i_e_base)),
            I_I_start=float(input_spec.get("I_I_start", i_i_base)),
            I_I_end=float(input_spec.get("I_I_end", i_i_base)),
            noise_std_E=noise_std_e,
            noise_std_I=noise_std_i,
            num_steps=num_steps,
            dt=dt,
            seed=seed,
        )

    if mode == "sine":
        return oscillating(
            N_E=n_e,
            N_I=n_i,
            I_E=i_e_base + float(input_spec.get("sine_y_offset", 0.0)),
            I_I=i_i_base + float(input_spec.get("sine_y_offset", 0.0)),
            noise_std=float(max(noise_std_e, noise_std_i)),
            num_steps=num_steps,
            dt=dt,
            seed=seed,
            oscillation_freq=float(input_spec.get("sine_freq_hz", 5.0)),
            oscillation_amplitude=float(input_spec.get("sine_amp", 2.0)),
            oscillation_phase=float(input_spec.get("sine_phase", 0.0)),
            phase_offset_I=float(input_spec.get("sine_phase_offset_i", 0.0)),
        )

    if mode == "external_spike_train":
        return external_spike_train(
            N_E=n_e,
            N_I=n_i,
            I_E_base=i_e_base,
            I_I_base=i_i_base,
            noise_std_E=noise_std_e,
            noise_std_I=noise_std_i,
            num_steps=num_steps,
            dt=dt,
            seed=seed,
            lambda0_hz=float(input_spec.get("lambda0_hz", 30.0)),
            mod_depth=float(input_spec.get("mod_depth", 0.5)),
            envelope_freq_hz=float(input_spec.get("envelope_freq_hz", 5.0)),
            phase_rad=float(input_spec.get("phase_rad", 0.0)),
            w_in=float(input_spec.get("w_in", 0.25)),
            tau_in_ms=float(input_spec.get("tau_in_ms", 3.0)),
            return_spikes=False,
        )

    if mode in {"pulse", "pulses"}:
        baseline = ramp(
            N_E=n_e,
            N_I=n_i,
            I_E_start=i_e_base,
            I_E_end=i_e_base,
            I_I_start=i_i_base,
            I_I_end=i_i_base,
            noise_std_E=noise_std_e,
            noise_std_I=noise_std_i,
            num_steps=num_steps,
            dt=dt,
            seed=seed,
        )
        return baseline

    raise NotImplementedError(f"Unsupported input mode '{mode}'")


def compile_graph_to_runtime(
    spec: dict[str, Any] | None = None,
    *,
    backend: Literal["pytorch"] = "pytorch",
    device: str | None = None,
) -> RuntimeBundle:
    """
    Compile graph spec to runtime objects directly consumable by run_network.

    Compile graph spec to runtime objects directly consumable by run_network.
    """
    plan = compile_graph(spec)

    ordered_pops, n_e, n_i = _build_population_order(plan)
    pop_by_id = {entry["id"]: entry for entry in ordered_pops}
    if n_e <= 0:
        raise ValueError("graph must include at least one E population")
    sim = plan["sim"]
    biophysics = plan["biophysics"]

    config_overrides: dict[str, Any] = {
        "dt": float(sim["dt_ms"]),
        "T": float(sim["T_ms"]),
        "seed": int(sim["seed"]),
        "neuron_model": str(sim["neuron_model"]),
        "N_E": n_e,
        "N_I": n_i,
        **biophysics,
    }

    default_cfg = DEFAULT_CONFIG.model_dump()
    edge_delay_samples: dict[str, list[float]] = {"EE": [], "EI": [], "IE": [], "II": []}

    input_nodes = plan["input_nodes"]
    if not input_nodes:
        raise ValueError("graph requires at least one input node")

    config = DEFAULT_CONFIG.model_copy(update=config_overrides)
    num_steps = int(ceil(config.T / config.dt))
    n_total = config.N_E + config.N_I

    W = np.zeros((n_total, n_total), dtype=float)
    D = np.full((n_total, n_total), np.nan, dtype=float)
    weights_seed = int(sim["seed"])
    default_weights = copy.deepcopy(DEFAULT_WEIGHTS)
    default_w = {
        "EE": default_weights["ee"],
        "EI": default_weights["ei"],
        "IE": default_weights["ie"],
        "II": default_weights["ii"],
    }
    default_clamp = float(default_weights.get("clamp_min", 0.0))
    enforce_nonnegative_w = bool(plan["constraints"].get("nonnegative_weights", False))

    for edge in plan["edges"]:
        if not bool(edge.get("enabled", True)):
            continue
        kind = str(edge.get("kind", ""))
        if kind == "input":
            continue
        if kind not in {"EE", "EI", "IE", "II"}:
            raise ValueError(f"Unsupported edge kind '{kind}'")
        src_id = str(edge["from"])
        dst_id = str(edge["to"])
        if src_id not in pop_by_id or dst_id not in pop_by_id:
            raise ValueError(f"Population edge '{edge['id']}' must connect population nodes")
        src = pop_by_id[src_id]
        dst = pop_by_id[dst_id]
        src_type = str(src["type"])
        dst_type = str(dst["type"])
        expected = {
            "EE": ("E", "E"),
            "EI": ("E", "I"),
            "IE": ("I", "E"),
            "II": ("I", "I"),
        }[kind]
        if (src_type, dst_type) != expected:
            raise ValueError(
                f"edge '{edge['id']}' kind '{kind}' incompatible with types {src_type}->{dst_type}"
            )
        params = edge.get("params", {})
        w = params.get("w") if isinstance(params.get("w"), dict) else default_w[kind]
        dist_params = {
            str(k): float(v)
            for k, v in w.items()
        }
        edge_seed = (
            int(params["seed"])
            if params.get("seed") is not None
            else _stable_seed(weights_seed, str(edge["id"]))
        )
        rng = np.random.RandomState(edge_seed)
        tgt_size = int(dst["stop"] - dst["start"])
        src_size = int(src["stop"] - src["start"])
        block = _sample_distribution(rng, dist_params, (tgt_size, src_size))
        clamp_min = params.get("clamp_min")
        if clamp_min is None and enforce_nonnegative_w:
            clamp_min = 0.0
        if clamp_min is not None:
            block = np.clip(block, float(clamp_min), None)
        # Keep synaptic drive scale consistent with legacy weight builder.
        if src_size > 0:
            block *= 1.0 / np.sqrt(float(src_size))
        W[int(dst["start"]) : int(dst["stop"]), int(src["start"]) : int(src["stop"])] += block
        if params.get("delay_ms") is not None:
            delay_ms = float(params["delay_ms"])
            D[int(dst["start"]) : int(dst["stop"]), int(src["start"]) : int(src["stop"])] = delay_ms
            edge_delay_samples[kind].append(delay_ms)

    if enforce_nonnegative_w:
        np.maximum(W, 0.0, out=W)

    config_overrides["delay_ee"] = (
        float(np.mean(edge_delay_samples["EE"])) if edge_delay_samples["EE"] else float(default_cfg["delay_ee"])
    )
    config_overrides["delay_ei"] = (
        float(np.mean(edge_delay_samples["EI"])) if edge_delay_samples["EI"] else float(default_cfg["delay_ei"])
    )
    config_overrides["delay_ie"] = (
        float(np.mean(edge_delay_samples["IE"])) if edge_delay_samples["IE"] else float(default_cfg["delay_ie"])
    )
    config_overrides["delay_ii"] = (
        float(np.mean(edge_delay_samples["II"])) if edge_delay_samples["II"] else float(default_cfg["delay_ii"])
    )
    config = DEFAULT_CONFIG.model_copy(update=config_overrides)

    external_input = np.zeros((num_steps, n_total), dtype=float)
    nonnegative_input = bool(plan["constraints"].get("nonnegative_input", True))
    for input_id in input_nodes:
        input_spec = plan["input_specs"].get(input_id, {})
        if not isinstance(input_spec, dict):
            continue
        mode = str(input_spec.get("mode", "tonic"))
        input_seed = int(input_spec.get("seed", sim["seed"]))
        source_input = _build_external_input_for_node(
            mode=mode,
            n_e=config.N_E,
            n_i=config.N_I,
            dt=config.dt,
            num_steps=num_steps,
            seed=input_seed,
            input_spec=input_spec,
        )
        input_edges: list[dict[str, Any]] = []
        selected_union = np.zeros(n_total, dtype=bool)
        for edge in plan["edges"]:
            if not bool(edge.get("enabled", True)):
                continue
            if str(edge.get("kind")) != "input":
                continue
            if str(edge.get("from")) != str(input_id):
                continue
            dst_id = str(edge.get("to"))
            if dst_id not in pop_by_id:
                continue
            dst = pop_by_id[dst_id]
            edge_seed = _stable_seed(input_seed, str(edge.get("id", f"{input_id}->{dst_id}")))
            selected = _resolve_target_indices(
                edge.get("params", {}).get("target"),
                int(dst["start"]),
                int(dst["stop"]),
                seed=edge_seed,
            )
            if selected.size == 0:
                continue
            input_edges.append(
                {
                    "edge": edge,
                    "selected": selected,
                    "seed": edge_seed,
                }
            )
            selected_union[selected] = True
        if not np.any(selected_union):
            raise ValueError(
                f"input node '{input_id}' has no enabled input edge targets; connect it to a population with an input edge"
            )

        if mode in {"pulse", "pulses"}:
            pulse_t_ms = float(input_spec.get("pulse_t_ms", 200.0))
            pulse_width_ms = float(input_spec.get("pulse_width_ms", 20.0))
            pulse_interval_ms = float(input_spec.get("pulse_interval_ms", 100.0))
            pulse_amp_e = float(input_spec.get("pulse_amp_E", 1.0))
            pulse_amp_i = float(input_spec.get("pulse_amp_I", 1.0))
            e_ids = np.where(selected_union[: config.N_E])[0].astype(int)
            i_ids = np.where(selected_union[config.N_E :])[0].astype(int) + config.N_E
            if mode == "pulse":
                if e_ids.size > 0:
                    source_input = add_pulse_to_input(
                        source_input,
                        target_neurons=e_ids,
                        pulse_t=pulse_t_ms,
                        pulse_width_ms=pulse_width_ms,
                        pulse_amp=pulse_amp_e,
                        dt=config.dt,
                        num_steps=num_steps,
                    )
                if i_ids.size > 0:
                    source_input = add_pulse_to_input(
                        source_input,
                        target_neurons=i_ids,
                        pulse_t=pulse_t_ms,
                        pulse_width_ms=pulse_width_ms,
                        pulse_amp=pulse_amp_i,
                        dt=config.dt,
                        num_steps=num_steps,
                    )
            else:
                if e_ids.size > 0:
                    source_input = add_pulse_train_to_input(
                        source_input,
                        target_neurons=e_ids,
                        pulse_t=pulse_t_ms,
                        pulse_width_ms=pulse_width_ms,
                        pulse_amp=pulse_amp_e,
                        pulse_interval_ms=pulse_interval_ms,
                        dt=config.dt,
                        num_steps=num_steps,
                    )
                if i_ids.size > 0:
                    source_input = add_pulse_train_to_input(
                        source_input,
                        target_neurons=i_ids,
                        pulse_t=pulse_t_ms,
                        pulse_width_ms=pulse_width_ms,
                        pulse_amp=pulse_amp_i,
                        pulse_interval_ms=pulse_interval_ms,
                        dt=config.dt,
                        num_steps=num_steps,
                    )

        if input_edges:
            for payload in input_edges:
                edge = payload["edge"]
                selected = payload["selected"]
                gains = _resolve_input_edge_gains(
                    edge.get("params", {}),
                    count=selected.size,
                    seed=payload["seed"],
                    nonnegative_input=nonnegative_input,
                )
                external_input[:, selected] += source_input[:, selected] * gains[np.newaxis, :]
        else:
            external_input[:, selected_union] += source_input[:, selected_union]

    if nonnegative_input:
        np.maximum(external_input, 0.0, out=external_input)
    if backend != "pytorch":
        raise ValueError(f"Unsupported backend '{backend}'. Expected 'pytorch'.")
    try:
        import torch
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "PyTorch backend requested but torch is not installed."
        ) from exc
    weight_mats = resolve_weight_matrices_from_full(W=W, D=D, n_e=config.N_E)
    target_device = torch.device(device) if device else torch.device("cpu")
    weights_tensors = to_torch_weights(
        weight_mats,
        device=target_device,
        dtype=torch.float32,
    )
    from pinglab.backends.pytorch import lif_step

    return RuntimeBundle(
        config=config,
        external_input=torch.as_tensor(external_input, dtype=torch.float32, device=target_device),
        weights=weights_tensors,
        model=lif_step,
        backend="pytorch",
        device=str(target_device),
    )


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a pinglab-graph.v1 config JSON file.",
    )
    parser.add_argument(
        "config_path",
        type=Path,
        help="Path to config.json",
    )
    args = parser.parse_args()

    config_path = args.config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open(encoding="utf-8") as f:
        spec = json.load(f)

    plan = compile_graph(spec)
    sim = plan["sim"]
    totals = plan["totals"]
    warnings = plan.get("warnings", {})
    input_nodes = plan.get("input_nodes", [])
    missing_inputs = warnings.get("missing_input_specs", [])
    enabled_edges = sum(1 for e in plan["edges"] if bool(e.get("enabled", True)))

    table = Table(title="Pinglab Graph Validation")
    table.add_column("Field", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")

    table.add_row("Status", "VALID")
    table.add_row("Config path", str(config_path))
    table.add_row("Schema", str(plan["schema_version"]))
    table.add_row("Neuron model", str(sim["neuron_model"]))
    table.add_row("Seed", str(sim["seed"]))
    table.add_row("dt_ms", str(sim["dt_ms"]))
    table.add_row("T_ms", str(sim["T_ms"]))
    table.add_row("N_E", str(totals["N_E"]))
    table.add_row("N_I", str(totals["N_I"]))
    table.add_row("N_total", str(totals["N_total"]))
    table.add_row("Nodes", str(len(spec.get("nodes", []))))
    table.add_row("Edges (enabled/total)", f"{enabled_edges}/{len(plan['edges'])}")
    table.add_row("Input nodes", ", ".join(input_nodes) if input_nodes else "none")
    table.add_row("Missing input specs", ", ".join(missing_inputs) if missing_inputs else "none")

    Console().print(table)


if __name__ == "__main__":
    _main()
