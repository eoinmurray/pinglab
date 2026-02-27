import math

import numpy as np
import pytest

from pinglab.io import compile_graph, compile_graph_to_runtime


def test_compile_graph_scales_edge_weights_by_inverse_sqrt_source_size() -> None:
    graph = {
        "schema_version": "pinglab-graph.v1",
        "sim": {"dt_ms": 0.1, "T_ms": 100.0, "seed": 0, "neuron_model": "lif"},
        "nodes": [
            {"id": "in", "kind": "input", "type": "tonic", "size": 0},
            {"id": "E", "kind": "population", "type": "E", "size": 4},
            {"id": "I", "kind": "population", "type": "I", "size": 2},
        ],
        "edges": [
            {
                "id": "in_to_e",
                "from": "in",
                "to": "E",
                "kind": "input",
                "w": {"mean": 1.0, "std": 0.0},
            },
            {
                "id": "e_to_i",
                "from": "E",
                "to": "I",
                "kind": "EI",
                "w": {"mean": 1.0, "std": 0.0},
                "clamp_min": 0.0,
            }
        ],
        "inputs": {"in": {"mode": "tonic", "mean": 0.0, "std": 0.0}},
        "constraints": {"nonnegative_input": True, "nonnegative_weights": True},
    }

    runtime = compile_graph_to_runtime(graph)
    weights = runtime["weights"]

    expected = 1.0 / math.sqrt(4.0)
    assert np.allclose(weights.W_ei, expected)


def test_compile_graph_allows_meta_field_for_experiment_parameters() -> None:
    graph = {
        "schema_version": "pinglab-graph.v1",
        "sim": {"dt_ms": 0.1, "T_ms": 10.0, "seed": 0, "neuron_model": "lif"},
        "nodes": [
            {"id": "in", "kind": "input", "type": "tonic", "size": 0},
            {"id": "E", "kind": "population", "type": "E", "size": 2},
        ],
        "edges": [
            {"id": "in_to_e", "from": "in", "to": "E", "kind": "input", "w": {"mean": 1.0, "std": 0.0}}
        ],
        "inputs": {"in": {"mode": "tonic", "mean": 0.0, "std": 0.0}},
        "meta": {
            "scan": {"parameter": "e_to_i.w.mean", "values": [0.0, 0.1, 0.2]},
            "notes": "study.3 sweep setup",
        },
    }

    plan = compile_graph(graph)
    assert plan["meta"]["scan"]["parameter"] == "e_to_i.w.mean"
    assert plan["meta"]["scan"]["values"] == [0.0, 0.1, 0.2]
    assert plan["meta"]["notes"] == "study.3 sweep setup"


def test_compile_graph_rejects_non_object_meta() -> None:
    graph = {
        "schema_version": "pinglab-graph.v1",
        "sim": {"dt_ms": 0.1, "T_ms": 10.0, "seed": 0, "neuron_model": "lif"},
        "nodes": [
            {"id": "in", "kind": "input", "type": "tonic", "size": 0},
            {"id": "E", "kind": "population", "type": "E", "size": 2},
        ],
        "edges": [
            {"id": "in_to_e", "from": "in", "to": "E", "kind": "input", "w": {"mean": 1.0, "std": 0.0}}
        ],
        "inputs": {"in": {"mode": "tonic", "mean": 0.0, "std": 0.0}},
        "meta": "nope",
    }

    with pytest.raises(ValueError, match="meta must be an object"):
        compile_graph(graph)


def test_compile_graph_rejects_unknown_top_level_field() -> None:
    graph = {
        "schema_version": "pinglab-graph.v1",
        "sim": {"dt_ms": 0.1, "T_ms": 10.0, "seed": 0, "neuron_model": "lif"},
        "nodes": [
            {"id": "in", "kind": "input", "type": "tonic", "size": 0},
            {"id": "E", "kind": "population", "type": "E", "size": 2},
        ],
        "edges": [
            {"id": "in_to_e", "from": "in", "to": "E", "kind": "input", "w": {"mean": 1.0, "std": 0.0}}
        ],
        "inputs": {"in": {"mode": "tonic", "mean": 0.0, "std": 0.0}},
        "unknown": True,
    }

    with pytest.raises(ValueError, match="spec has unknown field\\(s\\): unknown"):
        compile_graph(graph)


def test_compile_graph_rejects_unknown_nested_fields() -> None:
    base = {
        "schema_version": "pinglab-graph.v1",
        "sim": {"dt_ms": 0.1, "T_ms": 10.0, "seed": 0, "neuron_model": "lif"},
        "nodes": [
            {"id": "in", "kind": "input", "type": "tonic", "size": 0},
            {"id": "E", "kind": "population", "type": "E", "size": 2},
        ],
        "edges": [
            {"id": "in_to_e", "from": "in", "to": "E", "kind": "input", "w": {"mean": 1.0, "std": 0.0}}
        ],
        "inputs": {"in": {"mode": "tonic", "mean": 0.0, "std": 0.0}},
    }

    graph_sim = dict(base)
    graph_sim["sim"] = {**base["sim"], "extra": 1}
    with pytest.raises(ValueError, match="sim has unknown field\\(s\\): extra"):
        compile_graph(graph_sim)

    graph_node = dict(base)
    graph_node["nodes"] = [
        {"id": "in", "kind": "input", "type": "tonic", "size": 0, "extra": 1},
        {"id": "E", "kind": "population", "type": "E", "size": 2},
    ]
    with pytest.raises(ValueError, match="node has unknown field\\(s\\): extra"):
        compile_graph(graph_node)

    graph_edge = dict(base)
    graph_edge["edges"] = [
        {"id": "in_to_e", "from": "in", "to": "E", "kind": "input", "w": {"mean": 1.0, "std": 0.0}, "extra": 1}
    ]
    with pytest.raises(ValueError, match="edge has unknown field\\(s\\): extra"):
        compile_graph(graph_edge)

    graph_input = dict(base)
    graph_input["inputs"] = {"in": {"mode": "tonic", "mean": 0.0, "std": 0.0, "extra": 1}}
    with pytest.raises(ValueError, match="inputs.in has unknown field\\(s\\): extra"):
        compile_graph(graph_input)


def test_compile_graph_rejects_unknown_input_node_mapping() -> None:
    graph = {
        "schema_version": "pinglab-graph.v1",
        "sim": {"dt_ms": 0.1, "T_ms": 10.0, "seed": 0, "neuron_model": "lif"},
        "nodes": [
            {"id": "in", "kind": "input", "type": "tonic", "size": 0},
            {"id": "E", "kind": "population", "type": "E", "size": 2},
        ],
        "edges": [
            {"id": "in_to_e", "from": "in", "to": "E", "kind": "input", "w": {"mean": 1.0, "std": 0.0}}
        ],
        "inputs": {
            "in": {"mode": "tonic", "mean": 0.0, "std": 0.0},
            "ghost": {"mode": "tonic", "mean": 0.0, "std": 0.0},
        },
    }

    with pytest.raises(ValueError, match="inputs has unknown input node 'ghost'"):
        compile_graph(graph)


def test_compile_graph_to_runtime_rejects_unknown_backend() -> None:
    graph = {
        "schema_version": "pinglab-graph.v1",
        "sim": {"dt_ms": 0.1, "T_ms": 10.0, "seed": 0, "neuron_model": "lif"},
        "nodes": [
            {"id": "in", "kind": "input", "type": "tonic", "size": 0},
            {"id": "E", "kind": "population", "type": "E", "size": 2},
        ],
        "edges": [
            {"id": "in_to_e", "from": "in", "to": "E", "kind": "input", "w": {"mean": 1.0, "std": 0.0}}
        ],
        "inputs": {"in": {"mode": "tonic", "mean": 0.0, "std": 0.0}},
    }

    with pytest.raises(ValueError, match="Unsupported backend 'jax'"):
        compile_graph_to_runtime(graph, backend="jax")  # type: ignore[arg-type]


def test_compile_graph_to_runtime_supports_pytorch_backend() -> None:
    torch = pytest.importorskip("torch")
    graph = {
        "schema_version": "pinglab-graph.v1",
        "sim": {"dt_ms": 0.1, "T_ms": 10.0, "seed": 0, "neuron_model": "lif"},
        "nodes": [
            {"id": "in", "kind": "input", "type": "tonic", "size": 0},
            {"id": "E", "kind": "population", "type": "E", "size": 2},
        ],
        "edges": [
            {"id": "in_to_e", "from": "in", "to": "E", "kind": "input", "w": {"mean": 1.0, "std": 0.0}}
        ],
        "inputs": {"in": {"mode": "tonic", "mean": 0.0, "std": 0.0}},
    }

    runtime = compile_graph_to_runtime(graph, backend="pytorch")
    assert runtime["backend"] == "pytorch"
    assert isinstance(runtime["external_input"], torch.Tensor)
    assert isinstance(runtime["weights"]["W"], torch.Tensor)
