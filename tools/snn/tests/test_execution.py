"""Focused acceptance tests for the typed seam and graph executor."""

from __future__ import annotations

import torch

from execution import (
    ExecutionSpec,
    build,
    graph_capability_issues,
    plan_graph,
    simulate,
    train,
)
from tools import snnlang as snn


def _coupled_graph(*, direction="reciprocal", delay_ms=0.1):
    net = snn.Network("coupled_ping_gate", dt=0.1 * snn.ms)
    drive_a = net.input("drive_a", shape=("time", "batch", 3), signal_type="spikes", unit="spike")
    drive_b = net.input("drive_b", shape=("time", "batch", 2), signal_type="spikes", unit="spike")
    a = snn.components.ping(net, name="a", n_e=4, n_i=1, source=drive_a)
    b = snn.components.ping(net, name="b", n_e=6, n_i=2, source=drive_b)
    if direction in {"unidirectional", "reciprocal"}:
        net.connect(a.I.spikes, b.E.inhibitory, name="a_I_to_b_E", synapse=snn.GABA(tau=9 * snn.ms), weight=snn.Constant(4.0), constraint=snn.NonNegative(), connection="feedback", delay=delay_ms * snn.ms)
    if direction == "reciprocal":
        net.connect(b.I.spikes, a.E.inhibitory, name="b_I_to_a_E", synapse=snn.GABA(tau=9 * snn.ms), weight=snn.Constant(4.0), constraint=snn.NonNegative(), connection="feedback", delay=delay_ms * snn.ms)
    net.expose(a.E.spikes, a.I.spikes, b.E.spikes, b.I.spikes, name="coupled")
    return snn.compile(net, target=None).graph


def test_typed_request_defaults_to_legacy_and_graph_training_is_explicitly_gated():
    request = ExecutionSpec(kind="build")
    assert request.executor == "legacy"
    assert build(request).metrics["routing"] == "legacy"
    try:
        train(ExecutionSpec(kind="train", executor="graph", graph=_coupled_graph()))
    except NotImplementedError as exc:
        assert "Milestone 6" in str(exc)
    else:
        raise AssertionError("graph training must not silently route")


def test_arbitrary_sizes_independent_inputs_and_all_population_recordings():
    graph = _coupled_graph()
    assert not graph_capability_issues(graph)
    inputs = {"drive_a": torch.zeros(8, 1, 3), "drive_b": torch.zeros(8, 1, 2)}
    result = simulate(ExecutionSpec(kind="simulate", executor="graph", graph=graph, inputs=inputs, seed=3))
    assert result.executor == "graph"
    assert {"coupled_0", "coupled_1", "coupled_2", "coupled_3"} <= result.recordings.keys()
    assert result.recordings["coupled_0"].shape == (8, 1, 4)
    assert result.recordings["coupled_3"].shape == (8, 1, 2)


def test_delay_lowering_is_exact_in_steps_and_feedback_is_causal():
    one = plan_graph(_coupled_graph(delay_ms=0.1))
    delayed = plan_graph(_coupled_graph(delay_ms=0.3))
    assert next(p for p in one.projections if p.id == "a_I_to_b_E").delay_steps == 1
    assert next(p for p in delayed.projections if p.id == "a_I_to_b_E").delay_steps == 3


def test_non_integral_delay_is_rejected_before_execution():
    graph = _coupled_graph(delay_ms=0.15)
    try:
        plan_graph(graph)
    except ValueError as exc:
        assert "integer number" in str(exc)
    else:
        raise AssertionError("fractional-step delay must fail planning")
