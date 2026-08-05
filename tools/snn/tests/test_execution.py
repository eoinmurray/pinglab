"""Focused acceptance tests for the typed seam and graph executor."""

from __future__ import annotations

from pathlib import Path

import torch
import config
import models as M

from execution import (
    ExecutionSpec,
    DelayBuffer,
    build,
    graph_capability_issues,
    execution_spec_from_args,
    execute_request,
    plan_graph,
    simulate,
    train,
)
from tools import snnlang as snn
from tools.snnlang.examples.build_examples import ping_classifier
from tool import parse_args


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


def test_legacy_and_bundle_cli_arguments_both_lower_to_typed_specs(tmp_path):
    legacy = execution_spec_from_args(parse_args(["sim"]))
    assert legacy.executor == "legacy" and legacy.bundle is None
    root = ping_classifier().write(tmp_path / "ping.bundle")
    graph = execution_spec_from_args(parse_args(["sim", "--bundle", str(root), "--executor", "graph"]))
    assert graph.executor == "graph" and graph.bundle == root
    called = []
    result = execute_request(legacy, legacy=lambda: (called.append(True) or build(legacy)))
    assert called and result.executor == "legacy"


def test_representative_shd_checkpoint_and_recording_requests_remain_legacy():
    shd = execution_spec_from_args(
        parse_args(["train", "--dataset", "shd", "--max-samples", "8", "--epochs", "1"])
    )
    assert shd.executor == "legacy"
    assert shd.options["dataset"] == "shd"
    assert shd.options["max_samples"] == 8
    checkpoint = execution_spec_from_args(
        parse_args(["sim", "--load-weights", "checkpoint.pth", "--outputs", "rasters"])
    )
    assert checkpoint.executor == "legacy"
    assert checkpoint.checkpoint == Path("checkpoint.pth")
    assert checkpoint.options["outputs"] == ["rasters"]


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

    pulse = torch.tensor([[1.0]])
    zeros = torch.zeros_like(pulse)
    buffer = DelayBuffer(3, pulse)
    received = []
    for emitted in (pulse, zeros, zeros, zeros, zeros):
        received.append(float(buffer.read().item()))
        buffer.push(emitted)
    assert received == [0.0, 0.0, 0.0, 1.0, 0.0]


def test_input_delay_pulse_arrives_on_exact_timestep_and_handles_boundary():
    graph = _coupled_graph(direction="uncoupled")
    projection = next(p for p in graph["projections"] if p["id"] == "a_input")
    projection["delay"] = {"value": 0.3, "unit": "ms"}
    parameter = next(p for p in graph["parameters"] if p["id"] == "a_input.weight")
    parameter["initializer"] = {"kind": "constant", "value": 1.0}
    inputs = {"drive_a": torch.zeros(5, 1, 3), "drive_b": torch.zeros(5, 1, 2)}
    inputs["drive_a"][0, 0, 0] = 1.0
    result = simulate(ExecutionSpec(kind="simulate", executor="graph", graph=graph, inputs=inputs))
    conductance = result.recordings["a_input.conductance"][:, 0]
    assert torch.count_nonzero(conductance[:3]) == 0
    assert torch.count_nonzero(conductance[3]) > 0


def test_non_integral_delay_is_rejected_before_execution():
    graph = _coupled_graph(delay_ms=0.15)
    try:
        plan_graph(graph)
    except ValueError as exc:
        assert "integer number" in str(exc)
    else:
        raise AssertionError("fractional-step delay must fail planning")


def test_single_ping_seeded_parameters_and_forward_match_legacy_exactly():
    bundle = ping_classifier()
    graph = bundle.graph
    torch.manual_seed(17)
    M.N_IN = 784
    M.N_OUT = 10
    config.set_sim_dt(0.1, 1.2)
    M.T_steps = 12
    legacy = config.build_net(
        "ping", w_in=(0.2, 0.03), w_in_sparsity=0.0,
        w_ei=(0.5, 0.05), w_ie=(1.0, 0.1),
        ei_strength=0.5, ei_ratio=2.0, sparsity=0.0,
        hidden_sizes=[256], readout_mode="mem-mean",
    )
    graph_model = build(ExecutionSpec(kind="build", executor="graph", graph=graph, seed=17)).model
    assert graph_model is not None
    mapping = {
        "sensory_ping_input.weight": legacy.W_ff[0],
        "classifier_projection.weight": legacy.W_ff[1],
        "sensory_ping_E_to_I.weight": legacy.W_ei["1"],
        "sensory_ping_I_to_E.weight": legacy.W_ie["1"],
    }
    for name, expected in mapping.items():
        torch.testing.assert_close(graph_model.parameter_map()[name], expected, rtol=0, atol=0)

    spikes = torch.zeros(12, 2, 784)
    spikes[0::2, :, :48] = 1.0
    legacy.recording = True
    legacy_logits = legacy(input_spikes=spikes)
    native = graph_model({"image": spikes}, record=True)
    torch.testing.assert_close(native.outputs["class_logits"], legacy_logits, rtol=0, atol=2e-7)
    torch.testing.assert_close(native.recordings["cell_0"], legacy.spike_record["hid"], rtol=0, atol=0)
    torch.testing.assert_close(native.recordings["cell_1"], legacy.spike_record["inh"], rtol=0, atol=0)
