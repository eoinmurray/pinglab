from __future__ import annotations

import json

import models as M
import pytest
import torch
from conformance import (
    CONFORMANCE_REPORT_SCHEMA,
    ComparisonPolicy,
    canonical_json_tensor,
    compare_conformance_layers,
    remap_named_tensors,
    write_conformance_report,
)
from execution import (
    ExecutionSpec,
    GraphExecutor,
    build,
    legacy_parameter_map_v1,
    train,
)

from tools import snnlang as snn
from tools.snnlang import training


def test_layered_conformance_requires_complete_exact_named_coverage(tmp_path):
    reference = {
        "parameters": {"input.weight": torch.tensor([[1.0, 2.0]])},
        "forward": {"logits": torch.tensor([[0.25, 0.75]])},
    }
    report = compare_conformance_layers("hand-checkable", reference, reference)
    assert report.passed
    report.require_passed()
    path = write_conformance_report(tmp_path / "conformance.json", report)
    payload = json.loads(path.read_text())
    assert payload["schema"] == CONFORMANCE_REPORT_SCHEMA
    assert payload["summary"] == {"comparisons": 2, "failed": 0, "passed": 2}

    incomplete = compare_conformance_layers(
        "missing", reference, {"parameters": reference["parameters"]}
    )
    assert not incomplete.passed
    assert incomplete.comparisons[0].reason == "missing from candidate"
    with pytest.raises(AssertionError, match="forward.logits"):
        incomplete.require_passed()


def test_numeric_policy_reports_error_and_never_hides_shape_or_dtype_mismatch():
    reference = {"gradients": {"readout.weight": torch.tensor([1.0, 2.0])}}
    close = {"gradients": {"readout.weight": torch.tensor([1.0, 2.00001])}}
    policy = {
        "gradients": {"readout.weight": ComparisonPolicy(mode="numeric", atol=2e-5)}
    }
    report = compare_conformance_layers("tolerant", reference, close, policies=policy)
    assert report.passed
    assert report.comparisons[0].max_abs_error == pytest.approx(1e-5, rel=0.01)

    wrong_dtype = {
        "gradients": {"readout.weight": torch.tensor([1.0, 2.0], dtype=torch.float64)}
    }
    mismatch = compare_conformance_layers(
        "dtype", reference, wrong_dtype, policies=policy
    )
    assert mismatch.comparisons[0].reason == "dtype mismatch"


def test_conformance_rejects_implicit_or_unused_tolerance_rules():
    with pytest.raises(ValueError, match="exact.*tolerances"):
        ComparisonPolicy(atol=1e-6)
    with pytest.raises(ValueError, match="absent fields"):
        compare_conformance_layers(
            "unused",
            {"forward": {"logits": torch.zeros(1)}},
            {"forward": {"logits": torch.zeros(1)}},
            policies={"forward": {"rates": ComparisonPolicy(mode="numeric")}},
        )


def test_canonical_json_tensor_compares_structural_layers_independent_of_key_order():
    first = canonical_json_tensor({"b": [2, 3], "a": 1})
    second = canonical_json_tensor({"a": 1, "b": [2, 3]})
    assert torch.equal(first, second)


def test_explicit_name_remapping_rejects_partial_and_duplicate_maps():
    values = {"graph.a": torch.ones(1), "graph.b": torch.zeros(1)}
    assert set(
        remap_named_tensors(values, {"graph.a": "legacy.a", "graph.b": "legacy.b"})
    ) == {"legacy.a", "legacy.b"}
    with pytest.raises(ValueError, match="must be complete"):
        remap_named_tensors(values, {"graph.a": "legacy.a"})
    with pytest.raises(ValueError, match="duplicate destination"):
        remap_named_tensors(values, {"graph.a": "legacy.a", "graph.b": "legacy.a"})


@pytest.mark.parametrize("active_recurrence", [False, True])
def test_minimal_legacy_and_graph_ping_forward_share_parameters_and_logits(
    active_recurrence,
):
    M.N_IN = 2
    M.N_OUT = 2
    M.dt = 0.1
    M.T_ms = 4.0
    M.T_steps = 40
    net = snn.Network("legacy_graph_ping", dt=0.1 * snn.ms)
    events = net.input(
        "events", shape=("time", "batch", 2), signal_type="spikes", unit="spike"
    )
    cell = snn.components.ping(
        net,
        name="cell",
        n_e=4,
        n_i=1,
        source=events,
        include_silent_recurrence=True,
    )
    scores = snn.readouts.MeanVoltage(source=cell.E.spikes, classes=2, name="scores")
    net.output("class_logits", scores)
    bundle = snn.compile(net)
    built = build(
        ExecutionSpec(kind="build", executor="graph", graph=bundle.graph, seed=7)
    )
    assert isinstance(built.model, GraphExecutor)
    graph_model = built.model
    graph_parameters = graph_model.parameter_map()
    with torch.no_grad():
        graph_parameters["cell_input.weight"].fill_(10.0)
        for name in (
            "cell_E_to_E.weight",
            "cell_E_to_I.weight",
            "cell_I_to_E.weight",
            "cell_I_to_I.weight",
        ):
            graph_parameters[name].zero_()
        if active_recurrence:
            graph_parameters["cell_E_to_I.weight"].fill_(2.0)
            graph_parameters["cell_I_to_E.weight"].fill_(5.0)
        graph_parameters["scores_projection.weight"].copy_(
            torch.tensor([[1.0, 0.5], [0.25, 1.5], [1.25, 0.75], [0.5, 1.0]])
        )

    legacy = M.COBANet(
        hidden_sizes=[4],
        n_inh_per_layer={1: 1},
        readout_mode="mem-mean",
        w_in=(0.0, 0.0),
        w_hid=(0.0, 0.0),
        w_ee=(0.0, 0.0),
        w_ei=(0.0, 0.0),
        w_ie=(0.0, 0.0),
        w_ii=(0.0, 0.0),
    )
    legacy.recording = True
    mapping = legacy_parameter_map_v1(bundle.graph)
    legacy_parameters = dict(legacy.named_parameters())
    with torch.no_grad():
        for graph_name, legacy_name in mapping.items():
            legacy_parameters[legacy_name].copy_(graph_parameters[graph_name])

    inputs = torch.zeros(40, 2, 2)
    inputs[:, 0, 0] = 1
    inputs[::2, 1, 1] = 1
    graph = graph_model({"events": inputs}, record="full")
    legacy_logits = legacy(input_spikes=inputs)
    if active_recurrence:
        assert torch.count_nonzero(legacy.spike_record["inh"]) > 0
        assert torch.count_nonzero(legacy.spike_record["gi_e_1"]) > 0
    report = compare_conformance_layers(
        "minimal-legacy-graph-ping",
        {
            "parameters": remap_named_tensors(graph.parameters, mapping),
            "forward": {
                "e_spikes": legacy.spike_record["hid"],
                "i_spikes": legacy.spike_record["inh"],
                "e_voltage": legacy.spike_record["v_e_1"],
                "i_voltage": legacy.spike_record["v_i_1"],
                "input_conductance": legacy.spike_record["ge_e_1"],
                "e_to_i_conductance": legacy.spike_record["ge_i_1"],
                "i_to_e_conductance": legacy.spike_record["gi_e_1"],
                "logits": legacy_logits.detach(),
            },
        },
        {
            "parameters": {
                name: value.detach() for name, value in legacy_parameters.items()
            },
            "forward": {
                "e_spikes": graph.recordings["cell_E.spikes"],
                "i_spikes": graph.recordings["cell_I.spikes"],
                "e_voltage": graph.recordings["cell_E.voltage"],
                "i_voltage": graph.recordings["cell_I.voltage"],
                "input_conductance": graph.recordings["cell_input.conductance"],
                "e_to_i_conductance": graph.recordings["cell_E_to_I.conductance"],
                "i_to_e_conductance": graph.recordings["cell_I_to_E.conductance"],
                "logits": graph.outputs["class_logits"].detach(),
            },
        },
        policies={
            "forward": {
                "logits": ComparisonPolicy(mode="numeric", atol=1e-6, rtol=1e-6)
            }
        },
    )
    report.require_passed()


def test_legacy_and_graph_one_step_backward_and_adamw_are_conformant():
    M.N_IN = 2
    M.N_OUT = 2
    M.dt = 0.1
    M.T_ms = 4.0
    M.T_steps = 40
    net = snn.Network("legacy_graph_backward", dt=0.1 * snn.ms)
    events = net.input(
        "events", shape=("time", "batch", 2), signal_type="spikes", unit="spike"
    )
    cell = snn.components.ping(
        net,
        name="cell",
        n_e=4,
        n_i=1,
        source=events,
        include_silent_recurrence=True,
    )
    scores = snn.readouts.MeanVoltage(source=cell.E.spikes, classes=2, name="scores")
    net.output("class_logits", scores)
    recurrent = [
        "cell_E_to_E.weight",
        "cell_E_to_I.weight",
        "cell_I_to_E.weight",
        "cell_I_to_I.weight",
    ]
    trainable = ["cell_input.weight", "scores_projection.weight", *recurrent]
    recipe = snn.TrainSpec(
        objectives=[training.CrossEntropy(prediction=scores, target="label")],
        parameter_groups=[
            training.ParameterGroup(trainable, name="all", lr=0.01),
        ],
        optimizer=training.AdamW(weight_decay=0.0),
        surrogate=training.FastSigmoid(slope=5.0),
        presentation_duration=4.0 * snn.ms,
    )
    bundle = snn.compile(net, training=recipe)
    initial = build(
        ExecutionSpec(
            kind="build",
            executor="graph",
            graph=bundle.graph,
            training=bundle.training,
            seed=7,
        )
    )
    assert isinstance(initial.model, GraphExecutor)
    mapping = legacy_parameter_map_v1(bundle.graph)
    legacy = M.COBANet(
        hidden_sizes=[4],
        n_inh_per_layer={1: 1},
        readout_mode="mem-mean",
        w_in=(0.0, 0.0),
        w_hid=(0.0, 0.0),
        w_ee=(0.0, 0.0),
        w_ei=(0.0, 0.0),
        w_ie=(0.0, 0.0),
        w_ii=(0.0, 0.0),
        trainable_w_ee=True,
        trainable_w_ei=True,
        trainable_w_ie=True,
        trainable_w_ii=True,
    )
    legacy_parameters = dict(legacy.named_parameters())
    with torch.no_grad():
        for graph_name, legacy_name in mapping.items():
            legacy_parameters[legacy_name].copy_(initial.parameters[graph_name])

    inputs = torch.zeros(40, 2, 2)
    inputs[:, 0, 0] = 1
    inputs[::2, 1, 1] = 1
    labels = torch.tensor([0, 1])
    graph = train(
        ExecutionSpec(
            kind="train",
            executor="graph",
            graph=bundle.graph,
            training=bundle.training,
            inputs={"events": inputs},
            targets={"label": labels},
            seed=7,
        )
    )
    optimizer = torch.optim.AdamW(
        [legacy_parameters[mapping[name]] for name in sorted(trainable)],
        lr=0.01,
        weight_decay=0.0,
    )
    optimizer.zero_grad(set_to_none=True)
    legacy_logits = legacy(input_spikes=inputs)
    legacy_loss = torch.nn.functional.cross_entropy(legacy_logits, labels)
    legacy_loss.backward()
    legacy_gradients = {
        mapping[name]: legacy_parameters[mapping[name]].grad.detach().clone()
        for name in trainable
    }
    assert torch.count_nonzero(legacy_gradients["W_ff.0"]) > 0
    assert torch.count_nonzero(legacy_gradients["W_ff.1"]) > 0
    assert any(
        torch.count_nonzero(legacy_gradients[mapping[name]]) > 0 for name in recurrent
    )
    optimizer.step()
    with torch.no_grad():
        for parameter in legacy_parameters.values():
            parameter.clamp_(min=0)
    legacy_optimizer = {}
    for name in sorted(legacy_gradients):
        for state, value in optimizer.state[legacy_parameters[name]].items():
            if isinstance(value, torch.Tensor):
                legacy_optimizer[f"{name}.{state}"] = value.detach()
    graph_optimizer = {
        f"{mapping[name]}.{state}": value
        for name, values in graph.optimizer_state.items()
        for state, value in values.items()
        if isinstance(value, torch.Tensor)
    }
    numeric = ComparisonPolicy(mode="numeric", atol=1e-6, rtol=1e-6)
    report = compare_conformance_layers(
        "legacy-graph-backward",
        {
            "loss": {"cross_entropy": legacy_loss.detach()},
            "gradients": legacy_gradients,
            "parameters": {
                name: value.detach() for name, value in legacy_parameters.items()
            },
            "optimizer": legacy_optimizer,
        },
        {
            "loss": {
                "cross_entropy": torch.tensor(graph.metrics["updates"][0]["loss"])
            },
            "gradients": remap_named_tensors(
                graph.gradients, {name: mapping[name] for name in graph.gradients}
            ),
            "parameters": remap_named_tensors(graph.parameters, mapping),
            "optimizer": graph_optimizer,
        },
        policies={
            "loss": {"cross_entropy": numeric},
            "gradients": {name: numeric for name in legacy_gradients},
            "parameters": {name: numeric for name in legacy_parameters},
            "optimizer": {name: numeric for name in legacy_optimizer},
        },
    )
    report.require_passed()
