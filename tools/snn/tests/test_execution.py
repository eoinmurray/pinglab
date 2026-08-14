"""Focused acceptance tests for the typed seam and graph executor."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import config
import models as M
import numpy as np
import pytest
import torch
from execution import (
    DelayBuffer,
    DenseArrayBinding,
    ExecutionSpec,
    GraphExecutor,
    GraphRuntimeState,
    build,
    execute_request,
    execution_spec_from_args,
    graph_capability_issues,
    load_dense_array_bindings,
    load_runtime_state,
    plan_graph,
    resolve_dense_array_bindings,
    resolve_device,
    runtime_state_signature,
    save_runtime_state,
    simulate,
    train,
)
from tool import main, parse_args

from tools import snnlang as snn
from tools.snnlang.examples.build_examples import ping_classifier


def _coupled_graph(*, direction="reciprocal", delay_ms=0.1):
    net = snn.Network("coupled_ping_gate", dt=0.1 * snn.ms)
    drive_a = net.input(
        "drive_a", shape=("time", "batch", 3), signal_type="spikes", unit="spike"
    )
    drive_b = net.input(
        "drive_b", shape=("time", "batch", 2), signal_type="spikes", unit="spike"
    )
    a = snn.components.ping(net, name="a", n_e=4, n_i=1, source=drive_a)
    b = snn.components.ping(net, name="b", n_e=6, n_i=2, source=drive_b)
    if direction in {"unidirectional", "reciprocal"}:
        net.connect(
            a.I.spikes,
            b.E.inhibitory,
            name="a_I_to_b_E",
            synapse=snn.GABA(tau=9 * snn.ms),
            weight=snn.Constant(4.0),
            constraint=snn.NonNegative(),
            connection="feedback",
            delay=delay_ms * snn.ms,
        )
    if direction == "reciprocal":
        net.connect(
            b.I.spikes,
            a.E.inhibitory,
            name="b_I_to_a_E",
            synapse=snn.GABA(tau=9 * snn.ms),
            weight=snn.Constant(4.0),
            constraint=snn.NonNegative(),
            connection="feedback",
            delay=delay_ms * snn.ms,
        )
    net.expose(a.E.spikes, a.I.spikes, b.E.spikes, b.I.spikes, name="coupled")
    return snn.compile(net, target=None).graph


def test_typed_request_defaults_to_legacy_and_graph_training_is_explicitly_gated():
    request = ExecutionSpec(kind="build")
    assert request.executor == "legacy"
    assert request.device == "auto"
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
    graph = execution_spec_from_args(
        parse_args(["sim", "--bundle", str(root), "--executor", "graph"])
    )
    assert graph.executor == "graph" and graph.bundle == root
    called = []
    result = execute_request(
        legacy, legacy=lambda: called.append(True) or build(legacy)
    )
    assert called and result.executor == "legacy"


def test_graph_cli_resolves_explicit_device_and_recording_profile(
    tmp_path, monkeypatch
):
    root = ping_classifier().write(tmp_path / "ping.bundle")
    graph = execution_spec_from_args(
        parse_args(
            [
                "sim",
                "--bundle",
                str(root),
                "--executor",
                "graph",
                "--device",
                "cpu",
                "--recording",
                "observables",
            ]
        )
    )
    assert graph.device == "cpu"
    assert graph.recording == "observables"
    assert resolve_device("cpu") == "cpu"
    monkeypatch.setenv("PINGLAB_DEVICE", "cpu")
    assert resolve_device("auto") == "cpu"


def test_recording_profiles_select_full_observable_or_no_traces():
    graph = _coupled_graph()
    inputs = {"drive_a": torch.zeros(4, 1, 3), "drive_b": torch.zeros(4, 1, 2)}
    full = simulate(
        ExecutionSpec(
            kind="simulate",
            executor="graph",
            graph=graph,
            inputs=inputs,
            recording="full",
        )
    )
    observables = simulate(
        ExecutionSpec(
            kind="simulate",
            executor="graph",
            graph=graph,
            inputs=inputs,
            recording="observables",
        )
    )
    none = simulate(
        ExecutionSpec(
            kind="simulate",
            executor="graph",
            graph=graph,
            inputs=inputs,
            recording="none",
        )
    )
    observable_names = {row["id"] for row in graph["observables"]}
    assert observable_names < set(full.recordings)
    assert set(observables.recordings) == observable_names
    assert not none.recordings
    assert full.metrics["recording"] == "full"
    assert observables.metrics["recording"] == "observables"


def _state_tensors(state: GraphRuntimeState):
    for group in (
        state.voltages,
        state.refractory,
        state.conductances,
        state.population_histories,
        state.input_histories,
    ):
        yield from group.values()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")
def test_graph_cpu_mps_parity_and_all_result_state_follows_device():
    graph = _coupled_graph()
    inputs = {"drive_a": torch.zeros(12, 1, 3), "drive_b": torch.zeros(12, 1, 2)}
    inputs["drive_a"][0, 0, 0] = 1.0
    cpu = simulate(
        ExecutionSpec(
            kind="simulate",
            executor="graph",
            graph=graph,
            inputs=inputs,
            seed=23,
            device="cpu",
            recording="observables",
        )
    )
    mps = simulate(
        ExecutionSpec(
            kind="simulate",
            executor="graph",
            graph=graph,
            inputs=inputs,
            seed=23,
            device="mps",
            recording="observables",
        )
    )
    assert mps.runtime_state is not None
    assert all(value.device.type == "mps" for value in mps.parameters.values())
    assert all(value.device.type == "mps" for value in mps.recordings.values())
    assert all(value.device.type == "mps" for value in mps.outputs.values())
    assert all(
        value.device.type == "mps" for value in _state_tensors(mps.runtime_state)
    )
    for name in cpu.recordings:
        torch.testing.assert_close(
            mps.recordings[name].cpu(),
            cpu.recordings[name],
            rtol=1e-5,
            atol=1e-6,
        )
    assert mps.metrics["device"] == "mps"


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
    result = simulate(
        ExecutionSpec(
            kind="simulate", executor="graph", graph=graph, inputs=inputs, seed=3
        )
    )
    assert result.executor == "graph"
    assert {
        "coupled_0",
        "coupled_1",
        "coupled_2",
        "coupled_3",
    } <= result.recordings.keys()
    assert result.recordings["coupled_0"].shape == (8, 1, 4)
    assert result.recordings["coupled_3"].shape == (8, 1, 2)


def _standard_readout_graph(
    readout: str, *, duration: float | None = None, mask: bool = False
):
    net = snn.Network("readout_fixture", dt=100 * snn.ms)
    source = net.input(
        "events", shape=("time", "batch", 2), signal_type="spikes", unit="spike"
    )
    valid = (
        net.input("valid", shape=("time", "batch"), signal_type="mask")
        if mask
        else None
    )
    if readout == "final":
        value = snn.readouts.FinalVoltage(source=source, classes=2, name="scores")
    elif readout == "count":
        value = snn.readouts.SpikeCount(source=source, classes=2, name="scores")
    elif readout == "rate":
        value = snn.readouts.SpikeRate(
            source=source,
            classes=2,
            name="scores",
            duration=duration,
            mask=valid,
        )
    elif readout == "cumulative":
        value = snn.readouts.CumulativePotential(
            source=source, classes=2, name="scores"
        )
    else:
        raise ValueError(readout)
    net.output("class_scores", value)
    graph = snn.compile(net, target="tools/snn").graph
    for parameter in graph["parameters"]:
        parameter["initializer"] = {"kind": "constant", "value": 1.0}
    return graph


def test_graph_standard_readouts_match_hand_calculated_fixtures():
    events = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 1.0], [0.0, 0.0]],
            [[0.0, 1.0], [1.0, 1.0]],
        ]
    )
    projected = events.sum(dim=2, keepdim=True).expand(-1, -1, 2)

    final = simulate(
        ExecutionSpec(
            kind="simulate",
            executor="graph",
            graph=_standard_readout_graph("final"),
            inputs={"events": events},
        )
    )
    torch.testing.assert_close(
        final.outputs["class_scores"], projected[-1], rtol=0, atol=0
    )

    count = simulate(
        ExecutionSpec(
            kind="simulate",
            executor="graph",
            graph=_standard_readout_graph("count"),
            inputs={"events": events},
        )
    )
    torch.testing.assert_close(
        count.outputs["class_scores"], projected.sum(dim=0), rtol=0, atol=0
    )

    rate = simulate(
        ExecutionSpec(
            kind="simulate",
            executor="graph",
            graph=_standard_readout_graph("rate", duration=0.3),
            inputs={"events": events},
        )
    )
    torch.testing.assert_close(
        rate.outputs["class_scores"], projected.sum(dim=0) / 0.3, rtol=0, atol=0
    )

    cumulative = simulate(
        ExecutionSpec(
            kind="simulate",
            executor="graph",
            graph=_standard_readout_graph("cumulative"),
            inputs={"events": events},
        )
    )
    torch.testing.assert_close(
        cumulative.outputs["class_scores"], projected.cumsum(dim=0), rtol=0, atol=0
    )


def test_graph_masked_spike_rate_uses_valid_duration_in_spikes_per_second():
    events = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 1.0], [0.0, 0.0]],
            [[0.0, 1.0], [1.0, 1.0]],
        ]
    )
    valid = torch.tensor(
        [
            [True, True],
            [False, True],
            [True, False],
        ]
    )
    projected = events.sum(dim=2, keepdim=True).expand(-1, -1, 2)
    expected_count = (projected * valid[:, :, None]).sum(dim=0)
    expected_seconds = valid.sum(dim=0).to(projected.dtype)[:, None] * 0.1
    result = simulate(
        ExecutionSpec(
            kind="simulate",
            executor="graph",
            graph=_standard_readout_graph("rate", mask=True),
            inputs={"events": events, "valid": valid},
        )
    )
    torch.testing.assert_close(
        result.outputs["class_scores"],
        expected_count / expected_seconds,
        rtol=0,
        atol=0,
    )


def test_dense_array_bindings_match_hand_calculation_and_record_protocol():
    graph = _standard_readout_graph("rate", mask=True)
    events = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 1.0], [0.0, 0.0]],
            [[0.0, 1.0], [1.0, 1.0]],
        ]
    )
    valid = torch.tensor([[True, True], [False, True], [True, False]])
    result = simulate(
        ExecutionSpec(
            kind="simulate",
            executor="graph",
            graph=graph,
            seed=17,
            input_bindings=(
                DenseArrayBinding(
                    "events", events, {"kind": "fixture", "id": "events-v1"}
                ),
                DenseArrayBinding("valid", valid, {"kind": "fixture", "id": "mask-v1"}),
            ),
            protocol={
                "dataset": {
                    "identity": "hand-calculated-v1",
                    "split": "test",
                    "sample_cap": 2,
                    "shuffle": False,
                }
            },
        )
    )
    projected = events.sum(dim=2, keepdim=True).expand(-1, -1, 2)
    expected_count = (projected * valid[:, :, None]).sum(dim=0)
    expected_seconds = valid.sum(dim=0).to(projected.dtype)[:, None] * 0.1
    torch.testing.assert_close(
        result.outputs["class_scores"],
        expected_count / expected_seconds,
        rtol=0,
        atol=0,
    )
    protocol = result.metrics["execution_protocol"]
    assert protocol["schema"] == "tools/snn.execution-protocol/v1"
    assert protocol["binding_schema"] == "tools/snn.dense-array-binding/v1"
    assert protocol["dataset"] == {
        "identity": "hand-calculated-v1",
        "split": "test",
        "sample_cap": 2,
        "shuffle": False,
        "batch_size": 2,
    }
    assert protocol["timing"] == {"dt_ms": 100.0, "steps": 3, "duration_ms": 300.0}
    assert protocol["masks"] == ["valid"]
    assert protocol["seeds"] == {"execution": 17}


@pytest.mark.parametrize(
    ("bindings", "message"),
    [
        ((DenseArrayBinding("events", torch.zeros(3, 1, 2)),), "missing=['valid']"),
        (
            (
                DenseArrayBinding("events", torch.zeros(3, 1, 2)),
                DenseArrayBinding("valid", torch.ones(3, 2, dtype=torch.bool)),
            ),
            "leading shape expected (3, 1)",
        ),
        (
            (
                DenseArrayBinding("events", torch.zeros(3, 1, 3)),
                DenseArrayBinding("valid", torch.ones(3, 1, dtype=torch.bool)),
            ),
            "trailing shape expected (2,)",
        ),
        (
            (
                DenseArrayBinding("events", torch.full((3, 1, 2), 0.5)),
                DenseArrayBinding("valid", torch.ones(3, 1, dtype=torch.bool)),
            ),
            "spike values must be boolean or zero/one",
        ),
    ],
)
def test_dense_array_bindings_fail_closed_on_contract_mismatch(bindings, message):
    with pytest.raises(ValueError) as exc:
        resolve_dense_array_bindings(
            _standard_readout_graph("rate", mask=True), bindings=bindings
        )
    assert message in str(exc.value)


def test_dense_array_file_loader_and_cli_emit_replayable_protocol(tmp_path):
    graph = _standard_readout_graph("count")
    bundle = snn.compiler.Bundle(
        graph=graph,
        training=None,
        manifest={
            "schema": "snnlang.bundle/v1",
            "graph_digest": snn.compiler.digest(graph),
            "files": [{"path": "graph.json", "digest": snn.compiler.digest(graph)}],
            "assets": [],
            "compiler": {"name": "test", "version": "1"},
            "target": None,
        },
        diagnostics=[],
        asset_sources={},
    ).write(tmp_path / "graph.bundle")
    input_file = tmp_path / "dense.npz"
    events = np.array([[[1.0, 0.0]], [[0.0, 1.0]]], dtype=np.float32)
    np.savez(input_file, events=events)
    bindings = load_dense_array_bindings(input_file, graph)
    assert len(bindings) == 1 and bindings[0].input_id == "events"
    assert bindings[0].source["digest"].startswith("sha256:")
    out = tmp_path / "out"
    assert (
        main(
            [
                "sim",
                "--executor",
                "graph",
                "--bundle",
                str(bundle),
                "--input-file",
                str(input_file),
                "--input-dataset-id",
                "fixture-snapshot-v1",
                "--input-split",
                "test",
                "--no-input-shuffle",
                "--seed",
                "23",
                "--out-dir",
                str(out),
            ]
        )
        == 0
    )
    metrics = json.loads((out / "metrics.json").read_text())
    protocol = metrics["execution_protocol"]
    assert protocol["dataset"] == {
        "identity": "fixture-snapshot-v1",
        "split": "test",
        "shuffle": False,
        "sample_cap": 1,
        "batch_size": 1,
    }
    assert protocol["timing"] == {"dt_ms": 100.0, "steps": 2, "duration_ms": 200.0}
    assert protocol["seeds"] == {"execution": 23}
    assert protocol["inputs"][0]["source"]["digest"].startswith("sha256:")


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
    result = simulate(
        ExecutionSpec(kind="simulate", executor="graph", graph=graph, inputs=inputs)
    )
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
        "ping",
        w_in=(0.2, 0.03),
        w_in_initial_zero_fraction=0.0,
        w_ei=(0.5, 0.05),
        w_ie=(1.0, 0.1),
        ei_strength=0.5,
        ei_ratio=2.0,
        recurrent_initial_zero_fraction=0.0,
        hidden_sizes=[256],
        readout_mode="mem-mean",
    )
    graph_model = build(
        ExecutionSpec(kind="build", executor="graph", graph=graph, seed=17)
    ).model
    assert isinstance(graph_model, GraphExecutor)
    mapping = {
        "sensory_ping_input.weight": legacy.W_ff[0],
        "classifier_projection.weight": legacy.W_ff[1],
        "sensory_ping_E_to_I.weight": legacy.W_ei["1"],
        "sensory_ping_I_to_E.weight": legacy.W_ie["1"],
    }
    for name, expected in mapping.items():
        torch.testing.assert_close(
            graph_model.parameter_map()[name], expected, rtol=0, atol=0
        )

    spikes = torch.zeros(12, 2, 784)
    spikes[0::2, :, :48] = 1.0
    legacy.recording = True
    legacy_logits = legacy(input_spikes=spikes)
    native = graph_model({"image": spikes}, record=True)
    torch.testing.assert_close(
        native.outputs["class_logits"], legacy_logits, rtol=0, atol=2e-7
    )
    torch.testing.assert_close(
        native.recordings["cell_0"], legacy.spike_record["hid"], rtol=0, atol=0
    )
    torch.testing.assert_close(
        native.recordings["cell_1"], legacy.spike_record["inh"], rtol=0, atol=0
    )


def _continuation_case():
    graph = _coupled_graph(delay_ms=0.3)
    for projection in graph["projections"]:
        if projection["id"] in {"a_input", "b_input"}:
            projection["delay"] = {"value": 0.3, "unit": "ms"}
    inputs = {
        "drive_a": torch.zeros(1000, 2, 3),
        "drive_b": torch.zeros(1000, 2, 2),
    }
    inputs["drive_a"][::3, :, :] = 1.0
    inputs["drive_b"][1::4, :, :] = 1.0
    return graph, inputs


def _assert_state_equal(left: GraphRuntimeState, right: GraphRuntimeState):
    assert left.signature == right.signature
    assert left.completed_steps == right.completed_steps
    for group in (
        "voltages",
        "refractory",
        "conductances",
        "population_histories",
        "input_histories",
    ):
        left_values = getattr(left, group)
        right_values = getattr(right, group)
        assert left_values.keys() == right_values.keys()
        for name in left_values:
            torch.testing.assert_close(
                left_values[name], right_values[name], rtol=0, atol=0
            )


def test_split_run_exactly_preserves_spikes_voltages_conductances_and_dynamic_state():
    graph, inputs = _continuation_case()
    whole = simulate(
        ExecutionSpec(
            kind="simulate", executor="graph", graph=graph, inputs=inputs, seed=11
        )
    )
    first = simulate(
        ExecutionSpec(
            kind="simulate",
            executor="graph",
            graph=graph,
            inputs={name: value[:500] for name, value in inputs.items()},
            seed=11,
        )
    )
    assert first.runtime_state is not None
    # The split occurs with live synapses, delayed events and refractory cells.
    assert any(
        torch.count_nonzero(x) for x in first.runtime_state.conductances.values()
    )
    assert any(
        torch.count_nonzero(x)
        for x in first.runtime_state.population_histories.values()
    )
    assert any(
        torch.count_nonzero(x) for x in first.runtime_state.input_histories.values()
    )
    assert any(torch.count_nonzero(x) for x in first.runtime_state.refractory.values())
    second = simulate(
        ExecutionSpec(
            kind="simulate",
            executor="graph",
            graph=graph,
            inputs={name: value[500:] for name, value in inputs.items()},
            seed=11,
        ),
        runtime_state=first.runtime_state,
    )
    for name, expected in whole.recordings.items():
        actual = torch.cat((first.recordings[name], second.recordings[name]))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert whole.runtime_state is not None and second.runtime_state is not None
    _assert_state_equal(second.runtime_state, whole.runtime_state)
    for name in whole.final_state:
        torch.testing.assert_close(
            second.final_state[name], whole.final_state[name], rtol=0, atol=0
        )


def test_runtime_state_portable_round_trip_preserves_dtype_and_values(tmp_path):
    graph, inputs = _continuation_case()
    result = simulate(
        ExecutionSpec(
            kind="simulate",
            executor="graph",
            graph=graph,
            inputs={name: value[:13] for name, value in inputs.items()},
            seed=2,
        )
    )
    assert result.runtime_state is not None
    root = save_runtime_state(tmp_path / "state", result.runtime_state)
    loaded = load_runtime_state(root, device="cpu")
    _assert_state_equal(loaded, result.runtime_state)


def test_runtime_state_allows_weight_branch_but_rejects_structural_changes():
    graph, inputs = _continuation_case()
    zero_parameter = next(
        row for row in graph["parameters"] if row["id"] == "a_I_to_b_E.weight"
    )
    zero_parameter["initializer"] = {"kind": "constant", "value": 0.0}
    first = simulate(
        ExecutionSpec(
            kind="simulate",
            executor="graph",
            graph=graph,
            inputs={name: value[:12] for name, value in inputs.items()},
            seed=5,
        )
    )
    assert first.runtime_state is not None
    branch = copy.deepcopy(graph)
    cross_parameter = next(
        row for row in branch["parameters"] if row["id"] == "a_I_to_b_E.weight"
    )
    cross_parameter["initializer"] = {"kind": "constant", "value": 9.0}
    assert runtime_state_signature(plan_graph(branch)) == first.runtime_state.signature
    simulate(
        ExecutionSpec(
            kind="simulate",
            executor="graph",
            graph=branch,
            inputs={name: value[12:] for name, value in inputs.items()},
            seed=5,
        ),
        runtime_state=first.runtime_state,
    )

    incompatible_graphs = []
    resized = copy.deepcopy(graph)
    next(row for row in resized["populations"] if row["id"] == "a_E")["size"] += 1
    incompatible_graphs.append(resized)
    new_dt = copy.deepcopy(graph)
    new_dt["timebase"]["dt"]["value"] = 0.05
    incompatible_graphs.append(new_dt)
    new_delay = copy.deepcopy(graph)
    next(row for row in new_delay["projections"] if row["id"] == "a_I_to_b_E")["delay"][
        "value"
    ] = 0.4
    incompatible_graphs.append(new_delay)
    new_synapse = copy.deepcopy(graph)
    next(row for row in new_synapse["projections"] if row["id"] == "a_I_to_b_E")[
        "synapse"
    ]["tau"]["value"] = 8.0
    incompatible_graphs.append(new_synapse)
    missing_projection = copy.deepcopy(graph)
    missing_projection["projections"] = [
        row for row in missing_projection["projections"] if row["id"] != "a_I_to_b_E"
    ]
    incompatible_graphs.append(missing_projection)
    renamed_projection = copy.deepcopy(graph)
    next(row for row in renamed_projection["projections"] if row["id"] == "a_I_to_b_E")[
        "id"
    ] = "renamed"
    incompatible_graphs.append(renamed_projection)
    reshaped_parameter = copy.deepcopy(graph)
    next(
        row
        for row in reshaped_parameter["parameters"]
        if row["id"] == "a_I_to_b_E.weight"
    )["shape"][0] += 1
    incompatible_graphs.append(reshaped_parameter)
    for incompatible in incompatible_graphs:
        try:
            simulate(
                ExecutionSpec(
                    kind="simulate",
                    executor="graph",
                    graph=incompatible,
                    inputs={name: value[12:] for name, value in inputs.items()},
                    seed=5,
                ),
                runtime_state=first.runtime_state,
            )
        except ValueError as exc:
            assert "runtime state is incompatible" in str(exc)
            assert "graph." in str(exc)
        else:
            raise AssertionError("structurally incompatible runtime state must fail")


def test_runtime_state_validates_batch_shape_and_dtype():
    graph, inputs = _continuation_case()
    first = simulate(
        ExecutionSpec(
            kind="simulate",
            executor="graph",
            graph=graph,
            inputs={name: value[:10] for name, value in inputs.items()},
            seed=8,
        )
    )
    assert first.runtime_state is not None
    wrong_batch = {name: value[10:, :1] for name, value in inputs.items()}
    try:
        simulate(
            ExecutionSpec(
                kind="simulate",
                executor="graph",
                graph=graph,
                inputs=wrong_batch,
                seed=8,
            ),
            runtime_state=first.runtime_state,
        )
    except ValueError as exc:
        assert "shape expected" in str(exc)
    else:
        raise AssertionError("runtime-state batch mismatch must fail")
    bad_state = first.runtime_state.detached()
    bad_state.voltages["a_E"] = bad_state.voltages["a_E"].double()
    try:
        simulate(
            ExecutionSpec(
                kind="simulate",
                executor="graph",
                graph=graph,
                inputs={name: value[10:] for name, value in inputs.items()},
                seed=8,
            ),
            runtime_state=bad_state,
        )
    except ValueError as exc:
        assert "dtype expected" in str(exc)
    else:
        raise AssertionError("runtime-state dtype mismatch must fail")


def test_graph_cli_runtime_state_round_trip_and_legacy_rejection(tmp_path):
    graph, inputs = _continuation_case()
    bundle = snn.compiler.Bundle(
        graph=graph,
        training=None,
        manifest={
            "schema": "snnlang.bundle/v1",
            "graph_digest": snn.compiler.digest(graph),
            "files": [{"path": "graph.json", "digest": snn.compiler.digest(graph)}],
            "assets": [],
            "compiler": {"name": "test", "version": "1"},
            "target": None,
        },
        diagnostics=[],
        asset_sources={},
    ).write(tmp_path / "graph.bundle")
    first_inputs = tmp_path / "first.npz"
    second_inputs = tmp_path / "second.npz"
    np.savez(
        first_inputs, **{name: value[:19].numpy() for name, value in inputs.items()}
    )
    np.savez(
        second_inputs, **{name: value[19:].numpy() for name, value in inputs.items()}
    )
    state_one = tmp_path / "state-one"
    state_two = tmp_path / "state-two"
    assert (
        main(
            [
                "sim",
                "--executor",
                "graph",
                "--bundle",
                str(bundle),
                "--input-file",
                str(first_inputs),
                "--out-dir",
                str(tmp_path / "first-out"),
                "--save-runtime-state",
                str(state_one),
            ]
        )
        == 0
    )
    with np.load(tmp_path / "first-out" / "parameters.npz") as parameters:
        expected = {row["id"] for row in graph["parameters"]}
        assert set(parameters.files) == expected
        for row in graph["parameters"]:
            assert parameters[row["id"]].size == int(np.prod(row["shape"]))
    assert (
        main(
            [
                "sim",
                "--executor",
                "graph",
                "--bundle",
                str(bundle),
                "--input-file",
                str(second_inputs),
                "--out-dir",
                str(tmp_path / "second-out"),
                "--load-runtime-state",
                str(state_one),
                "--save-runtime-state",
                str(state_two),
            ]
        )
        == 0
    )
    assert load_runtime_state(state_two).completed_steps == 1000
    try:
        main(
            [
                "sim",
                "--load-runtime-state",
                str(state_one),
                "--out-dir",
                str(tmp_path / "legacy-out"),
            ]
        )
    except SystemExit as exc:
        assert "require --executor graph" in str(exc)
    else:
        raise AssertionError("legacy executor must reject graph-runtime-state flags")
