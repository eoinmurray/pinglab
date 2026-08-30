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
from conformance import canonical_json_tensor, compare_conformance_layers
from execution import (
    DatasetEncoder,
    DatasetSnapshotBinding,
    DelayBuffer,
    DenseArrayBinding,
    EventStreamBinding,
    ExecutionSpec,
    GraphExecutor,
    GraphRuntimeState,
    PoissonInputBinding,
    TargetArrayBinding,
    build,
    capture_training_rng_state,
    derive_inference_products,
    execute_request,
    execution_spec_from_args,
    export_legacy_parameters_v1,
    graph_capability_issues,
    import_legacy_parameters_v1,
    legacy_parameter_map_v1,
    load_dense_array_bindings,
    load_event_stream_bindings,
    load_runtime_state,
    load_target_array_bindings,
    load_training_checkpoint,
    plan_graph,
    resolve_dataset_snapshot_binding,
    resolve_dense_array_bindings,
    resolve_device,
    resolve_event_stream_bindings,
    resolve_input_bindings,
    resolve_poisson_input_bindings,
    resolve_target_array_bindings,
    restore_training_rng_state,
    runtime_state_signature,
    save_runtime_state,
    save_training_checkpoint,
    simulate,
    train,
    validate_derived_inference_products,
    validate_inference_artifacts,
    write_inference_artifacts,
)
from tool import main, parse_args

from tools import snnlang as snn
from tools.snnlang import training
from tools.snnlang.examples.build_examples import deep_network, ping_classifier


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


def test_typed_request_defaults_to_legacy_and_graph_training_requires_recipe():
    request = ExecutionSpec(kind="build")
    assert request.executor == "legacy"
    assert request.device == "auto"
    assert build(request).metrics["routing"] == "legacy"
    try:
        train(ExecutionSpec(kind="train", executor="graph", graph=_coupled_graph()))
    except ValueError as exc:
        assert "training recipe" in str(exc)
    else:
        raise AssertionError("graph training must require an explicit recipe")


def _direct_train_bundle():
    net = snn.Network("direct_train", dt=0.1 * snn.ms)
    events = net.input(
        "events", shape=("time", "batch", 2), signal_type="spikes", unit="spike"
    )
    scores = snn.readouts.SpikeCount(source=events, classes=2, name="scores")
    snn.ops.linear(events, size=1, name="shadow")
    net.output("class_scores", scores)
    parameter_ids = [row["id"] for row in net.parameters]
    recipe = snn.TrainSpec(
        objectives=[training.CrossEntropy(prediction=scores, target="label")],
        parameter_groups=[
            training.ParameterGroup(
                ["scores_projection.weight"], name="readout", lr=0.1
            ),
            training.ParameterGroup(
                [pid for pid in parameter_ids if pid != "scores_projection.weight"],
                name="frozen",
                lr=0,
                frozen=True,
            ),
        ],
        optimizer=training.AdamW(weight_decay=0.0),
        presentation_duration=0.3 * snn.ms,
    )
    return snn.compile(net, training=recipe, target=None)


def test_graph_training_updates_named_parameters_and_optimizer_state_deterministically():
    bundle = _direct_train_bundle()
    inputs = torch.zeros(3, 2, 2)
    inputs[:, 0, 0] = 1
    inputs[:, 1, 1] = 1
    spec = ExecutionSpec(
        kind="train",
        executor="graph",
        graph=bundle.graph,
        training=bundle.training,
        inputs={"events": inputs},
        targets={"label": torch.tensor([0, 1])},
        seed=17,
        options={"updates": 5},
    )
    first = train(spec)
    second = train(spec)
    losses = [row["loss"] for row in first.metrics["updates"]]
    assert losses[-1] < losses[0]
    assert first.metrics["trainable_parameters"] == ["scores_projection.weight"]
    assert set(first.optimizer_state) == {"scores_projection.weight"}
    assert set(first.gradients) == {"scores_projection.weight"}
    assert torch.count_nonzero(first.parameters["shadow.weight"]) == 0
    torch.testing.assert_close(
        first.parameters["scores_projection.weight"],
        second.parameters["scores_projection.weight"],
        rtol=0,
        atol=0,
    )
    assert first.metrics["updates"] == second.metrics["updates"]


def test_graph_one_step_matches_direct_pytorch_gradient_weight_and_adamw_state():
    bundle = _direct_train_bundle()
    inputs = torch.zeros(3, 2, 2)
    inputs[:, 0, 0] = 1
    inputs[:, 1, 1] = 1
    labels = torch.tensor([0, 1])
    result = train(
        ExecutionSpec(
            kind="train",
            executor="graph",
            graph=bundle.graph,
            training=bundle.training,
            inputs={"events": inputs},
            targets={"label": labels},
            seed=17,
        )
    )
    direct = torch.nn.Parameter(torch.zeros(2, 2))
    optimizer = torch.optim.AdamW([direct], lr=0.1, weight_decay=0.0)
    loss = torch.nn.functional.cross_entropy((inputs @ direct).sum(dim=0), labels)
    loss.backward()
    expected_gradient = direct.grad.detach().clone()
    optimizer.step()
    torch.testing.assert_close(
        result.gradients["scores_projection.weight"], expected_gradient, rtol=0, atol=0
    )
    torch.testing.assert_close(
        result.parameters["scores_projection.weight"], direct.detach(), rtol=0, atol=0
    )
    for key in ("step", "exp_avg", "exp_avg_sq"):
        torch.testing.assert_close(
            result.optimizer_state["scores_projection.weight"][key],
            optimizer.state[direct][key],
            rtol=0,
            atol=0,
        )


def test_graph_training_authenticates_recipe_from_bundle(tmp_path):
    bundle = _direct_train_bundle()
    root = bundle.write(tmp_path / "train.bundle")
    inputs = torch.zeros(3, 2, 2)
    inputs[:, 0, 0] = 1
    inputs[:, 1, 1] = 1
    result = train(
        ExecutionSpec(
            kind="train",
            executor="graph",
            bundle=root,
            inputs={"events": inputs},
            targets={"label": torch.tensor([0, 1])},
            seed=17,
        )
    )
    assert result.metrics["training_schema"] == "snnlang.training/v1"
    assert result.metrics["updates"][0]["loss"] == pytest.approx(0.6931471824645996)


def test_graph_training_backpropagates_through_recurrence_and_spike_budget():
    net = snn.Network("tiny_recurrent_train", dt=0.1 * snn.ms)
    events = net.input(
        "events", shape=("time", "batch", 2), signal_type="spikes", unit="spike"
    )
    cell = snn.components.ping(net, name="cell", n_e=4, n_i=2, source=events)
    input_parameter = next(
        row for row in net.parameters if row["id"] == "cell_input.weight"
    )
    input_parameter["initializer"] = snn.Constant(100.0).json()
    scores = snn.readouts.SpikeCount(source=cell.E.spikes, classes=2, name="scores")
    net.output("class_scores", scores)
    parameter_ids = [row["id"] for row in net.parameters]
    recipe = snn.TrainSpec(
        objectives=[training.CrossEntropy(prediction=scores, target="label")],
        parameter_groups=[training.ParameterGroup(parameter_ids, name="all", lr=1e-3)],
        optimizer=training.AdamW(weight_decay=0.0),
        regularizers=[
            training.SpikeBudgetPenalty(
                signals=(cell.E.spikes, cell.I.spikes),
                ceiling_hz=0.0,
                strength=0.01,
            )
        ],
        surrogate=training.FastSigmoid(slope=1.0),
        presentation_duration=3 * snn.ms,
    )
    bundle = snn.compile(net, training=recipe, target=None)
    result = train(
        ExecutionSpec(
            kind="train",
            executor="graph",
            graph=bundle.graph,
            training=bundle.training,
            inputs={"events": torch.ones(30, 2, 2)},
            targets={"label": torch.tensor([0, 1])},
            seed=3,
        )
    )
    assert "cell_E_to_I.weight" in result.gradients
    assert "cell_I_to_E.weight" in result.gradients
    e_rates = result.recordings["cell_E.spikes"].sum(dim=0).mean(dim=1) / 0.003
    i_rates = result.recordings["cell_I.spikes"].sum(dim=0).mean(dim=1) / 0.003
    expected = 0.01 * torch.stack((e_rates.square(), i_rates.square())).mean()
    assert result.metrics["updates"][0]["components"][
        "regularizer[0]"
    ] == pytest.approx(float(expected.detach()))


def test_training_checkpoint_round_trip_and_resume_are_exact(tmp_path):
    bundle = _direct_train_bundle()
    inputs = torch.zeros(3, 2, 2)
    inputs[:, 0, 0] = 1
    inputs[:, 1, 1] = 1
    common = dict(
        kind="train",
        executor="graph",
        graph=bundle.graph,
        training=bundle.training,
        inputs={"events": inputs},
        targets={"label": torch.tensor([0, 1])},
        seed=17,
    )
    uninterrupted = train(ExecutionSpec(**common, options={"updates": 4}))
    checkpoint_dir = tmp_path / "checkpoint"
    first_half = train(
        ExecutionSpec(
            **common,
            options={
                "updates": 2,
                "save_final_checkpoint": checkpoint_dir,
                "save_selected_checkpoint": tmp_path / "selected",
            },
        )
    )
    loaded = load_training_checkpoint(checkpoint_dir)
    assert loaded.completed_updates == 2
    assert loaded.graph_digest == bundle.training["graph_digest"]
    assert (tmp_path / "selected" / "manifest.json").is_file()
    resumed = train(
        ExecutionSpec(
            **common,
            checkpoint=checkpoint_dir,
            options={"updates": 2},
        )
    )
    assert resumed.metrics["resumed_from_update"] == 2
    assert [row["update"] for row in resumed.metrics["updates"]] == [3, 4]
    assert resumed.metrics["updates"] == uninterrupted.metrics["updates"][2:]
    assert uninterrupted.training_checkpoint is not None
    assert resumed.training_checkpoint is not None
    for name in uninterrupted.parameters:
        torch.testing.assert_close(
            resumed.parameters[name], uninterrupted.parameters[name], rtol=0, atol=0
        )
    conformance = compare_conformance_layers(
        "checkpoint-resume",
        {
            "topology": {"graph": canonical_json_tensor(bundle.graph)},
            "initialization": {
                "metadata": canonical_json_tensor(
                    uninterrupted.metrics["initialization"]
                )
            },
            "parameters": uninterrupted.parameters,
            "gradients": uninterrupted.gradients,
            "outputs": uninterrupted.outputs,
            "optimizer": {
                f"{name}.{state}": value
                for name, values in uninterrupted.optimizer_state.items()
                for state, value in values.items()
                if isinstance(value, torch.Tensor)
            },
            "checkpoint": {
                "data_state": canonical_json_tensor(
                    uninterrupted.training_checkpoint.data_state
                )
            },
        },
        {
            "topology": {"graph": canonical_json_tensor(bundle.graph)},
            "initialization": {
                "metadata": canonical_json_tensor(resumed.metrics["initialization"])
            },
            "parameters": resumed.parameters,
            "gradients": resumed.gradients,
            "outputs": resumed.outputs,
            "optimizer": {
                f"{name}.{state}": value
                for name, values in resumed.optimizer_state.items()
                for state, value in values.items()
                if isinstance(value, torch.Tensor)
            },
            "checkpoint": {
                "data_state": canonical_json_tensor(
                    resumed.training_checkpoint.data_state
                )
            },
        },
    )
    conformance.require_passed()
    for name in uninterrupted.optimizer_state:
        for state in uninterrupted.optimizer_state[name]:
            torch.testing.assert_close(
                resumed.optimizer_state[name][state],
                uninterrupted.optimizer_state[name][state],
                rtol=0,
                atol=0,
            )
    assert first_half.training_checkpoint is not None
    assert resumed.training_checkpoint.completed_updates == 4


def test_accelerator_rng_checkpoint_round_trip_and_topology_restore(
    tmp_path, monkeypatch
):
    bundle = _direct_train_bundle()
    trained = train(
        ExecutionSpec(
            kind="train",
            executor="graph",
            graph=bundle.graph,
            training=bundle.training,
            inputs={"events": torch.zeros(3, 1, 2)},
            targets={"label": torch.tensor([0])},
        )
    )
    checkpoint = trained.training_checkpoint
    assert checkpoint is not None
    checkpoint.rng_backend = "cuda"
    checkpoint.accelerator_rng_states = {
        "cuda:0": torch.tensor([1, 2, 3], dtype=torch.uint8),
        "cuda:1": torch.tensor([4, 5, 6], dtype=torch.uint8),
    }
    root = save_training_checkpoint(tmp_path / "cuda-checkpoint", checkpoint)
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["schema_version"] == 2
    assert manifest["rng_backend"] == "cuda"
    assert manifest["accelerator_rng_devices"] == ["cuda:0", "cuda:1"]
    loaded = load_training_checkpoint(root)
    assert loaded.rng_backend == "cuda"
    assert set(loaded.accelerator_rng_states) == {"cuda:0", "cuda:1"}

    restored = []
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(
        torch.cuda, "set_rng_state_all", lambda states: restored.extend(states)
    )
    restore_training_rng_state(loaded, "cuda")
    assert [state.tolist() for state in restored] == [[1, 2, 3], [4, 5, 6]]
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(ValueError, match="topology"):
        restore_training_rng_state(loaded, "cuda")


def test_accelerator_rng_capture_names_every_cuda_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(
        torch.cuda,
        "get_rng_state_all",
        lambda: [
            torch.tensor([7], dtype=torch.uint8),
            torch.tensor([8], dtype=torch.uint8),
        ],
    )
    backend, states = capture_training_rng_state("cuda:1")
    assert backend == "cuda"
    assert {name: value.tolist() for name, value in states.items()} == {
        "cuda:0": [7],
        "cuda:1": [8],
    }


def test_v1_cpu_checkpoint_remains_loadable(tmp_path):
    bundle = _direct_train_bundle()
    trained = train(
        ExecutionSpec(
            kind="train",
            executor="graph",
            graph=bundle.graph,
            training=bundle.training,
            inputs={"events": torch.zeros(3, 1, 2)},
            targets={"label": torch.tensor([0])},
        )
    )
    assert trained.training_checkpoint is not None
    root = save_training_checkpoint(tmp_path / "v1", trained.training_checkpoint)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["schema_version"] = 1
    manifest.pop("rng_backend")
    manifest.pop("accelerator_rng_devices")
    (root / "manifest.json").write_text(json.dumps(manifest) + "\n")
    loaded = load_training_checkpoint(root)
    assert loaded.rng_backend == "cpu"
    assert loaded.accelerator_rng_states == {}


def test_training_checkpoint_rejects_partial_parameter_mapping(tmp_path):
    bundle = _direct_train_bundle()
    inputs = torch.zeros(3, 2, 2)
    result = train(
        ExecutionSpec(
            kind="train",
            executor="graph",
            graph=bundle.graph,
            training=bundle.training,
            inputs={"events": inputs},
            targets={"label": torch.tensor([0, 1])},
        )
    )
    checkpoint = result.training_checkpoint
    assert checkpoint is not None
    checkpoint.parameters.pop("shadow.weight")
    root = save_training_checkpoint(tmp_path / "partial", checkpoint)
    with pytest.raises(ValueError, match="parameter names mismatch"):
        train(
            ExecutionSpec(
                kind="train",
                executor="graph",
                graph=bundle.graph,
                training=bundle.training,
                inputs={"events": inputs},
                targets={"label": torch.tensor([0, 1])},
                checkpoint=root,
            )
        )


def test_graph_inference_loads_portable_selected_checkpoint_with_provenance(tmp_path):
    bundle = _direct_train_bundle()
    inputs = torch.zeros(3, 2, 2)
    inputs[:, 0, 0] = 1
    inputs[:, 1, 1] = 1
    selected_path = tmp_path / "selected"
    trained = train(
        ExecutionSpec(
            kind="train",
            executor="graph",
            graph=bundle.graph,
            training=bundle.training,
            inputs={"events": inputs},
            targets={"label": torch.tensor([0, 1])},
            seed=17,
            options={"updates": 3, "save_selected_checkpoint": selected_path},
        )
    )
    selected = load_training_checkpoint(selected_path)
    inferred = simulate(
        ExecutionSpec(
            kind="simulate",
            executor="graph",
            graph=bundle.graph,
            inputs={"events": inputs},
            seed=17,
            checkpoint=selected_path,
        )
    )
    for name in selected.parameters:
        torch.testing.assert_close(
            inferred.parameters[name], selected.parameters[name], rtol=0, atol=0
        )
    assert inferred.metrics["checkpoint"] == {
        "format": "tools/snnsim.training-checkpoint/v1",
        "path": str(selected_path),
        "graph_digest": selected.graph_digest,
        "training_digest": selected.training_digest,
        "completed_updates": selected.completed_updates,
        "selected_loss": selected.selected_loss,
    }
    assert trained.selected_checkpoint is not None

    incompatible = copy.deepcopy(bundle.graph)
    incompatible["name"] = "different"
    with pytest.raises(ValueError, match="inference checkpoint graph digest"):
        simulate(
            ExecutionSpec(
                kind="simulate",
                executor="graph",
                graph=incompatible,
                inputs={"events": inputs},
                checkpoint=selected_path,
            )
        )


def test_dataset_training_resume_preserves_shuffle_and_batch_position(tmp_path):
    bundle = _direct_train_bundle()
    inputs = torch.zeros(3, 5, 2)
    for sample in range(5):
        inputs[:, sample, sample % 2] = 1
    common = dict(
        kind="train",
        executor="graph",
        graph=bundle.graph,
        training=bundle.training,
        inputs={"events": inputs},
        targets={"label": torch.tensor([0, 1, 0, 1, 0])},
        seed=23,
    )
    trajectory = {"epochs": 2, "batch_size": 2, "shuffle": True}
    uninterrupted = train(ExecutionSpec(**common, options=trajectory))
    checkpoint_dir = tmp_path / "dataset-checkpoint"
    first = train(
        ExecutionSpec(
            **common,
            options={
                **trajectory,
                "updates": 1,
                "save_final_checkpoint": checkpoint_dir,
            },
        )
    )
    assert first.training_checkpoint is not None
    assert first.training_checkpoint.data_state == {"epoch": 0, "batch": 1}
    resumed = train(
        ExecutionSpec(
            **common,
            checkpoint=checkpoint_dir,
            options=trajectory,
        )
    )
    assert resumed.metrics["updates"] == uninterrupted.metrics["updates"][1:]
    assert resumed.training_checkpoint is not None
    assert resumed.training_checkpoint.data_state == {"epoch": 2, "batch": 0}
    for name in uninterrupted.parameters:
        torch.testing.assert_close(
            resumed.parameters[name], uninterrupted.parameters[name], rtol=0, atol=0
        )


def test_target_array_binding_is_named_and_digest_bearing(tmp_path):
    bundle = _direct_train_bundle()
    path = tmp_path / "targets.npy"
    np.save(path, np.array([0, 1, 1], dtype=np.int64))
    bindings = load_target_array_bindings(path, bundle.training)
    assert bindings[0].target_id == "label"
    assert bindings[0].source["digest"].startswith("sha256:")
    resolved, rows = resolve_target_array_bindings(
        bundle.training, bindings=bindings, sample_count=3
    )
    assert resolved["label"].tolist() == [0, 1, 1]
    assert rows[0]["source"]["path"] == str(path)
    with pytest.raises(ValueError, match="target ids do not match recipe"):
        resolve_target_array_bindings(
            bundle.training,
            bindings=(TargetArrayBinding("wrong", torch.tensor([0, 1, 1])),),
            sample_count=3,
        )


def test_graph_training_cli_loads_targets_and_writes_resume_checkpoint(tmp_path):
    bundle = _direct_train_bundle().write(tmp_path / "train.bundle")
    inputs = np.zeros((3, 4, 2), dtype=np.float32)
    inputs[:, ::2, 0] = 1
    inputs[:, 1::2, 1] = 1
    input_path = tmp_path / "inputs.npy"
    target_path = tmp_path / "targets.npy"
    np.save(input_path, inputs)
    np.save(target_path, np.array([0, 1, 0, 1], dtype=np.int64))
    checkpoint = tmp_path / "checkpoint"
    out_dir = tmp_path / "run"
    assert (
        main(
            [
                "train",
                "--executor",
                "graph",
                "--bundle",
                str(bundle),
                "--input-file",
                str(input_path),
                "--target-file",
                str(target_path),
                "--batch-size",
                "2",
                "--input-shuffle",
                "--save-final-checkpoint",
                str(checkpoint),
                "--out-dir",
                str(out_dir),
            ]
        )
        == 0
    )
    metrics = json.loads((out_dir / "metrics.json").read_text())
    assert len(metrics["updates"]) == 2
    assert metrics["execution_protocol"]["targets"][0]["source"]["digest"].startswith(
        "sha256:"
    )
    assert load_training_checkpoint(checkpoint).data_state == {
        "epoch": 1,
        "batch": 0,
    }


def test_graph_training_cli_resolves_portable_dataset_snapshot(tmp_path):
    bundle = _direct_train_bundle().write(tmp_path / "train.bundle")
    snapshot = tmp_path / "dataset.npz"
    features = np.zeros((3, 4, 2), dtype=np.uint8)
    features[:, ::2, 0] = 1
    features[:, 1::2, 1] = 1
    np.savez(snapshot, features=features, labels=np.asarray([0, 1, 0, 1]))
    out_dir = tmp_path / "run"
    assert (
        main(
            [
                "train",
                "--executor",
                "graph",
                "--bundle",
                str(bundle),
                "--dataset-file",
                str(snapshot),
                "--dataset-encoder",
                "prebinned-spikes",
                "--dataset-target-id",
                "label",
                "--input-dataset-id",
                "fixture-v1",
                "--input-split",
                "train",
                "--max-samples",
                "3",
                "--input-shuffle",
                "--batch-size",
                "2",
                "--out-dir",
                str(out_dir),
            ]
        )
        == 0
    )
    protocol = json.loads((out_dir / "metrics.json").read_text())["execution_protocol"]
    assert protocol["representation"] == "dataset_snapshot"
    assert protocol["dataset"] == {
        "identity": "fixture-v1",
        "split": "train",
        "sample_cap": 3,
        "batch_size": 2,
        "shuffle": True,
    }
    assert protocol["dataset_binding"]["encoder"]["kind"] == "prebinned_spikes"


def test_legacy_parameter_map_uses_complete_semantic_names():
    graph = ping_classifier().graph
    assert legacy_parameter_map_v1(graph) == {
        "classifier_projection.weight": "W_ff.1",
        "sensory_ping_E_to_E.weight": "W_ee.1",
        "sensory_ping_E_to_I.weight": "W_ei.1",
        "sensory_ping_I_to_E.weight": "W_ie.1",
        "sensory_ping_I_to_I.weight": "W_ii.1",
        "sensory_ping_input.weight": "W_ff.0",
    }


def test_legacy_parameter_interchange_round_trips_exactly_and_rejects_partial():
    graph = ping_classifier().graph
    built = build(ExecutionSpec(kind="build", executor="graph", graph=graph, seed=9))
    exported = export_legacy_parameters_v1(graph, built.parameters)
    assert exported.provenance["direction"] == "graph_to_legacy"
    assert set(exported.parameters) == {
        "W_ff.0",
        "W_ff.1",
        "W_ee.1",
        "W_ei.1",
        "W_ie.1",
        "W_ii.1",
    }
    imported = import_legacy_parameters_v1(graph, exported.parameters)
    assert imported.provenance["direction"] == "legacy_to_graph"
    for name in built.parameters:
        torch.testing.assert_close(
            imported.parameters[name], built.parameters[name], rtol=0, atol=0
        )
    partial = dict(exported.parameters)
    partial.pop("W_ii.1")
    with pytest.raises(ValueError, match="exact keys"):
        import_legacy_parameters_v1(graph, partial)


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


def test_graph_cli_accepts_ordered_intervention_syntax():
    args = parse_args(
        [
            "sim",
            "--intervention",
            "drop:cell_E=0.25",
            "--intervention",
            "add:cell_E=5",
            "--inference-timestep-ms",
            "0.05",
        ]
    )
    assert args.intervention == ["drop:cell_E=0.25", "add:cell_E=5"]
    assert args.inference_timestep_ms == 0.05


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


def test_production_shaped_mnist_and_shd_graphs_execute_named_outputs():
    mnist = ping_classifier()
    shd = deep_network()
    assert mnist.graph["inputs"][0]["shape"] == ["time", "batch", 784]
    assert shd.graph["inputs"][0]["shape"] == ["time", "batch", 700]
    assert len(shd.graph["populations"]) == 6
    assert [row["target"] for row in shd.training["objectives"]] == ["gesture"]
    mnist_result = simulate(
        ExecutionSpec(
            kind="simulate",
            executor="graph",
            graph=mnist.graph,
            inputs={"image": torch.zeros(2, 1, 784)},
            recording="observables",
            seed=5,
        )
    )
    shd_result = simulate(
        ExecutionSpec(
            kind="simulate",
            executor="graph",
            graph=shd.graph,
            inputs={"events": torch.zeros(2, 1, 700)},
            recording="observables",
            seed=5,
        )
    )
    assert mnist_result.outputs["class_logits"].shape == (1, 10)
    assert shd_result.outputs["gesture_logits"].shape == (1, 20)
    assert set(shd_result.recordings) == {
        "association_E_spikes",
        "decision_E_spikes",
        "encoder_E_spikes",
    }


def test_production_shaped_deep_shd_recipe_trains_all_recurrent_layers():
    bundle = deep_network()
    result = train(
        ExecutionSpec(
            kind="train",
            executor="graph",
            graph=bundle.graph,
            training=bundle.training,
            inputs={"events": torch.zeros(2, 1, 700)},
            targets={"gesture": torch.tensor([0])},
            seed=9,
        )
    )
    assert set(result.gradients) == set(
        bundle.training["resolved_parameters"]["trainable"]
    )
    assert {
        "encoder_E_to_I.weight",
        "encoder_I_to_E.weight",
        "association_E_to_I.weight",
        "association_I_to_E.weight",
        "decision_E_to_I.weight",
        "decision_I_to_E.weight",
    } <= set(result.gradients)
    assert result.metrics["updates"][0]["components"]["regularizer[0]"] == 0.0


def test_production_ping_fine_timestep_and_variable_rate_protocol():
    bundle = ping_classifier()
    result = simulate(
        ExecutionSpec(
            kind="simulate",
            executor="graph",
            graph=bundle.graph,
            poisson_bindings=(
                PoissonInputBinding(
                    "image", 2, 3, (0.0, 5.0, 25.0), 41, categorical=True
                ),
            ),
            recording="observables",
            seed=41,
            options={"inference_overrides": {"timestep_ms": 0.05}},
        )
    )
    protocol = result.metrics["execution_protocol"]
    assert protocol["timing"] == {
        "dt_ms": 0.05,
        "steps": 4,
        "duration_ms": pytest.approx(0.2),
    }
    assert set(protocol["inputs"][0]["realized_rates_hz"]) <= {0.0, 5.0, 25.0}
    assert result.outputs["class_logits"].shape == (3, 10)
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
    graph = snn.compile(net, target="tools/snnsim").graph
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
    assert protocol["schema"] == "tools/snnsim.execution-protocol/v1"
    assert protocol["binding_schema"] == "tools/snnsim.dense-array-binding/v1"
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


def test_event_stream_binding_matches_hand_calculated_spike_count():
    graph = _standard_readout_graph("count")
    binding = EventStreamBinding(
        "events",
        steps=torch.tensor([0, 1, 2, 2]),
        batches=torch.tensor([0, 0, 0, 1]),
        channels=torch.tensor([0, 1, 0, 1]),
        steps_count=3,
        batch_size=2,
        source={"kind": "fixture", "id": "events-v1"},
    )
    result = simulate(
        ExecutionSpec(
            kind="simulate",
            executor="graph",
            graph=graph,
            event_bindings=(binding,),
            seed=31,
            protocol={
                "dataset": {
                    "identity": "event-fixture-v1",
                    "split": "test",
                    "shuffle": False,
                }
            },
        )
    )
    torch.testing.assert_close(
        result.outputs["class_scores"],
        torch.tensor([[3.0, 3.0], [1.0, 1.0]]),
        rtol=0,
        atol=0,
    )
    protocol = result.metrics["execution_protocol"]
    assert protocol["binding_schema"] == "tools/snnsim.event-stream-binding/v1"
    assert protocol["representation"] == "event_stream"
    assert protocol["inputs"][0]["event_count"] == 4
    assert protocol["dataset"] == {
        "identity": "event-fixture-v1",
        "split": "test",
        "shuffle": False,
        "sample_cap": 2,
        "batch_size": 2,
    }
    assert protocol["timing"] == {
        "dt_ms": 100.0,
        "steps": 3,
        "duration_ms": 300.0,
    }
    assert protocol["resolution"] == {
        "coordinates": "zero_based_integer_steps",
        "ordering": "step,batch,channel",
        "duplicates": "reject",
        "materialization": "binary_dense",
    }


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"steps": torch.tensor([3])}, "step coordinates must be in [0, 3)"),
        (
            {
                "steps": torch.tensor([1, 0]),
                "batches": torch.tensor([0, 0]),
                "channels": torch.tensor([0, 1]),
            },
            "coordinates must be ordered",
        ),
        (
            {
                "steps": torch.tensor([0, 0]),
                "batches": torch.tensor([0, 0]),
                "channels": torch.tensor([1, 1]),
            },
            "contains duplicate coordinates",
        ),
        ({"channels": torch.tensor([2])}, "channel coordinates must be in [0, 2)"),
    ],
)
def test_event_stream_bindings_fail_closed(updates, message):
    values = {
        "steps": torch.tensor([0]),
        "batches": torch.tensor([0]),
        "channels": torch.tensor([0]),
    }
    values.update(updates)
    binding = EventStreamBinding("events", **values, steps_count=3, batch_size=1)
    with pytest.raises(ValueError) as exc:
        resolve_event_stream_bindings(
            _standard_readout_graph("count"), bindings=(binding,)
        )
    assert message in str(exc.value)


def test_event_stream_binding_rejects_non_spike_graph_input():
    graph = _standard_readout_graph("count")
    graph["inputs"][0]["signal_type"] = "continuous"
    binding = EventStreamBinding(
        "events",
        steps=torch.tensor([0]),
        batches=torch.tensor([0]),
        channels=torch.tensor([0]),
        steps_count=1,
        batch_size=1,
    )
    with pytest.raises(ValueError, match="requires signal_type spikes"):
        resolve_event_stream_bindings(graph, bindings=(binding,))


def test_event_stream_spikes_can_share_a_request_with_dense_valid_time_mask():
    graph = _standard_readout_graph("rate", mask=True)
    events = EventStreamBinding(
        "events",
        steps=torch.tensor([0, 0, 1]),
        batches=torch.tensor([0, 1, 0]),
        channels=torch.tensor([0, 1, 1]),
        steps_count=3,
        batch_size=2,
    )
    valid = DenseArrayBinding(
        "valid",
        torch.tensor([[True, True], [True, False], [False, False]]),
    )
    result = simulate(
        ExecutionSpec(
            kind="simulate",
            executor="graph",
            graph=graph,
            event_bindings=(events,),
            input_bindings=(valid,),
        )
    )
    torch.testing.assert_close(
        result.outputs["class_scores"],
        torch.tensor([[10.0, 10.0], [10.0, 10.0]]),
        rtol=0,
        atol=0,
    )
    protocol = result.metrics["execution_protocol"]
    assert protocol["representation"] == "mixed"
    assert protocol["binding_schema"] == "tools/snnsim.mixed-input-bindings/v1"
    assert protocol["masks"] == ["valid"]
    resolved = resolve_input_bindings(
        graph, dense_bindings=(valid,), event_bindings=(events,)
    )
    assert set(resolved.tensors) == {"events", "valid"}


def test_event_stream_file_loader_and_cli_emit_replayable_protocol(tmp_path):
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
    event_file = tmp_path / "events.npz"
    np.savez(
        event_file,
        steps=np.array([0, 1], dtype=np.int64),
        batches=np.array([0, 0], dtype=np.int64),
        channels=np.array([0, 1], dtype=np.int64),
        steps_count=np.array(2, dtype=np.int64),
        batch_size=np.array(1, dtype=np.int64),
    )
    bindings = load_event_stream_bindings(event_file, graph)
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
                "--event-file",
                str(event_file),
                "--input-dataset-id",
                "event-snapshot-v1",
                "--input-split",
                "test",
                "--no-input-shuffle",
                "--seed",
                "41",
                "--out-dir",
                str(out),
            ]
        )
        == 0
    )
    metrics = json.loads((out / "metrics.json").read_text())
    protocol = metrics["execution_protocol"]
    assert protocol["representation"] == "event_stream"
    assert protocol["inputs"][0]["event_count"] == 2
    assert protocol["dataset"]["identity"] == "event-snapshot-v1"
    assert protocol["dataset"]["split"] == "test"
    assert protocol["dataset"]["shuffle"] is False
    assert protocol["seeds"] == {"execution": 41}
    assert protocol["inputs"][0]["source"]["digest"].startswith("sha256:")
    with pytest.raises(SystemExit, match="--event-file requires --executor graph"):
        main(
            [
                "sim",
                "--event-file",
                str(event_file),
                "--out-dir",
                str(tmp_path / "legacy-out"),
            ]
        )


def test_named_multi_input_event_file_resolves_each_graph_port(tmp_path):
    graph = _coupled_graph(direction="uncoupled")
    event_file = tmp_path / "multi-events.npz"
    arrays = {
        "drive_a.steps": np.array([0, 1], dtype=np.int64),
        "drive_a.batches": np.array([0, 0], dtype=np.int64),
        "drive_a.channels": np.array([0, 2], dtype=np.int64),
        "drive_a.steps_count": np.array(2, dtype=np.int64),
        "drive_a.batch_size": np.array(1, dtype=np.int64),
        "drive_b.steps": np.array([], dtype=np.int64),
        "drive_b.batches": np.array([], dtype=np.int64),
        "drive_b.channels": np.array([], dtype=np.int64),
        "drive_b.steps_count": np.array(2, dtype=np.int64),
        "drive_b.batch_size": np.array(1, dtype=np.int64),
    }
    np.savez(event_file, **arrays)
    bindings = load_event_stream_bindings(event_file, graph)
    resolved = resolve_event_stream_bindings(graph, bindings=bindings)
    assert set(resolved.tensors) == {"drive_a", "drive_b"}
    assert resolved.tensors["drive_a"].shape == (2, 1, 3)
    assert resolved.tensors["drive_a"].sum() == 2
    assert resolved.tensors["drive_b"].shape == (2, 1, 2)
    assert resolved.tensors["drive_b"].sum() == 0


def test_fixed_rate_poisson_binding_has_exact_boundary_fixtures():
    graph = _standard_readout_graph("count")
    zero = resolve_poisson_input_bindings(
        graph,
        bindings=(PoissonInputBinding("events", 3, 2, (0.0,), 7),),
    )
    assert torch.count_nonzero(zero.tensors["events"]) == 0
    graph["timebase"]["dt"] = {"value": 1.0, "unit": "ms"}
    full = resolve_poisson_input_bindings(
        graph,
        bindings=(PoissonInputBinding("events", 3, 2, (1000.0,), 7),),
    )
    assert torch.all(full.tensors["events"] == 1)
    assert full.protocol["binding_schema"] == "tools/snnsim.poisson-input-binding/v1"
    assert full.protocol["inputs"][0]["selection"] == "constant"


def test_graph_inference_overrides_poisson_duration_and_rate():
    graph = _standard_readout_graph("count")
    result = simulate(
        ExecutionSpec(
            kind="simulate",
            executor="graph",
            graph=graph,
            poisson_bindings=(PoissonInputBinding("events", 2, 1, (0.0,), 7),),
            options={
                "inference_overrides": {
                    "duration_ms": 300.0,
                    "input_rate_hz": 10.0,
                }
            },
        )
    )
    protocol = result.metrics["execution_protocol"]
    assert protocol["timing"] == {
        "dt_ms": 100.0,
        "duration_ms": 300.0,
        "steps": 3,
    }
    assert protocol["inputs"][0]["rates_hz"] == [10.0]
    assert result.metrics["inference_overrides"] == {
        "schema": "tools/snnsim.inference-overrides/v1",
        "requested": {"duration_ms": 300.0, "input_rate_hz": 10.0},
        "resolved": {
            "duration_ms": 300.0,
            "timestep_ms": 100.0,
            "projection_scales": {},
            "input_rate_hz": 10.0,
        },
    }


def test_graph_inference_timestep_recompiles_and_preserves_duration(tmp_path):
    bundle = _direct_train_bundle()
    checkpoint = tmp_path / "checkpoint"
    train(
        ExecutionSpec(
            kind="train",
            executor="graph",
            graph=bundle.graph,
            training=bundle.training,
            inputs={"events": torch.ones(3, 1, 2)},
            targets={"label": torch.tensor([0])},
            seed=7,
            options={"save_final_checkpoint": checkpoint},
        )
    )
    result = simulate(
        ExecutionSpec(
            kind="simulate",
            executor="graph",
            graph=bundle.graph,
            checkpoint=checkpoint,
            poisson_bindings=(PoissonInputBinding("events", 3, 1, (0.0,), 13),),
            options={"inference_overrides": {"timestep_ms": 0.05}},
        )
    )
    assert bundle.graph["timebase"]["dt"] == {"value": 0.1, "unit": "ms"}
    assert result.metrics["execution_protocol"]["timing"] == {
        "dt_ms": 0.05,
        "steps": 6,
        "duration_ms": pytest.approx(0.3),
    }
    provenance = result.metrics["inference_overrides"]
    assert provenance["resolved"]["timestep_ms"] == 0.05
    assert provenance["resolved"]["duration_ms"] == pytest.approx(0.3)
    assert result.metrics["source_graph_digest"] == bundle.training["graph_digest"]
    assert (
        result.metrics["effective_graph_digest"]
        != result.metrics["source_graph_digest"]
    )


def test_graph_inference_timestep_rejects_non_resampleable_inputs():
    graph = _standard_readout_graph("count")
    with pytest.raises(ValueError, match="resampleable Poisson"):
        simulate(
            ExecutionSpec(
                kind="simulate",
                executor="graph",
                graph=graph,
                inputs={"events": torch.zeros(2, 1, 2)},
                options={"inference_overrides": {"timestep_ms": 50.0}},
            )
        )


def test_graph_inference_projection_scale_is_request_local():
    graph = _coupled_graph(direction="uncoupled")
    inputs = {
        "drive_a": torch.zeros(2, 1, 3),
        "drive_b": torch.zeros(2, 1, 2),
    }
    baseline = build(ExecutionSpec(kind="build", executor="graph", graph=graph, seed=5))
    projection = graph["projections"][0]
    parameter_id = projection["parameters"][0]
    result = simulate(
        ExecutionSpec(
            kind="simulate",
            executor="graph",
            graph=graph,
            inputs=inputs,
            seed=5,
            options={
                "inference_overrides": {"projection_scales": {projection["id"]: 0.25}}
            },
        )
    )
    torch.testing.assert_close(
        result.parameters[parameter_id], baseline.parameters[parameter_id] * 0.25
    )
    assert graph["parameters"][0]["initializer"] != {"kind": "constant", "value": 0.25}


def test_graph_inference_overrides_reject_ambiguous_or_unknown_requests():
    graph = _standard_readout_graph("count")
    with pytest.raises(ValueError, match="require Poisson"):
        simulate(
            ExecutionSpec(
                kind="simulate",
                executor="graph",
                graph=graph,
                inputs={"events": torch.zeros(2, 1, 2)},
                options={"inference_overrides": {"duration_ms": 100.0}},
            )
        )
    with pytest.raises(ValueError, match="unknown projections"):
        simulate(
            ExecutionSpec(
                kind="simulate",
                executor="graph",
                graph=graph,
                inputs={"events": torch.zeros(2, 1, 2)},
                options={
                    "inference_overrides": {"projection_scales": {"missing": 1.0}}
                },
            )
        )


def test_graph_inference_interventions_are_ordered_and_recorded():
    graph = _coupled_graph(direction="uncoupled")
    inputs = {
        "drive_a": torch.zeros(3, 1, 3),
        "drive_b": torch.zeros(3, 1, 2),
    }
    add = {
        "kind": "add_poisson_spikes",
        "population_id": "a_E",
        "rate_hz": 10000.0,
        "seed": 11,
    }
    drop = {
        "kind": "drop_spikes",
        "population_id": "a_E",
        "probability": 1.0,
        "seed": 12,
    }
    dropped = simulate(
        ExecutionSpec(
            kind="simulate",
            executor="graph",
            graph=graph,
            inputs=inputs,
            options={"inference_interventions": [add, drop]},
        )
    )
    added = simulate(
        ExecutionSpec(
            kind="simulate",
            executor="graph",
            graph=graph,
            inputs=inputs,
            options={"inference_interventions": [drop, add]},
        )
    )
    assert torch.count_nonzero(dropped.recordings["a_E.spikes"]) == 0
    assert torch.all(added.recordings["a_E.spikes"] == 1)
    provenance = added.metrics["inference_interventions"]
    assert provenance["schema"] == "tools/snnsim.inference-interventions/v1"
    assert provenance["requested"] == [drop, add]
    assert provenance["resolved"][1]["probability_per_step"] == 1.0


def test_graph_inference_intervention_stream_resumes_exactly():
    graph = _coupled_graph(direction="uncoupled")
    intervention = {
        "kind": "add_poisson_spikes",
        "population_id": "a_E",
        "rate_hz": 5000.0,
        "seed": 31,
    }
    full_model = GraphExecutor(plan_graph(graph), seed=4)
    full = full_model(
        {
            "drive_a": torch.zeros(4, 2, 3),
            "drive_b": torch.zeros(4, 2, 2),
        },
        interventions=(intervention,),
    )
    resumed_model = GraphExecutor(plan_graph(graph), seed=4)
    first = resumed_model(
        {
            "drive_a": torch.zeros(2, 2, 3),
            "drive_b": torch.zeros(2, 2, 2),
        },
        interventions=(intervention,),
    )
    second = resumed_model(
        {
            "drive_a": torch.zeros(2, 2, 3),
            "drive_b": torch.zeros(2, 2, 2),
        },
        runtime_state=first.runtime_state,
        interventions=(intervention,),
    )
    torch.testing.assert_close(
        full.recordings["a_E.spikes"],
        torch.cat((first.recordings["a_E.spikes"], second.recordings["a_E.spikes"])),
        rtol=0,
        atol=0,
    )


def test_graph_inference_interventions_reject_invalid_targets_and_values():
    graph = _coupled_graph(direction="uncoupled")
    inputs = {
        "drive_a": torch.zeros(1, 1, 3),
        "drive_b": torch.zeros(1, 1, 2),
    }
    with pytest.raises(ValueError, match="unknown population"):
        simulate(
            ExecutionSpec(
                kind="simulate",
                executor="graph",
                graph=graph,
                inputs=inputs,
                options={
                    "inference_interventions": [
                        {
                            "kind": "drop_spikes",
                            "population_id": "missing",
                            "probability": 0.5,
                        }
                    ]
                },
            )
        )
    with pytest.raises(ValueError, match="rate times dt"):
        simulate(
            ExecutionSpec(
                kind="simulate",
                executor="graph",
                graph=graph,
                inputs=inputs,
                options={
                    "inference_interventions": [
                        {
                            "kind": "add_poisson_spikes",
                            "population_id": "a_E",
                            "rate_hz": 10001.0,
                        }
                    ]
                },
            )
        )


def test_inference_artifact_manifest_authenticates_cache_identity(tmp_path):
    graph = _standard_readout_graph("count")
    result = simulate(
        ExecutionSpec(
            kind="simulate",
            executor="graph",
            graph=graph,
            inputs={"events": torch.ones(2, 1, 2)},
            seed=23,
        )
    )
    root = tmp_path / "inference"
    manifest = write_inference_artifacts(root, result, graph=graph, seed=23)
    assert manifest["schema"] == "tools/snnsim.inference-artifacts/v1"
    assert manifest["request_seed"] == 23
    assert {row["path"] for row in manifest["files"]} == {
        "recording.npz",
        "outputs.npz",
        "parameters.npz",
        "metrics.json",
    }
    outputs = next(row for row in manifest["files"] if row["path"] == "outputs.npz")
    assert outputs["arrays"] == [
        {"name": "class_scores", "shape": [1, 2], "dtype": "float32"}
    ]
    assert validate_inference_artifacts(root, graph=graph, seed=23) == manifest
    with pytest.raises(ValueError, match="request seed"):
        validate_inference_artifacts(root, seed=24)


def test_inference_artifact_validation_rejects_payload_corruption(tmp_path):
    graph = _standard_readout_graph("count")
    result = simulate(
        ExecutionSpec(
            kind="simulate",
            executor="graph",
            graph=graph,
            inputs={"events": torch.zeros(1, 1, 2)},
        )
    )
    root = tmp_path / "inference"
    write_inference_artifacts(root, result, graph=graph, seed=0)
    with (root / "outputs.npz").open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(ValueError, match="outputs.npz digest"):
        validate_inference_artifacts(root)


def test_derived_inference_products_use_named_public_tensors(tmp_path):
    bundle = ping_classifier()
    result = simulate(
        ExecutionSpec(
            kind="simulate",
            executor="graph",
            graph=bundle.graph,
            inputs={"image": torch.zeros(2, 2, 784)},
            seed=3,
        )
    )
    source = tmp_path / "source"
    source_manifest = write_inference_artifacts(
        source, result, graph=bundle.graph, seed=3
    )
    derived = tmp_path / "derived"
    summary = derive_inference_products(
        source,
        derived,
        logits_id="class_logits",
        labels=np.asarray([0, 1], dtype=np.int64),
        spike_recordings=("sensory_ping_E.spikes",),
    )
    assert summary["schema"] == "tools/snnsim.derived-inference/v1"
    assert summary["source_artifact_digest"] == source_manifest["artifact_digest"]
    assert summary["accuracy"] == 0.5
    rates = np.load(derived / "rates.npz", allow_pickle=False)
    rasters = np.load(derived / "rasters.npz", allow_pickle=False)
    try:
        assert rates["sensory_ping_E.spikes"].shape == (2, 256)
        assert rasters["sensory_ping_E.spikes.shape"].tolist() == [2, 2, 256]
        assert rasters["sensory_ping_E.spikes.steps"].dtype == np.int64
    finally:
        rates.close()
        rasters.close()
    validate_derived_inference_products(
        derived, source_artifact_digest=source_manifest["artifact_digest"]
    )


def test_derived_inference_products_fail_closed_on_names_and_corruption(tmp_path):
    bundle = ping_classifier()
    result = simulate(
        ExecutionSpec(
            kind="simulate",
            executor="graph",
            graph=bundle.graph,
            inputs={"image": torch.zeros(1, 1, 784)},
        )
    )
    source = tmp_path / "source"
    write_inference_artifacts(source, result, graph=bundle.graph, seed=0)
    with pytest.raises(ValueError, match="do not contain logits"):
        derive_inference_products(
            source,
            tmp_path / "missing",
            logits_id="missing",
            labels=np.asarray([0]),
        )
    derived = tmp_path / "derived"
    derive_inference_products(
        source,
        derived,
        logits_id="class_logits",
        labels=np.asarray([0]),
    )
    with (derived / "predictions.npy").open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(ValueError, match="predictions.npy digest"):
        validate_derived_inference_products(derived)


def test_categorical_poisson_samples_one_reproducible_rate_per_presentation():
    graph = _standard_readout_graph("count")
    binding = PoissonInputBinding("events", 4, 5, (0.0, 1.0, 5.0), 41, categorical=True)
    first = resolve_poisson_input_bindings(graph, bindings=(binding,))
    second = resolve_poisson_input_bindings(graph, bindings=(binding,))
    torch.testing.assert_close(first.tensors["events"], second.tensors["events"])
    row = first.protocol["inputs"][0]
    assert len(row["realized_rates_hz"]) == 5
    assert set(row["realized_rates_hz"]) <= {0.0, 1.0, 5.0}
    assert row["selection"] == "uniform_independent_per_presentation"


def test_dense_dataset_snapshot_rate_poisson_is_selected_and_seeded(tmp_path):
    graph = _standard_readout_graph("count")
    snapshot = tmp_path / "mnist.npz"
    np.savez(
        snapshot,
        features=np.asarray(
            [[0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]],
            dtype=np.float32,
        ),
        labels=np.asarray([0, 1, 0, 1], dtype=np.int64),
    )
    binding = DatasetSnapshotBinding(
        path=snapshot,
        input_id="events",
        target_id="label",
        dataset_id="mnist-fixture-sha256",
        split="train",
        encoder=DatasetEncoder(
            "rate_poisson", duration_ms=300.0, max_rate_hz=10.0, seed=17
        ),
        sample_cap=3,
        shuffle=True,
        order_seed=23,
    )
    first, targets = resolve_dataset_snapshot_binding(graph, binding)
    second, _ = resolve_dataset_snapshot_binding(graph, binding)
    torch.testing.assert_close(first.tensors["events"], second.tensors["events"])
    assert first.tensors["events"].shape == (3, 3, 2)
    assert targets[0].target_id == "label"
    protocol = first.protocol
    assert protocol["binding_schema"] == "tools/snnsim.dataset-snapshot-binding/v1"
    assert protocol["dataset"] == {
        "identity": "mnist-fixture-sha256",
        "split": "train",
        "sample_cap": 3,
        "batch_size": 3,
        "shuffle": True,
    }
    assert protocol["dataset_binding"]["selected_indices"] == [3, 0, 2]
    assert protocol["dataset_binding"]["source"]["digest"].startswith("sha256:")


def test_prebinned_dataset_snapshot_trains_with_bound_labels(tmp_path):
    bundle = _direct_train_bundle()
    snapshot = tmp_path / "prebinned.npz"
    features = np.zeros((3, 2, 2), dtype=np.uint8)
    features[:, 0, 0] = 1
    features[:, 1, 1] = 1
    np.savez(snapshot, features=features, labels=np.asarray([0, 1]))
    result = train(
        ExecutionSpec(
            kind="train",
            executor="graph",
            graph=bundle.graph,
            training=bundle.training,
            dataset_binding=DatasetSnapshotBinding(
                path=snapshot,
                input_id="events",
                target_id="label",
                dataset_id="prebinned-fixture",
                split="train",
                encoder=DatasetEncoder("prebinned_spikes"),
            ),
            seed=13,
        )
    )
    assert result.metrics["execution_protocol"]["representation"] == "dataset_snapshot"
    assert result.metrics["execution_protocol"]["targets"][0]["source"][
        "digest"
    ].startswith("sha256:")
    assert result.metrics["updates"][0]["loss"] == pytest.approx(0.6931471824645996)


def test_event_dataset_snapshot_bins_selected_samples_and_records_collisions(tmp_path):
    graph = _standard_readout_graph("count")
    snapshot = tmp_path / "shd.npz"
    np.savez(
        snapshot,
        labels=np.asarray([4, 7], dtype=np.int64),
        event_sample=np.asarray([0, 0, 0, 1], dtype=np.int64),
        event_time_ms=np.asarray([0.0, 99.9, 99.9, 200.0], dtype=np.float32),
        event_channel=np.asarray([0, 1, 1, 0], dtype=np.int64),
    )
    resolved, targets = resolve_dataset_snapshot_binding(
        graph,
        DatasetSnapshotBinding(
            path=snapshot,
            input_id="events",
            target_id="digit",
            dataset_id="shd-fixture",
            split="train",
            encoder=DatasetEncoder("event_bin", duration_ms=300.0),
        ),
    )
    assert resolved.tensors["events"].shape == (3, 2, 2)
    assert resolved.tensors["events"].sum() == 3
    encoder = resolved.protocol["dataset_binding"]["encoder"]
    assert encoder["retained_events"] == 4
    assert encoder["binary_collisions"] == 1
    assert targets[0].value.tolist() == [4, 7]


def test_dataset_snapshot_rejects_ambiguous_encoder_fields_and_bad_events(tmp_path):
    graph = _standard_readout_graph("count")
    dense = tmp_path / "dense.npz"
    np.savez(
        dense,
        features=np.zeros((1, 2), dtype=np.uint8),
        labels=np.asarray([0]),
    )
    with pytest.raises(ValueError, match="does not accept duration"):
        resolve_dataset_snapshot_binding(
            graph,
            DatasetSnapshotBinding(
                path=dense,
                input_id="events",
                dataset_id="dense",
                split="train",
                encoder=DatasetEncoder("prebinned_spikes", duration_ms=100.0),
            ),
        )
    events = tmp_path / "events.npz"
    np.savez(
        events,
        labels=np.asarray([0]),
        event_sample=np.asarray([0]),
        event_time_ms=np.asarray([300.0], dtype=np.float32),
        event_channel=np.asarray([0]),
    )
    with pytest.raises(ValueError, match="out of snapshot bounds"):
        resolve_dataset_snapshot_binding(
            graph,
            DatasetSnapshotBinding(
                path=events,
                input_id="events",
                dataset_id="events",
                split="train",
                encoder=DatasetEncoder("event_bin", duration_ms=300.0),
            ),
        )


def test_poisson_binding_rejects_invalid_rate_probability():
    graph = _standard_readout_graph("count")
    with pytest.raises(ValueError, match="rate times dt exceeds probability one"):
        resolve_poisson_input_bindings(
            graph,
            bindings=(PoissonInputBinding("events", 1, 1, (10001.0,), 0),),
        )


def test_graph_cli_generates_fixed_rate_poisson_protocol(tmp_path):
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
    out = tmp_path / "out"
    assert (
        main(
            [
                "sim",
                "--executor",
                "graph",
                "--bundle",
                str(bundle),
                "--poisson-protocol",
                "fixed-rate",
                "--input-rate",
                "0",
                "--n-batch",
                "1",
                "--t-ms",
                "200",
                "--seed",
                "13",
                "--out-dir",
                str(out),
            ]
        )
        == 0
    )
    protocol = json.loads((out / "metrics.json").read_text())["execution_protocol"]
    assert protocol["representation"] == "poisson"
    assert protocol["inputs"][0]["realized_rates_hz"] == [0.0]
    assert protocol["seeds"] == {"execution": 13, "poisson": {"events": 13}}


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


def test_disabled_projection_keeps_initialization_position_but_carries_no_drive():
    enabled_graph = _coupled_graph(direction="uncoupled")
    disabled_graph = copy.deepcopy(enabled_graph)
    disabled_graph["projections"][0]["enabled"] = False
    enabled = GraphExecutor(plan_graph(enabled_graph), seed=29)
    disabled = GraphExecutor(plan_graph(disabled_graph), seed=29)
    assert disabled.plan.projections[0].enabled is False
    for name, value in enabled.parameter_map().items():
        torch.testing.assert_close(
            value, disabled.parameter_map()[name], rtol=0, atol=0
        )
    result = disabled(
        {
            "drive_a": torch.ones(2, 1, 3),
            "drive_b": torch.zeros(2, 1, 2),
        }
    )
    projection_id = disabled.plan.projections[0].id
    assert torch.count_nonzero(result.recordings[f"{projection_id}.conductance"]) == 0


def test_explicit_initializers_constraints_units_and_realized_statistics():
    graph = _coupled_graph(direction="uncoupled")
    rows = {row["id"]: row for row in graph["parameters"]}
    first_id = graph["projections"][0]["parameters"][0]
    rows[first_id]["initializer"] = {
        "kind": "lower_clamped_normal",
        "mean": 0.5,
        "std": 0.1,
        "initial_zero_fraction": 0.5,
        "zeroing": "exact_k",
    }
    model = GraphExecutor(plan_graph(graph), seed=37)
    metadata = model.initialization_metadata[first_id]
    assert metadata["unit"] == "nS"
    assert metadata["constraint"] == {"kind": "non_negative"}
    assert metadata["scaling"] == "fan_in_normalized"
    assert metadata["statistics"]["count"] == 4
    assert metadata["statistics"]["zero_fraction"] == pytest.approx(0.5)
    assert (
        metadata
        == GraphExecutor(plan_graph(graph), seed=37).initialization_metadata[first_id]
    )


def test_signed_normal_and_uniform_initializers_have_distinct_semantics():
    graph = _coupled_graph(direction="uncoupled")
    rows = {row["id"]: row for row in graph["parameters"]}
    ids = [projection["parameters"][0] for projection in graph["projections"][:2]]
    rows[ids[0]]["initializer"] = {"kind": "signed_normal", "mean": -1.0, "std": 0.0}
    rows[ids[0]]["constraint"] = None
    rows[ids[1]]["initializer"] = {"kind": "uniform", "low": 2.0, "high": 2.0}
    model = GraphExecutor(plan_graph(graph), seed=1)
    assert torch.all(model.parameter_map()[ids[0]] < 0)
    assert torch.all(model.parameter_map()[ids[1]] > 0)


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


@pytest.mark.parametrize("selection", ["mixed", "empty"])
def test_recording_field_selection_preserves_outputs_and_branch_state(selection):
    graph = _coupled_graph()
    inputs = {"drive_a": torch.ones(12, 1, 3), "drive_b": torch.ones(12, 1, 2)}
    spec = dict(
        kind="simulate",
        executor="graph",
        graph=graph,
        inputs=inputs,
        device="cpu",
        seed=42,
    )
    full = simulate(ExecutionSpec(**spec))
    fields = (
        []
        if selection == "empty"
        else [
            graph["observables"][0]["id"],
            next(k for k in full.recordings if k.endswith(".conductance")),
            next(k for k in full.recordings if k.endswith(".voltage")),
        ]
    )
    selected = simulate(ExecutionSpec(**spec, recording_fields=fields))
    assert set(selected.recordings) == set(fields)
    for key in fields:
        assert torch.equal(selected.recordings[key], full.recordings[key])
    assert selected.outputs.keys() == full.outputs.keys()
    for key in full.outputs:
        assert torch.equal(selected.outputs[key], full.outputs[key])
    for expected, actual in zip(
        _state_tensors(full.runtime_state),
        _state_tensors(selected.runtime_state),
        strict=True,
    ):
        assert torch.equal(expected, actual)
    with pytest.raises(ValueError, match="unavailable recording fields"):
        simulate(ExecutionSpec(**spec, recording_fields=["missing"]))
