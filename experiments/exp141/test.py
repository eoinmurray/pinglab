"""Check native evaluation policies against exp141's real graph and bindings.

Small, disjoint digits subsets exercise native training and the pipeline without
retraining the published classifiers. Independent inference verifies selected
weights, partial-batch aggregation, sample replay and saved test provenance.
"""

import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from experiments.exp141 import dataset, recipe
from experiments.exp141.analyse import training_history
from experiments.exp141.compute import record_sample
from experiments.exp141.custom.network_artifacts import snapshot_parameters
from pingstore.library import load_json
from snnlab.sim import (
    CheckpointSelection,
    DatasetPartition,
    DatasetSnapshot,
    DatasetSnapshotBinding,
    DenseArrayBinding,
    Performance,
    SnapshotSpec,
    TrainingMetrics,
    ValidationSpec,
    resolve_dataset_snapshot_binding,
    simulate,
    train,
)
from snnlab.sim import TestSpec as HeldOutTestSpec


@pytest.fixture
def digits_case(tmp_path):
    cfg = recipe.configuration()
    cfg.update(
        epochs=2,
        batch_size=2,
        duration_ms=8.0,
        input_rate_hz=500.0,
        n_hidden=8,
        n_inhibitory=2,
        performance={**cfg["performance"], "compile_mode": None},
    )
    original = dataset.prepare(tmp_path / "original.npz", cfg)
    partitions = {}
    for split, count in (("train", 5), ("validation", 3), ("test", 7)):
        partition = original.partitions[split]
        partitions[split] = DatasetPartition(
            partition.features[:count],
            partition.labels[:count],
            partition.sample_ids[:count],
        )
    # Keep the diagnostic even when its original row falls outside the tiny subset.
    source = original.partitions[original.provenance["sample_partition"]]
    row = source.sample_ids.tolist().index(cfg["sample_index"])
    partitions["diagnostic"] = DatasetPartition(
        source.features[row : row + 1],
        source.labels[row : row + 1],
        source.sample_ids[row : row + 1],
    )
    snapshot = DatasetSnapshot.from_partitions(
        dataset_id=original.dataset_id,
        partitions=partitions,
        disjoint=original.disjoint,
        preprocessing=original.preprocessing,
        provenance={**original.provenance, "sample_partition": "diagnostic"},
    )
    path = snapshot.write(tmp_path / "digits.npz")
    return cfg, DatasetSnapshot.read(path)


def test_digits_partitions_are_disjoint_and_reproducible(tmp_path):
    cfg = recipe.configuration()
    first = dataset.prepare(tmp_path / "first.npz", cfg)
    second = dataset.prepare(tmp_path / "second.npz", cfg)
    assert first.counts == second.counts
    assert first.provenance == second.provenance
    assert [first.counts[split] for split in ("train", "validation", "test")] == [
        1437,
        180,
        180,
    ]
    partitions = [partition.sample_ids for partition in first.partitions.values()]
    np.testing.assert_array_equal(np.sort(np.concatenate(partitions)), np.arange(1797))
    for split, partition in first.partitions.items():
        repeated = second.partitions[split]
        for field in ("features", "labels", "sample_ids"):
            np.testing.assert_array_equal(
                getattr(partition, field), getattr(repeated, field)
            )
    # Calibrated before the archive migration: preserve sklearn's exact row order.
    expected_ids = {
        "train": "1eb5c4d41e55b63552bfb74eaffccb976b3b9e29902b06fe86d6273f62944d23",
        "validation": "9a6e545ab4f736b423e86e0e5cea9d234714d24fae6fb314947f73b42a098e84",
        "test": "82cd96421a4090c4d9924deaa33f62146e44cf9377df83fdf1626e141240eaf6",
    }
    for split, expected in expected_ids.items():
        ids = first.partitions[split].sample_ids.astype("<i8")
        assert hashlib.sha256(ids.tobytes()).hexdigest() == expected


@pytest.mark.parametrize("mode", ["coba", "ping"])
def test_native_dataset_preserves_encoded_training(digits_case, tmp_path, mode):
    """Archive bookkeeping must not change stochastic inputs or learned weights."""
    cfg, digits = digits_case
    partition = digits.partitions["train"]
    legacy_path = tmp_path / "legacy.npz"
    np.savez_compressed(
        legacy_path, features=partition.features, labels=partition.labels
    )
    native = dataset.binding(digits, "train", cfg)
    legacy = DatasetSnapshotBinding(
        path=legacy_path,
        input_id="drive",
        dataset_id="sklearn_digits",
        split="train",
        feature_key="features",
        label_key="labels",
        target_id="digit",
        encoder=native.encoder,
    )
    bundle = recipe.author_network(mode, cfg)
    results = []
    for binding in (legacy, native):
        result = train(
            bundle=bundle,
            input_bindings=(binding,),
            epochs=1,
            batch_size=cfg["batch_size"],
            seed=cfg["seed"],
            device="cpu",
            performance=Performance(**cfg["performance"]),
            shuffle=True,
            metrics=TrainingMetrics(validation="none", initial=(), test="none"),
            checkpoint_selection=CheckpointSelection.disabled(),
            diagnostics=False,
        )
        results.append(result)
    reference, migrated = results
    assert migrated.metrics["epochs"] == reference.metrics["epochs"]
    for name, value in reference.parameters.items():
        torch.testing.assert_close(migrated.parameters[name], value, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["coba", "ping"])
@pytest.mark.parametrize("compile_mode", [None, "default"])
def test_native_test_matches_independent_selected_replay(
    digits_case, tmp_path, mode, compile_mode
):
    cfg, digits = digits_case
    cfg = {**cfg, "performance": {**cfg["performance"], "compile_mode": compile_mode}}
    bundle = recipe.author_network(mode, cfg)
    # Fix the test spikes for an independent numerical check. Native test and
    # simulation phases intentionally use different stochastic encoding streams.
    encoded, _ = resolve_dataset_snapshot_binding(
        bundle.graph,
        dataset.binding(digits, "test", cfg),
        device="cpu",
        execution_seed=cfg["seed"],
    )
    bindings = (DenseArrayBinding("drive", encoded.tensors["drive"]),)
    labels = torch.from_numpy(digits.partitions["test"].labels.copy())
    epochs = []
    result = train(
        graph=bundle.graph,
        training=bundle.training,
        input_bindings=(dataset.binding(digits, "train", cfg),),
        validation=ValidationSpec(
            input_bindings=(dataset.binding(digits, "validation", cfg),),
            encoding_seeds=[cfg["seed"]],
        ),
        test=HeldOutTestSpec(
            input_bindings=bindings,
            targets={"digit": labels},
        ),
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        seed=cfg["seed"],
        device="cpu",
        performance=Performance(**cfg["performance"]),
        shuffle=True,
        metrics=TrainingMetrics(**cfg["training_metrics"]),
        checkpoint_selection=CheckpointSelection.from_dict(cfg["checkpoint_selection"]),
        save_final_checkpoint=tmp_path / "checkpoint",
        save_selected_checkpoint=tmp_path / "selected",
        diagnostics=False,
        on_epoch_end=lambda report: epochs.append(report.epoch),
        snapshot=SnapshotSpec(path=tmp_path / "train", stage="both", parameters=True),
    )
    assert result.metrics["performance"]["requested"] == cfg["performance"]
    assert epochs == [1, 2]
    history = result.metrics["epochs"]
    assert [row["epoch"] for row in history] == [0, 1, 2]
    assert all("validation_loss" in row for row in history)
    measured = result.metrics["test"]
    assert measured["examples"] == 7
    assert measured["checkpoint"] == "selected"
    assert measured["selection_record"]["values"] == [
        min(row["validation_loss"] for row in history[1:])
    ]
    # Independent replay confirms checkpoint choice and weighting of the odd batch.
    replay = simulate(
        graph=bundle.graph,
        checkpoint=tmp_path / "selected",
        input_bindings=bindings,
        batch_size=cfg["batch_size"],
        seed=cfg["seed"],
        device="cpu",
        # Compare compiled training's selected weights with an eager forward pass.
        performance=Performance(compile_mode=None),
        diagnostics=False,
    )
    scores = replay.outputs["class_scores"]
    assert measured["loss"] == pytest.approx(float(F.cross_entropy(scores, labels)))
    assert measured["accuracy"] == pytest.approx(
        float((scores.argmax(1) == labels).float().mean())
    )
    for name, parameter in result.parameters.items():
        torch.testing.assert_close(
            parameter, result.snapshots["end"].parameters[name], rtol=0, atol=0
        )


@pytest.mark.parametrize("mode", ["coba", "ping"])
def test_selected_test_and_sample_share_weights(digits_case, tmp_path, mode):
    cfg, digits = digits_case
    bundle = recipe.author_network(mode, cfg)
    result = train(
        graph=bundle.graph,
        training=bundle.training,
        input_bindings=(dataset.binding(digits, "train", cfg),),
        validation=ValidationSpec(
            input_bindings=(dataset.binding(digits, "validation", cfg),),
            encoding_seeds=[cfg["seed"]],
        ),
        test=HeldOutTestSpec(
            input_bindings=(dataset.binding(digits, "test", cfg),),
            encoding_seeds=[cfg["seed"]],
        ),
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        seed=cfg["seed"],
        device="cpu",
        performance=Performance(**cfg["performance"]),
        shuffle=True,
        metrics=TrainingMetrics(**cfg["training_metrics"]),
        checkpoint_selection=CheckpointSelection.from_dict(cfg["checkpoint_selection"]),
        save_final_checkpoint=tmp_path / "checkpoint",
        save_selected_checkpoint=tmp_path / "selected-checkpoint",
        diagnostics=False,
        snapshot=SnapshotSpec(path=tmp_path / "train", stage="both", parameters=True),
    )
    history = result.metrics["epochs"]
    assert [row["epoch"] for row in history] == [0, 1, 2]
    assert all("validation_loss" in row for row in history)
    measured = training_history(result.metrics, cfg, dict(test_samples=7)).test
    assert result.selected_checkpoint is not None
    assert measured["epoch"] == result.selected_checkpoint.data_state["epoch"]
    drive = dataset.sample_drive(digits, bundle, cfg)
    replay = record_sample(
        mode,
        bundle,
        drive,
        tmp_path / "selected-checkpoint",
        tmp_path,
        "cpu",
        cfg,
    )
    assert replay.metrics["performance"]["requested"] == cfg["performance"]
    for name, parameter in result.selected_checkpoint.parameters.items():
        # Sample inference must restore selected weights, even when
        # native training's end snapshot retains the final optimization state.
        with np.load(
            tmp_path / "best-validation" / "start" / "parameters.npz"
        ) as saved:
            snapshot = load_json(
                tmp_path / "best-validation" / "start" / "snapshot.json"
            )
            np.testing.assert_array_equal(
                saved[snapshot["parameters"][name]["key"]], parameter.numpy()
            )
    sample = replay.numpy(batch=0)
    np.testing.assert_array_equal(
        sample.outputs["class_scores"],
        sample.recorded_signals["readout.spikes"].sum(axis=0),
    )
    for field, value in (
        ("examples", 6),
        ("checkpoint", "final"),
        ("accuracy", float("nan")),
    ):
        corrupt = {**result.metrics, "test": {**result.metrics["test"], field: value}}
        with pytest.raises(ValueError):
            training_history(corrupt, cfg, dict(test_samples=7))


def test_native_test_dataset_is_only_resolved_at_completion(digits_case, tmp_path):
    cfg, digits = digits_case
    bundle = recipe.author_network("ping", cfg)
    test_binding = dataset.binding(digits, "test", cfg)
    # An interrupted run must not even try to open the test dataset.
    partial = train(
        graph=bundle.graph,
        training=bundle.training,
        input_bindings=(dataset.binding(digits, "train", cfg),),
        validation=ValidationSpec(
            input_bindings=(dataset.binding(digits, "validation", cfg),),
            encoding_seeds=[cfg["seed"]],
        ),
        test=HeldOutTestSpec(
            input_bindings=(replace(test_binding, path=tmp_path / "absent.npz"),),
        ),
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        seed=cfg["seed"],
        device="cpu",
        performance=Performance(**cfg["performance"]),
        shuffle=True,
        metrics=TrainingMetrics(**cfg["training_metrics"]),
        checkpoint_selection=CheckpointSelection.from_dict(cfg["checkpoint_selection"]),
        updates=1,
        save_final_checkpoint=tmp_path / "partial",
        diagnostics=False,
    )
    assert partial.metrics["test"] is None
    resumed = train(
        graph=bundle.graph,
        training=bundle.training,
        input_bindings=(dataset.binding(digits, "train", cfg),),
        validation=ValidationSpec(
            input_bindings=(dataset.binding(digits, "validation", cfg),),
            encoding_seeds=[cfg["seed"]],
        ),
        test=HeldOutTestSpec(
            input_bindings=(test_binding,),
            encoding_seeds=[cfg["seed"]],
        ),
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        seed=cfg["seed"],
        device="cpu",
        performance=Performance(**cfg["performance"]),
        shuffle=True,
        metrics=TrainingMetrics(**cfg["training_metrics"]),
        checkpoint_selection=CheckpointSelection.from_dict(cfg["checkpoint_selection"]),
        checkpoint=tmp_path / "partial",
        save_final_checkpoint=tmp_path / "checkpoint",
        save_selected_checkpoint=tmp_path / "selected-checkpoint",
        diagnostics=False,
    )
    measured = resumed.metrics["test"]
    assert measured["examples"] == 7
    assert measured["checkpoint"] == "selected"
    assert measured["encoding_draws"] == [cfg["seed"]]
    assert np.isfinite(measured["loss"])
    assert 0 <= measured["accuracy"] <= 1


def test_compute_analyse_present_preserve_selected_test(
    digits_case, tmp_path, monkeypatch
):
    from experiments.exp141 import analyse, compute, present
    from pingstore import core

    cfg, digits = digits_case
    repo = tmp_path / "repo"
    report = repo / "experiments" / "exp141" / "report" / "report.mdx"
    report.parent.mkdir(parents=True)
    report.write_text("---\nid: exp141\ncollection: datasets\n---\n")
    # Source-control capture is separate from the native artifacts and real
    # Pingstore lifecycle exercised here; no test run enters the working store.
    monkeypatch.setattr(core, "capture_code", lambda *_: {"git_commit": "test"})
    for module in (compute, analyse, present):
        monkeypatch.setattr(module, "REPO", repo)
    monkeypatch.setattr(recipe, "configuration", lambda: cfg)
    monkeypatch.setattr(analyse, "configuration", lambda: cfg)
    monkeypatch.setattr(present, "configuration", lambda: cfg)
    monkeypatch.setenv("SNNLAB_DEVICE", "cpu")

    def prepare_fixture(path, configuration):
        # Use the same real digits fixture as the independent checkpoint checks.
        Path(path).write_bytes(digits.path.read_bytes())
        return DatasetSnapshot.read(path)

    monkeypatch.setattr(dataset, "prepare", prepare_fixture)
    trained = []
    native_train = compute.train

    def retain_training_result(**kwargs):
        result = native_train(**kwargs)
        trained.append(result)
        return result

    monkeypatch.setattr(compute, "train", retain_training_result)
    computed = compute.compute()
    computation = core.source_run(repo / ".pingstore", computed, stage="compute")
    assert not (computation.directory / ".scratch").exists()
    assert {path.name for path in computation.export.iterdir()} == {
        "results.json",
        "recording.npz",
        *(
            name
            for mode in cfg["models"]
            for name in (
                f"{mode}.bundle",
                f"{mode}--circuit.svg",
                f"{mode}--final--end",
                f"{mode}--best-validation--start",
            )
        ),
    }
    from snnlab import lang

    for mode, result in zip(cfg["models"], trained, strict=True):
        restored = computation.restore_bundle(
            f"{mode}.bundle", tmp_path / f"{mode}.bundle"
        )
        assert (
            lang.load_bundle(restored).graph == recipe.author_network(mode, cfg).graph
        )
        for weights, stage, checkpoint in (
            ("final", "end", result.training_checkpoint),
            ("best-validation", "start", result.selected_checkpoint),
        ):
            saved = snapshot_parameters(computation, mode, weights, stage)
            assert set(saved) == set(checkpoint.parameters)
            for name, parameter in checkpoint.parameters.items():
                np.testing.assert_array_equal(saved[name], parameter.numpy())
    analysed = analyse.analyse(computed)
    presented = present.present(analysed)
    source = core.source_run(repo / ".pingstore", presented, stage="present")
    numbers = load_json(source.file("numbers.json"))
    assert numbers["configuration"] == cfg
    assert "final_training" not in numbers
    assert numbers["sample"]["checkpoint"] == "validation_selected"
    for mode in cfg["models"]:
        measured = numbers["test"][mode]
        assert measured["examples"] == 7
        assert measured["checkpoint"] == "selected"
        assert numbers["sample"]["modes"][mode]["epoch"] == measured["epoch"]
        plot = source.file(numbers["figures"][mode]).read_text()
        assert "Held-out test (selected)" in plot
        assert "Training (selected checkpoint)" not in plot
        assert source.file(numbers["raster_figures"][mode]).is_file()
