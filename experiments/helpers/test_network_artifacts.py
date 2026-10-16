"""Native execution evidence survives Pingstore completion and downstream use."""

import json

import numpy as np
import pytest
import torch
from experiments.exp124.analyse import parameters as training_parameters
from experiments.exp127 import recipe
from experiments.helpers.network_artifacts import (
    model_execution,
    snapshot_parameters,
    snapshot_request,
)
from pingstore import stages
from pingstore.contracts import PingstoreError, load_json
from pingstore.layout import normalize_export_layout
from pingstore.library_artifacts import (
    native_artifact_files,
    validate_native_artifacts,
)
from snnlab import lang
from snnlab.sim import DenseArrayBinding, ExecutionSpec, build, infer, simulate, train


@pytest.fixture
def stage_repo(tmp_path, monkeypatch):
    monkeypatch.setattr(stages, "memberships", lambda repo: {"exp127": "demo"})
    monkeypatch.setattr(
        stages, "_capture_code", lambda *args: {"git_commit": "fixture"}
    )
    monkeypatch.setitem(recipe.CONFIG, "duration_ms", 4)
    torch.set_num_threads(1)
    return tmp_path


@pytest.mark.parametrize("kind", ["simulate", "infer", "train"])
def test_native_requests_retain_authenticated_evidence(stage_repo, kind):
    bundle = recipe.network()
    spec = ExecutionSpec(
        kind=kind,
        graph=bundle.graph,
        training=bundle.training,
        device="cpu",
        seed=127,
        input_bindings=(DenseArrayBinding("inputs", torch.ones(4, 2, 3)),),
        targets={"class": torch.tensor([0, 1])} if kind == "train" else {},
        epochs=1 if kind == "train" else 0,
    )
    with stages.stage_run(stage_repo, "exp127", "compute") as run:
        result = {"simulate": simulate, "infer": infer, "train": train}[kind](
            snapshot_request(spec, bundle, run.export / kind)
        )
    source = stages.source_run(stage_repo / ".pingstore", run.run_id)
    for stage in ("start", "end"):
        saved = snapshot_parameters(source, kind, stage)
        for name, value in saved.items():
            np.testing.assert_array_equal(
                value, result.snapshots[stage].parameters[name].numpy()
            )
    restored = source.restore_bundle(
        f"{kind}/network.bundle", stage_repo / f"restored-{kind}"
    )
    assert lang.load_bundle(restored).graph == bundle.graph
    assert all(
        len(p.relative_to(source.export).parts) <= 2 for p in source.export.rglob("*")
    )


def test_custom_loop_snapshots_capture_bound_and_final_weights(stage_repo):
    bundle = recipe.network()
    model = build(ExecutionSpec(kind="build", graph=bundle.graph, device="cpu")).model
    with torch.no_grad():
        next(iter(model.parameter_map().values())).fill_(2)
    with stages.stage_run(stage_repo, "exp127", "compute") as run:
        with model_execution(model, bundle, run.export / "coba/train", kind="train"):
            with torch.no_grad():
                next(iter(model.parameter_map().values())).fill_(3)
    source = stages.source_run(stage_repo / ".pingstore", run.run_id)
    assert all(
        (v == 2).all() for v in training_parameters(source, "coba", "start").values()
    )
    assert all(
        (v == 3).all() for v in training_parameters(source, "coba", "end").values()
    )


def test_completion_rejects_missing_end_snapshot(stage_repo):
    bundle = recipe.network()
    model = build(ExecutionSpec(kind="build", graph=bundle.graph, device="cpu")).model
    with pytest.raises(PingstoreError, match="missing start/end"):
        with stages.stage_run(stage_repo, "exp127", "compute") as run:
            with model_execution(model, bundle, run.export / "sim"):
                pass
            (run.export / "sim/end/snapshot.json").unlink()
    assert not (stage_repo / ".pingstore/runs" / run.run_id).exists()


def test_bundle_assets_are_byte_preserved_and_restored(stage_repo):
    net = lang.Network("asset_example")
    pop = net.population("cells", size=1, neuron=lang.CUBA_LIF())
    net.output("spikes", pop.spikes)
    net.asset("configuration", media_type="application/json")
    asset = stage_repo / "source.json"
    asset.write_text(
        json.dumps({"path": "unit/start/parameters.npz"}, separators=(",", ":")) + "\n"
    )
    bundle = lang.compile(net, assets={"configuration": asset})
    with stages.stage_run(stage_repo, "exp127", "compute") as run:
        bundle.write(run.export / "unit/network.bundle")
    source = stages.source_run(stage_repo / ".pingstore", run.run_id)
    restored = source.restore_bundle("unit/network.bundle", stage_repo / "restored")
    loaded = lang.load_bundle(restored)
    assert (
        restored / loaded.manifest["assets"][0]["path"]
    ).read_bytes() == asset.read_bytes()


def test_normalization_of_copied_evidence_is_idempotent(stage_repo):
    bundle = recipe.network()
    with stages.stage_run(stage_repo, "exp127", "compute") as run:
        simulate(
            snapshot_request(
                ExecutionSpec(
                    kind="simulate",
                    graph=bundle.graph,
                    device="cpu",
                    input_bindings=(DenseArrayBinding("inputs", torch.ones(4, 1, 3)),),
                ),
                bundle,
                run.export / "simulation",
            )
        )
    record = load_json(run.directory / "run.json")
    before = {
        p.relative_to(run.export): p.read_bytes()
        for p in run.export.rglob("*")
        if p.is_file()
    }
    normalize_export_layout(run.directory, record)
    after = {
        p.relative_to(run.export): p.read_bytes()
        for p in run.export.rglob("*")
        if p.is_file()
    }
    assert before == after
    validate_native_artifacts(
        run.export, record["scientific_artifacts"]["native_paths"]
    )


def test_parameter_tampering_is_detected(stage_repo):
    bundle = recipe.network()
    model = build(ExecutionSpec(kind="build", graph=bundle.graph, device="cpu")).model
    with stages.stage_run(stage_repo, "exp127", "compute") as run:
        with model_execution(model, bundle, run.export / "sim"):
            pass
        (run.export / "sim/start/parameters.npz").write_bytes(b"tampered")
        files = native_artifact_files(run.export)
        with pytest.raises(PingstoreError, match="checksum mismatch"):
            validate_native_artifacts(run.export, {p: p for p in files})
        # Do not complete the deliberately corrupted fixture.
        (run.export / "sim/start/parameters.npz").unlink()
        import shutil

        shutil.rmtree(run.export / "sim")


def test_legacy_shd_parameters_remain_readable(tmp_path):
    directory = tmp_path / "export/coba"
    directory.mkdir(parents=True)
    np.savez_compressed(directory / "initial_weights.npz", weight=np.ones((2, 2)))
    source = stages.SourceRun(tmp_path, {})
    np.testing.assert_array_equal(
        training_parameters(source, "coba", "start")["weight"], np.ones((2, 2))
    )


def test_shd_training_and_sample_retain_native_weights_without_recovery_state(
    stage_repo, monkeypatch
):
    from experiments.exp124 import compute as shd
    from experiments.exp124 import recipe as shd_recipe

    monkeypatch.setattr(stages, "memberships", lambda repo: {"exp124": "datasets"})
    cfg = shd_recipe.configuration(epochs=1, smoke=True)
    cfg.update(duration_ms=8, train_examples=20, validation_examples=20)
    with stages.stage_run(stage_repo, "exp124", "compute", configuration=cfg) as run:
        dataset = run.scratch / "dataset.npz"
        np.savez_compressed(
            dataset,
            event_time_ms=np.zeros(40),
            event_channel=np.zeros(40, dtype=np.int64),
            event_offsets=np.arange(41),
            train_labels=np.arange(20),
            validation_labels=np.arange(20),
        )
        for condition in shd_recipe.MODELS:
            bundle, checkpoint = shd.train_condition(
                condition, run, cfg, dataset, "cpu"
            )
            arrays = shd.record_sample(
                condition, bundle, checkpoint, torch.ones(8, 1, 700), run, cfg
            )
            assert np.isfinite(arrays[f"{condition}_scores"]).all()
            for population in shd_recipe.POPULATIONS[condition]:
                assert arrays[f"{condition}_{population}"].shape[0] == 8
    source = stages.source_run(stage_repo / ".pingstore", run.run_id)
    assert not (source.directory / ".scratch").exists()
    for condition in shd_recipe.MODELS:
        final = training_parameters(source, condition, "end")
        before_inference = snapshot_parameters(
            source, f"{condition}--inference", "start"
        )
        after_inference = snapshot_parameters(source, f"{condition}--inference", "end")
        assert final.keys() == before_inference.keys() == after_inference.keys()
        for name, value in final.items():
            np.testing.assert_array_equal(value, before_inference[name])
            np.testing.assert_array_equal(value, after_inference[name])
        assert not (source.unit(condition) / "final_weights.npz").exists()
        assert not (source.unit(condition) / "tensors.npz").exists()
        restored = source.restore_bundle(
            f"{condition}/train/network.bundle", stage_repo / f"restored-{condition}"
        )
        assert lang.load_bundle(restored).graph["name"] == f"exp124_{condition}"
