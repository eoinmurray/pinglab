"""External backends keep their numerics and produce explicit local evidence."""

import numpy as np
import pytest
import torch
from experiments.helpers.external_artifacts import external_execution
from pingstore import stages
from pingstore.contracts import load_json
from pingstore.library_artifacts import validate_native_artifacts


@pytest.fixture
def local_repo(tmp_path, monkeypatch):
    monkeypatch.setattr(
        stages,
        "memberships",
        lambda _: {"exp080": "demo", "exp081": "demo", "exp125": "datasets"},
    )
    monkeypatch.setattr(
        stages, "_capture_code", lambda *args: {"git_commit": "fixture"}
    )
    torch.set_num_threads(1)
    return tmp_path


def test_local_snapshot_preserves_external_parameters_and_definition(local_repo):
    parameters = {"weight": np.ones((2, 3))}
    with stages.stage_run(local_repo, "exp080", "compute") as run:
        with external_execution(
            run.export / "local",
            backend="custom",
            configuration={"dt_ms": 0.1},
            parameters=lambda: parameters,
        ):
            parameters["weight"] *= 2
    source = stages.source_run(local_repo / ".pingstore", run.run_id)
    for stage, expected in (("start", 1), ("end", 2)):
        metadata = load_json(source.file("local", stage, "snapshot.json"))
        assert metadata["backend"] == "custom"
        assert metadata["configuration"] == {"dt_ms": 0.1}
        with np.load(source.file("local", stage, "parameters.npz")) as arrays:
            assert (arrays["weight"] == expected).all()
    validate_native_artifacts(
        source.export, source.record["scientific_artifacts"]["native_paths"]
    )


def test_nonspiking_membrane_compute_retains_local_snapshots(local_repo, monkeypatch):
    from experiments.exp081 import compute, recipe

    monkeypatch.setattr(compute, "REPO", local_repo)
    monkeypatch.setenv("EXP081_DEVICE", "cpu")
    cfg = recipe.configuration(smoke=True)
    cfg.update(
        presentation_ms=1,
        input_rate_grid_hz=[0, 5],
        moment_draws=2,
        distribution_draws=2,
    )
    monkeypatch.setattr(recipe, "configuration", lambda **kwargs: cfg)
    identity = compute.compute()
    source = stages.source_run(local_repo / ".pingstore", identity)
    assert len(list(source.export.rglob("*snapshot.json"))) == 4
    assert all(
        load_json(p)["backend"] == "local nonspiking membrane"
        for p in source.export.rglob("*snapshot.json")
    )


def test_snntorch_compute_keeps_backend_and_snapshots(local_repo, monkeypatch):
    from experiments.exp125 import analyse, compute

    monkeypatch.setattr(compute, "REPO", local_repo)
    monkeypatch.setattr(analyse, "REPO", local_repo)
    monkeypatch.setattr(
        compute, "CONFIG", dict(compute.CONFIG, epochs=1, steps=2, hidden=4)
    )
    identity = compute.compute(1)
    source = stages.source_run(local_repo / ".pingstore", identity)
    assert len(list(source.export.rglob("*snapshot.json"))) == 4
    assert all(
        load_json(p)["backend"] == "snntorch.Leaky"
        for p in source.export.rglob("*snapshot.json")
    )
    assert (source.export / "selected_weights.npz").is_file()
    assert not (source.export / "initial_weights.npz").exists()
    assert not (source.export / "final_weights.npz").exists()
    measured = analyse.analyse(identity)
    assert stages.source_run(local_repo / ".pingstore", measured).file("results.json").is_file()


def test_nonspiking_decoder_training_saves_actual_torch_weights(tmp_path, monkeypatch):
    from experiments.exp080 import compute, recipe

    monkeypatch.setenv("EXP080_DEVICE", "cpu")
    monkeypatch.setattr(compute, "N_TIMESTEPS", 4)
    monkeypatch.setattr(compute, "HIDDEN_UNITS", 4)
    cfg = recipe.configuration(smoke=True)
    cfg.update(epochs=1, train_count=2, validation_count=2)
    images = np.full((4, 28, 28), 64, dtype=np.uint8)
    compute.train_seed(
        images, np.array([0, 1, 0, 1]), 42, tmp_path, cfg, state_cache={}
    )
    for stage in ("start", "end"):
        directory = tmp_path / "seed-42--training" / stage
        assert load_json(directory / "snapshot.json")["backend"].startswith("torch MLP")
        with np.load(directory / "parameters.npz") as arrays:
            assert arrays["0.weight"].shape == (4, 784)
