"""Check the migration's scientific contracts without executing the experiment.

These tests compile authored graphs, inspect committed settings and validate
synthetic spike arrays. They do not load MNIST, build runtime models, simulate,
backpropagate or create Pingstore runs.

An isolated source-only checkout proves that entry points import and expose
help without the original scratch directory. Mocked native execution checks
typed input bindings and start snapshots; native recording round trips use
synthetic arrays rather than simulations.
"""

import shutil
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from experiments.exp144 import analyse, compute, dataset, recipe
from experiments.exp144.custom.initializers import diagonal_background
from pingstore.layout import canonical_export_unit, normalize_export_layout
from snnlab import lang
from snnlab.sim import NPZRecordingSink, RecordingBlock, RecordingData

REPO = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("protocol", ["comparison", "screen", "long"])
def test_protocol_graphs_preserve_architecture_without_training(protocol):
    cfg = recipe.configuration(protocol)
    for case in cfg["cases"]:
        bundle = recipe.author_network(cfg, case)
        graph = bundle.graph
        assert graph["refractory_timing"] == "legacy_coba"
        assert {p["id"]: p["size"] for p in graph["populations"]} == {
            "E": 1024,
            "I": 256,
            "readout": 10,
        }
        parameters = {p["id"]: p for p in graph["parameters"]}
        # Authoring uses target × source; the runtime exposes source × target.
        assert parameters["input_to_E.weight"]["shape"] == [1024, 784]
        assert (
            parameters["input_to_E.weight"]["initializer"]["initial_zero_fraction"]
            == 0.95
        )
        for name in ("E_to_I.weight", "I_to_E.weight"):
            assert parameters[name]["initializer"][
                "initial_zero_fraction"
            ] == pytest.approx(1 - case["recurrence_density"])
        assert {p["id"] for p in graph["projections"]} == {
            "input_to_E",
            "E_to_I",
            "I_to_E",
            "background_to_I",
            "readout_projection",
        }
        background = parameters["background_to_I.weight"]
        assert background["initializer"] == {
            "kind": "custom_initializer",
            "definition": "exp144.diagonal_background/v1",
            "config": {"jump_us": cfg["i_background_jump_us"]},
        }
        assert background["initialization_scaling"] == "direct"
        assert bundle.training is None
        assert (
            not {
                "diagnostic",
                "learning_rate",
                "weight_decay",
                "surrogate_slope",
                "gradient_clip",
            }
            & cfg.keys()
        )


def test_protocol_defaults_and_fresh_configuration():
    short = recipe.configuration()
    assert (short["duration_ms"], short["burn_in_ms"], short["seeds"]) == (
        500.0,
        0.0,
        [42],
    )
    assert [
        (case["i_rate_hz"], case["recurrence_density"]) for case in short["cases"]
    ] == [(0, 1), (5, 0.1), (10, 0.1)]
    long = recipe.configuration("long")
    assert (long["duration_ms"], long["burn_in_ms"], long["seeds"]) == (
        11000.0,
        1000.0,
        [42, 43, 44],
    )
    screen = recipe.configuration("screen")
    assert len(screen["cases"]) == 20
    assert all(case["recurrence_density"] == 1 for case in screen["cases"])
    short["analysis"]["correlation_bins_ms"].append(999)
    short["cases"][0]["input_parent_us"] = 99
    assert recipe.configuration()["cases"][0]["input_parent_us"] == 3
    assert recipe.configuration()["analysis"]["correlation_bins_ms"] == [5, 10, 50]
    with pytest.raises(ValueError, match="unknown protocol"):
        recipe.configuration("latest")
    with pytest.raises(ValueError, match="source configuration"):
        recipe.validate_configuration(short)


def test_reject_legacy_source():
    with pytest.raises(ValueError):
        recipe.validate_configuration({"schema": "scratch.exp143.mnist-ai/v1"})


def test_background_initializer_is_diagonal_and_rng_neutral():
    before = torch.random.get_rng_state()
    weights = diagonal_background(
        (4, 4), {"jump_us": 0.15}, device="cpu", dtype=torch.float32
    )
    torch.testing.assert_close(weights, torch.eye(4) * 0.15, rtol=0, atol=0)
    assert torch.equal(before, torch.random.get_rng_state())
    with pytest.raises(ValueError, match="equal source"):
        diagonal_background(
            (2, 3), {"jump_us": 0.15}, device="cpu", dtype=torch.float32
        )


def test_bundle_round_trip_keeps_authored_background(tmp_path):
    cfg = recipe.configuration()
    bundle = recipe.author_network(cfg, cfg["cases"][1])
    bundle.write(tmp_path / "network.bundle", visualise=False)
    restored = lang.load_bundle(tmp_path / "network.bundle")
    assert restored.graph == bundle.graph
    assert restored.training is None
    # Loading in a fresh process must restore the extension by its saved import
    # reference, without an earlier call to author_network() in that process.
    script = """
import sys
sys.path.insert(0, sys.argv[1])
from snnlab import lang
bundle = lang.load_bundle(sys.argv[2])
assert bundle.training is None
assert any(row['initializer'].get('definition') == 'exp144.diagonal_background/v1'
           for row in bundle.graph['parameters'])
"""
    loaded = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            script,
            str(REPO),
            str(tmp_path / "network.bundle"),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert loaded.returncode == 0, loaded.stderr


def test_native_background_encoding_preserves_shared_uniforms():
    cfg = recipe.configuration()
    cfg.update(duration_ms=100.0, n_input=3, n_i=2)
    image, backgrounds = dataset.encode_drive(torch.ones(1, 3), cfg, 42)
    assert image.shape == (1000, 1, 3)
    uniform = torch.rand(
        (1000, 1, 2),
        generator=torch.Generator().manual_seed(42 + cfg["background_seed_offset"]),
    )
    for rate, actual in backgrounds.items():
        torch.testing.assert_close(
            actual, (uniform < rate * cfg["dt_ms"] / 1000).float()
        )
    assert torch.all(backgrounds[5.0] <= backgrounds[10.0])


def test_compute_uses_one_native_simulation_per_case(tmp_path, monkeypatch):
    cfg = recipe.configuration()
    run = SimpleNamespace(
        export=tmp_path / "export", record={"execution": {}}, run_id="mock-run"
    )
    run.export.mkdir()

    @contextmanager
    def staged(*args, **kwargs):
        assert kwargs["configuration"] == cfg
        yield run

    monkeypatch.setattr(compute, "stage_run", staged)
    monkeypatch.setattr(
        compute.dataset,
        "load_sample",
        lambda *args: (torch.zeros(1, 784), {"partition": "train", "sample_index": 0}),
    )
    image = torch.zeros(5, 1, 784)
    backgrounds = {rate: torch.zeros(5, 1, 256) for rate in (0.0, 5.0, 10.0)}
    monkeypatch.setattr(
        compute.dataset, "encode_drive", lambda *args: (image, backgrounds)
    )
    # Diagrams are not needed to check dispatch or evidence contracts.
    monkeypatch.setattr(lang.Bundle, "visualise", lambda *args, **kwargs: None)
    calls = []

    def simulated(**kwargs):
        case = cfg["cases"][len(calls)]
        calls.append(kwargs)
        assert kwargs["seed"] == 42
        assert kwargs["input_bindings"][0].input_id == "image"
        assert kwargs["input_bindings"][0].value is image
        assert kwargs["input_bindings"][1].value is backgrounds[case["i_rate_hz"]]
        assert kwargs["snapshot"].stage == "start" and kwargs["snapshot"].parameters
        assert (
            kwargs["snapshot"].path
            == run.export / case["id"] / "seed-42/simulation/start"
        )
        assert kwargs["recording"].sink_id == "snnlab.npz-recording-blocks/v1"
        assert kwargs["recording"].block_steps == cfg["recording_block_steps"]
        assert not kwargs["retain_runtime_state"]
        weights = {
            "input_to_E.weight": torch.zeros(784, 1024),
            "background_to_I.weight": torch.eye(256) * 0.15,
        }
        for name, shape in (
            ("E_to_I.weight", (1024, 256)),
            ("I_to_E.weight", (256, 1024)),
        ):
            weights[name] = torch.zeros(shape)
            weights[name].view(-1)[
                : round(weights[name].numel() * case["recurrence_density"])
            ] = 1
        return SimpleNamespace(
            snapshots={"start": SimpleNamespace(parameters=weights)},
            write_metrics=lambda path: path.write_text(
                '{"schema": "snnlab.metrics/v1"}'
            ),
        )

    monkeypatch.setattr(compute, "simulate", simulated)
    assert (
        compute.compute(SimpleNamespace(protocol="comparison", run_id=None))
        == "mock-run"
    )
    assert len(calls) == len(cfg["cases"])
    assert all(
        value == ["start"]
        for value in run.record["scientific_artifacts"]["snapshot_stages"].values()
    )


def test_native_recording_survives_pingstore_layout(tmp_path):
    export = tmp_path / "export"
    sink = NPZRecordingSink(export / "case/seed-42/recording")
    spikes = torch.tensor([[[0.0, 1.0]], [[1.0, 0.0]], [[0.0, 0.0]], [[1.0, 1.0]]])
    for start in (0, 2):
        sink(
            RecordingBlock(
                "E.spikes",
                "dense",
                spikes[start : start + 2],
                start,
                start + 2,
                0.1,
                cells=(0, 1),
                unit="spike",
                batch_size=1,
                sample_ids=(0,),
            )
        )
    record = {}
    normalize_export_layout(tmp_path, record)
    restored = RecordingData.read(
        canonical_export_unit(export, "case", "seed-42", "recording")
    ).select_batch(0)
    np.testing.assert_array_equal(restored.signals["E.spikes"], spikes[:, 0].numpy())
    np.testing.assert_allclose(restored.info["E.spikes"].time_ms, [0, 0.1, 0.2, 0.3])


def test_spike_shape_and_binary_validation():
    cfg = recipe.configuration()
    spikes = np.zeros((5000, 1024), dtype=bool)
    analyse.validate_spikes(spikes, cfg, "E")
    with pytest.raises(ValueError, match="spike recording"):
        analyse.validate_spikes(spikes[:-1], cfg, "E")
    invalid = spikes.astype(float)
    invalid[0, 0] = np.nan
    with pytest.raises(ValueError, match="spike recording"):
        analyse.validate_spikes(invalid, cfg, "E")


def test_source_only_entrypoints_without_scratch(tmp_path):
    shutil.copytree(
        Path(__file__).parent,
        tmp_path / "experiments/exp144",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    shutil.copytree(
        REPO / "tools/pingstore",
        tmp_path / "tools/pingstore",
        ignore=shutil.ignore_patterns("__pycache__", "tests"),
    )
    script = """
import importlib, sys
from contextlib import contextmanager
from pathlib import Path
root = Path.cwd()
sys.path[:0] = [str(root), str(root / "tools")]
import snnlab.sim as sim
import pingstore.stages as stages
import torchvision.datasets as datasets
def forbidden(*args, **kwargs):
    raise AssertionError("import or help attempted scientific work")
for name in ("build", "simulate", "train"):
    setattr(sim, name, forbidden)
stages.stage_run = stages.source_run = forbidden
datasets.MNIST = forbidden
for name in ("recipe", "dataset", "compute", "analyse", "present"):
    module = importlib.import_module("experiments.exp144." + name)
    if hasattr(module, "main"):
        sys.argv = [name, "--help"]
        try:
            module.main()
        except SystemExit as error:
            assert error.code == 0
        else:
            raise AssertionError("help did not exit")
assert not (root / ".scratch").exists()
assert not (root / ".pingstore").exists()
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
