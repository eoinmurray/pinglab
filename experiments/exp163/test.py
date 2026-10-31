"""Spatial symmetry, normalization, measurement, native architecture and independent imports."""

import ast
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from experiments.exp163 import recipe
from experiments.exp163.analyse import measure, spatial_measure
from pingstore.library import source_run
from snnlab.meanfield import Trajectory
from snnlab.sim import NetworkSnapshot


@pytest.fixture
def reference():
    return source_run(
        Path(__file__).resolve().parents[2] / ".pingstore",
        "exp162-r001-compute",
        experiment="exp162",
        stage="compute",
    )


def test_selected_defaults_preserve_original_success_threshold(reference):
    cfg = recipe.configuration(
        reference_configuration=reference.record["execution"]["configuration"]
    )
    assert (
        cfg["coupling_us"],
        cfg["cue_hz"],
        cfg["background_hz"],
        cfg["spatial"]["contrast"],
    ) == (0.6, 8.0, 22.0, 0.4)
    assert cfg["spatial"]["min_receiver_contrast"] == 0.15
    assert cfg["phase"]["min_plv"] == 0.95
    for value in [0, -1, float("nan"), float("inf")]:
        with pytest.raises(ValueError):
            recipe.configuration(
                reference_configuration=reference.record["execution"]["configuration"],
                coupling_us=value,
            )


def test_spatial_matrices_preserve_equal_sensory_totals_and_cross_incoming_sums(
    reference,
):
    cfg = recipe.configuration(
        reference_configuration=reference.record["execution"]["configuration"]
    )
    original = NetworkSnapshot.read(
        reference.file("execution/start/snapshot.json")
    ).load_parameters()
    actual = recipe.spatial_weights(
        original, cfg, reference.record["execution"]["configuration"]
    )
    for c in "AB":
        np.testing.assert_allclose(
            actual["sensory_" + c + ".weight"].numpy().sum(),
            original["sensory_" + c + ".weight"].numpy().sum(),
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            actual[c + "_E_to_C_E.weight"].numpy().sum(axis=0, dtype=np.float64),
            0.6,
            rtol=1e-6,
        )
    np.testing.assert_allclose(
        actual["sensory_A.weight"].numpy(),
        actual["sensory_B.weight"].numpy()[:, ::-1],
        rtol=1e-6,
    )
    assert np.argmax(actual["A_E_to_C_E.weight"].numpy()[:, 40]) == 40
    np.testing.assert_array_equal(
        actual["C_I_to_E.weight"], original["C_I_to_E.weight"]
    )


def test_centre_swap_mirrors_weights_without_changing_any_input_spikes(reference):
    a = recipe.configuration(
        reference_configuration=reference.record["execution"]["configuration"]
    )
    b = recipe.configuration(
        reference_configuration=reference.record["execution"]["configuration"],
        swap=True,
    )
    original = NetworkSnapshot.read(
        reference.file("execution/start/snapshot.json")
    ).load_parameters()
    x = recipe.spatial_weights(
        original, a, reference.record["execution"]["configuration"]
    )
    y = recipe.spatial_weights(
        original, b, reference.record["execution"]["configuration"]
    )
    for c in "AB":
        np.testing.assert_allclose(
            x["sensory_" + c + ".weight"].numpy()[:, ::-1],
            y["sensory_" + c + ".weight"].numpy(),
            rtol=1e-6,
        )
    for cfg in [a, b]:
        cfg["duration_ms"] = 20
    dx = recipe.make_inputs(a)
    dy = recipe.make_inputs(b)
    for key in dx:
        np.testing.assert_array_equal(dx[key], dy[key])


def test_spatial_metric_accepts_crossover_and_rejects_fixed_side(reference):
    cfg = recipe.configuration(
        reference_configuration=reference.record["execution"]["configuration"]
    )
    cfg["windows_ms"] = [[10.0, 20.0], [30.0, 40.0]]
    t = np.arange(0, 50, 0.1)
    x = recipe.coordinates(cfg)
    masks = {
        c: abs(x - center) <= cfg["spatial"]["roi_half_width"]
        for c, center in cfg["spatial"]["centers"].items()
    }
    for switch, expected in [(True, True), (False, False)]:
        spikes = {
            c + "_E": np.zeros((len(t), 1, len(x)), dtype=np.uint8) for c in "ABC"
        }
        epochs = []
        for i, (start, end) in enumerate(cfg["windows_ms"]):
            indices = np.flatnonzero((t >= start) & (t <= end))
            winner = "AB"[i]
            for c in "AB":
                for offset in range(3):
                    spikes[c + "_E"][indices[offset], 0, masks[c]] = 1
            favored = winner if switch else "A"
            for c in "AB":
                for offset in range(3 if c == favored else 2):
                    spikes["C_E"][indices[offset], 0, masks[c]] = 1
            epochs.append(dict(window_ms=[start, end], cued=winner))
        metrics = dict(epochs=epochs, switched=True)
        spatial_measure(spikes, t, cfg, metrics)
        assert metrics["prototype_passed"] is expected
        assert abs(metrics["epochs"][0]["spatial"]["C_contrast"] - 0.2) < 1e-10


def test_native_architecture_keeps_all_pathways_and_cue_switch(reference):
    cfg = recipe.configuration(
        reference_configuration=reference.record["execution"]["configuration"]
    )
    bundle, _ = recipe.author_network(cfg)
    graph = bundle.graph
    paths = {p["id"]: p for p in graph["projections"]}
    assert (
        len(graph["populations"]) == 6
        and len(graph["inputs"]) == 5
        and len(paths) == 21
    )
    for epoch in ("before", "after"):
        assert sum(paths["cue_" + c]["modulation"][epoch] for c in "AB") == 1
    for c in "AB":
        assert not any(
            p["source"].startswith("C_") and p["target"].startswith(c + "_")
            for p in paths.values()
        )


def test_phase_metric_accepts_switch_and_rejects_old_winner(reference):
    cfg = recipe.configuration(
        reference_configuration=reference.record["execution"]["configuration"]
    )
    t = np.arange(0, 2400, 0.1)
    switch = cfg["switch_ms"]
    pa = np.where(
        t < switch, 40 * t / 1000, 40 * switch / 1000 + 35 * (t - switch) / 1000
    )
    pb = np.where(
        t < switch, 35 * t / 1000, 35 * switch / 1000 + 40 * (t - switch) / 1000
    )
    for pc, passed in [(40 * t / 1000, True), (pa, False)]:
        state = {
            c + "_E.rate": 20 + 10 * np.cos(2 * np.pi * p)
            for c, p in [("A", pa), ("B", pb), ("C", pc)]
        }
        metrics, _ = measure(Trajectory(t, state), cfg)
        assert metrics["switched"] is passed


def test_imports_are_standalone_and_side_effect_free(tmp_path):
    root = Path(__file__).resolve().parents[2]
    for path in Path(__file__).parent.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            modules = (
                [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else []
            )
            for module in modules:
                if module.startswith("experiments.exp"):
                    assert module.split(".")[1] == "exp163"
        if path.name != "test.py":
            assert ".scratch/" not in path.read_text()
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from experiments.exp163 import recipe,compute,analyse,present",
        ],
        cwd=tmp_path,
        env=dict(
            os.environ, PYTHONPATH=str(root) + os.pathsep + str(root / "libraries")
        ),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert not list(tmp_path.iterdir())
