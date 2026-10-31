"""Spiking competition settings, cue topology, phase discrimination and imports."""

import ast
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from experiments.exp162 import recipe
from experiments.exp162.analyse import measure
from snnlab.meanfield import Trajectory


@pytest.fixture
def reference_cfg():
    n = dict(
        populations=dict(
            E=dict(size=8, capacitance_nf=1.0, leak_us=0.05, refractory_ms=3.0),
            I=dict(size=2, capacitance_nf=0.5, leak_us=0.1, refractory_ms=1.5),
        ),
        local={
            k: dict(tau_ms=tau)
            for k, tau in [
                ("E_to_E", 2.0),
                ("E_to_I", 1.0),
                ("I_to_E", 9.0),
                ("I_to_I", 6.0),
            ]
        },
        dt_ms=0.1,
        input_channels=32,
        resting_mv=-65.0,
        threshold_mv=-50.0,
        reset_mv=-65.0,
        sensory=dict(gain=9.2, tau_ms=2.0, delay_ms=0.1),
        cross=dict(tau_ms=0.5, delay_ms=2.0),
    )
    weights = {
        "PING_B_" + k: v
        for k, v in [
            ("input_to_E", 0.2),
            ("E_to_E", 0.395),
            ("E_to_I", 0.75),
            ("I_to_E", 1.5),
            ("I_to_I", 0.76),
        ]
    }
    return dict(network=n, incoming_weights_us=weights, coupling_us=0.04)


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_numeric_overrides_fail(reference_cfg, value):
    for key in ("coupling_us", "cue_hz", "background_hz"):
        with pytest.raises(ValueError):
            recipe.configuration(reference_configuration=reference_cfg, **{key: value})


def test_configuration_is_fresh_and_defaults_match_selected(reference_cfg):
    cfg = recipe.configuration(reference_configuration=reference_cfg)
    assert cfg["coupling_us"] == 0.12
    cfg["network"]["populations"]["E"]["size"] = 4
    cfg["incoming_weights_us"]["PING_B_E_to_E"] = 100
    assert reference_cfg["network"]["populations"]["E"]["size"] == 8
    assert reference_cfg["incoming_weights_us"]["PING_B_E_to_E"] == 0.395


def test_native_architecture_retains_symmetry_and_applies_cross_scale_once(
    reference_cfg,
):
    cfg = recipe.configuration(reference_configuration=reference_cfg)
    bundle, weights = recipe.author_network(cfg)
    graph = bundle.graph
    paths = {p["id"]: p for p in graph["projections"]}
    assert (
        len(graph["populations"]) == 6
        and len(graph["inputs"]) == 5
        and len(paths) == 21
    )
    for c in "AB":
        for p in "EI":
            key = c + "_E_to_C_" + p
            assert weights[key] == 0.04
            assert weights[key] * recipe.scales(cfg)[key] == 0.12
        assert not any(
            p["source"].startswith("C_") and p["target"].startswith(c + "_")
            for p in paths.values()
        )
    assert not any(
        p["source"].startswith("A_")
        and p["target"].startswith("B_")
        or p["source"].startswith("B_")
        and p["target"].startswith("A_")
        for p in paths.values()
    )
    for epoch in ("before", "after"):
        assert sum(paths["cue_" + c]["modulation"][epoch] for c in "AB") == 1


def test_coupling_changes_preserve_complete_input_trains(reference_cfg):
    a = recipe.configuration(reference_configuration=reference_cfg, coupling_us=0.04)
    b = recipe.configuration(reference_configuration=reference_cfg, coupling_us=0.12)
    for cfg in [a, b]:
        cfg["duration_ms"] = 20
    x = recipe.make_inputs(a)
    y = recipe.make_inputs(b)
    assert set(x) == {"X_A", "X_B", "X_C", "U"}
    for key in x:
        np.testing.assert_array_equal(x[key], y[key])
        assert x[key].dtype == np.uint8


def test_phase_metric_accepts_switch_and_rejects_persistent_old_winner(reference_cfg):
    cfg = recipe.configuration(reference_configuration=reference_cfg)
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
    metrics, _ = measure(
        Trajectory(t, {c + "_E.rate": np.zeros(t.shape) for c in "ABC"}), cfg
    )
    assert not metrics["switched"]


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
                    assert module.split(".")[1] == "exp162"
        if path.name != "test.py":
            assert ".scratch/" not in path.read_text()
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from experiments.exp162 import recipe,compute,analyse,present",
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
