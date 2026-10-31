"""One-way causality, saved physics, phase discrimination and standalone imports."""

import ast
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from experiments.exp159 import analyse, compute, recipe
from snnlab.sim import Performance, build


@pytest.fixture
def native():
    cfg = recipe.configuration()
    bundle = recipe.author_network(cfg)
    initialized = build(
        graph=bundle.graph,
        seed=85,
        device="cpu",
        performance=Performance(compile_mode=None),
    )
    return bundle, recipe.configuration(
        incoming_weights_us=compute.mean_incoming_weights(
            bundle, initialized.parameters
        )
    )


def test_configuration_is_fresh_and_has_no_grid():
    first = recipe.configuration()
    second = recipe.configuration()
    first["network"]["populations"]["E"]["size"] = 1
    assert second["network"]["populations"]["E"]["size"] == 80
    assert "sweep" not in second
    assert second["network"]["input_rates_hz"] == dict(drive_A=26, drive_B=24)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(coupling_us=0),
        dict(coupling_us=0.13),
        dict(difference_hz=50),
        dict(difference_hz=float("nan")),
    ],
)
def test_invalid_overrides_fail_before_deriving_settings(kwargs):
    with pytest.raises(ValueError):
        recipe.configuration(**kwargs)


def test_native_graph_contains_no_feedback(native):
    bundle, cfg = native
    projections = bundle.graph["projections"]
    assert len(projections) == 12
    assert not any(
        p["source"].startswith("PING_B") and p["target"].startswith("PING_A")
        for p in projections
    )
    model = recipe.build_model(bundle, cfg, coupling_us=0.06)
    assert len(model.states) == 16
    for control in model.controls.values():
        assert control(499.9) == 0 and control(500) == 1


def test_short_native_pair_preserves_sender_and_changes_only_receiver(native):
    bundle, cfg = native
    cfg["network"]["onset_ms"] = 1.0
    cfg["duration_ms"] = 20
    bundle = recipe.author_network(cfg)
    solutions = [
        compute.integrate(recipe.build_model(bundle, cfg, coupling_us=k), cfg)
        for k in (0, 0.06)
    ]
    compute.verify_pair(*solutions, onset_ms=1)
    assert (
        np.max(
            np.abs(
                solutions[0].state["PING_B_E.rate"]
                - solutions[1].state["PING_B_E.rate"]
            )
        )
        > 1e-3
    )


def test_saved_coupling_tampering_is_rejected(native, tmp_path):
    bundle, cfg = native
    cfg["duration_ms"] = 3
    model = recipe.build_model(bundle, cfg, coupling_us=0)
    compute.integrate(model, cfg).write(tmp_path / "control-trajectory.json")
    model.write(tmp_path / "control-model.json")
    source = SimpleNamespace(file=lambda name: tmp_path / name)
    analyse.read_trajectory(source, "control", cfg)
    cfg["incoming_weights_us"]["PING_A_E_to_I"] *= 2
    with pytest.raises(ValueError, match="coupling"):
        analyse.read_trajectory(source, "control", cfg)


def test_synthetic_phase_metrics_distinguish_drift_from_locking():
    from snnlab.meanfield import Trajectory

    cfg = recipe.configuration()
    t = np.arange(0, 1500.1, 0.1)
    for frequency, locked in [(50, True), (51, False)]:
        state = {
            "PING_A_E.rate": 20 + 10 * np.cos(2 * np.pi * 50 * t / 1000),
            "PING_B_E.rate": 20 + 10 * np.cos(2 * np.pi * frequency * t / 1000 + 0.3),
            "PING_B_I_to_E.conductance": 0.1
            + 0.03 * np.cos(2 * np.pi * frequency * t / 1000),
        }
        metrics, _ = analyse.measure(Trajectory(t, state), cfg)
        assert metrics["locked"] is locked


def test_imports_are_side_effect_free_and_have_no_foreign_experiment_dependencies(
    tmp_path,
):
    root = Path(__file__).resolve().parents[2]
    for path in Path(__file__).parent.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            elif isinstance(node, ast.Import):
                modules = [n.name for n in node.names]
            else:
                continue
            for module in modules:
                if module.startswith("experiments.exp"):
                    assert module.split(".")[1] == "exp159"
        if path.name != "test.py":
            assert ".scratch/" not in path.read_text()
    env = dict(os.environ, PYTHONPATH=str(root) + os.pathsep + str(root / "libraries"))
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from experiments.exp159 import recipe,compute,analyse,present",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert not list(tmp_path.iterdir())
