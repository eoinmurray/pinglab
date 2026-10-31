"""Non-simulation checks of source isolation, observation windows and reconstruction."""

import ast
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import lfilter
from experiments.exp164 import recipe
from experiments.exp164.analyse import reconstruct

REPO = Path(__file__).resolve().parents[2]


def reference_settings():
    return dict(
        switch_ms=1200.0, encoding="rates", network={"populations": {"E": {"size": 64}}}
    )


def test_fresh_configuration_and_strict_pre_switch_windows():
    source = reference_settings()
    a = recipe.configuration(source_configuration=source)
    b = recipe.configuration(source_configuration=source)
    a["window_ms"][0] = 0
    assert b["window_ms"] == [650.0, 1100.0]
    for overrides in [
        dict(end_ms=1200.0),
        dict(end_ms=float("nan")),
        dict(window_ms=(1000.0, 900.0)),
        dict(closeup_ms=(500.0, 1000.0)),
    ]:
        with pytest.raises(ValueError):
            recipe.configuration(source_configuration=source, **overrides)
    source["encoding"] = "gains"
    with pytest.raises(ValueError):
        recipe.configuration(source_configuration=source)


def test_reconstruction_preserves_causal_delay_and_full_spike_kick():
    spikes = np.zeros((12, 2))
    spikes[2, 0] = 1
    weights = np.array([[2.0, 4.0], [6.0, 8.0]])
    actual = reconstruct(spikes, weights, decay=0.5, delay_steps=3)
    expected = np.zeros(12)
    expected[5:] = 3 * 0.5 ** np.arange(7)
    np.testing.assert_array_equal(actual, expected)


def test_reconstruction_commutes_with_recipient_averaging():
    rng = np.random.default_rng(12)
    spikes = (rng.random((80, 3)) < 0.2).astype(float)
    weights = rng.random((3, 5))
    delayed = np.pad(spikes @ weights, ((4, 0), (0, 0)))[:80]
    expected = lfilter([1.0], [1.0, -0.8], delayed, axis=0).mean(1)
    np.testing.assert_allclose(
        reconstruct(spikes, weights, decay=0.8, delay_steps=4),
        expected,
        rtol=1e-14,
        atol=1e-14,
    )


def test_no_foreign_experiment_or_scratch_imports():
    for path in Path(__file__).parent.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            modules = (
                [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else []
            )
            for module in modules:
                if module.startswith("experiments.exp"):
                    assert module.split(".")[1] == "exp164"
        if path.name != "test.py":
            assert ".scratch/" not in path.read_text()


def test_imports_have_no_execution_effects(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from experiments.exp164 import recipe,compute,analyse,present",
        ],
        cwd=tmp_path,
        env=dict(
            os.environ, PYTHONPATH=str(REPO) + os.pathsep + str(REPO / "libraries")
        ),
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert not list(tmp_path.iterdir())
