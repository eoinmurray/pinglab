"""Focused checks; no simulation or upstream stage execution."""

import ast
import json
from pathlib import Path
import numpy as np
import pytest
from experiments.exp168 import recipe
from experiments.exp168.analyse import raster_from_events


def test_fixed_zero_constraints_and_exact_endpoint_differences():
    ai, ping = (recipe.configuration(label) for label in recipe.CONDITIONS)
    assert ai["input_i_gain"] == 0.3 and ping["input_i_gain"] == 0
    assert (ai["external_rate_hz"], ping["external_rate_hz"]) == (1.25, 2.5)
    assert (
        ai["input_connection_probability"],
        ping["input_connection_probability"],
    ) == (0.005, 0.05)
    assert {k for k in ai if ai[k] != ping[k]} == {
        "condition",
        "external_rate_hz",
        "input_connection_probability",
        "input_i_gain",
    }
    for cfg in (ai, ping):
        assert (
            cfg["initializer_parent_means_us"]["E_to_E"]
            == cfg["initializer_parent_means_us"]["I_to_I"]
            == 0
        )
        assert cfg["measurement_ms"] == [50.0, 2050.0]
    ai["measurement_ms"][0] = 0
    assert recipe.configuration("AI")["measurement_ms"][0] == 50
    for kwargs in (
        {"condition": "unknown"},
        {"condition": "AI", "seed": -1},
        {"condition": "AI", "device": "bad"},
    ):
        with pytest.raises(ValueError):
            recipe.configuration(**kwargs)


def test_native_authoring_preserves_absence_and_constant_gain(tmp_path):
    # Native compilation only; no simulation.
    destination = tmp_path / "network.bundle"
    recipe.author_network(recipe.configuration("PING")).write(
        destination, visualise=False
    )
    graph = json.loads((destination / "graph.json").read_text())
    projections = {p["id"]: p for p in graph["projections"]}
    assert set(projections) == {"E_to_I", "I_to_E", "X_to_E", "X_to_I"}
    assert (
        projections["X_to_I"]["modulation"]["before"]
        == projections["X_to_I"]["modulation"]["after"]
        == 0
    )
    assert {p["id"]: p["size"] for p in graph["populations"]} == {"E": 4000, "I": 1000}


def test_event_quantization_and_duplicate_rejection():
    cfg = {"measurement_ms": [50.0, 51.0], "dt_ms": 0.05, "n_e": 3}
    matrix, mask = raster_from_events(
        np.array([49.95, 50.0, 50.95, 51.0]), np.array([0, 1, 2, 0]), cfg, "E"
    )
    assert matrix.sum() == 2 and matrix[0, 1] and matrix[19, 2]
    assert mask.tolist() == [False, True, True, False]
    with pytest.raises(ValueError):
        raster_from_events(np.array([50.0, 50.0]), np.array([1, 1]), cfg, "E")
    with pytest.raises(ValueError):
        raster_from_events(np.array([50.0]), np.array([3]), cfg, "E")


def test_no_other_experiment_or_external_scratch_imports():
    for path in Path(__file__).parent.glob("*.py"):
        if path.name == "test.py":
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                assert not module.startswith("experiments.exp") or module.startswith(
                    "experiments.exp168"
                )
                assert "scratch" not in module
        assert ".scratch/" not in path.read_text()
