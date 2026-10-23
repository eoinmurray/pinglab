"""Check the drive schedule, state continuation and sparse-event diagnostics.

Tests use small active circuits and synthetic rhythms. They verify causal state
preservation and the specific weight intervention without a full-size run.
"""

import ast
from pathlib import Path

import numpy as np
import pytest
import torch
from experiments.exp148 import dataset, recipe
from experiments.exp148.custom.execution import execute, executor
from experiments.exp148.custom.metrics import measure, validate_events


def test_schedule_and_configuration():
    cfg = recipe.configuration()
    assert (cfg["n_e"], cfg["n_i"], cfg["n_external"]) == (4000, 1000, 4000)
    assert recipe.windows(cfg) == (
        ("baseline", 50, 550),
        ("ramp", 550, 1050),
        ("hold", 1050, 1550),
    )
    np.testing.assert_equal(
        recipe.drive_gain(np.array([0, 550, 800, 1050, 1550]), cfg), [1, 1, 0.5, 0, 0]
    )
    bounds = recipe.execution_bounds(cfg)
    assert bounds[0] == 0 and bounds[-1] == 31000
    assert np.all(np.diff(bounds) > 0)
    ramp = np.array(bounds)[(np.array(bounds) >= 11000) & (np.array(bounds) <= 21000)]
    np.testing.assert_equal(np.diff(ramp), 20)
    for value in (-1, 1.1, float("nan")):
        with pytest.raises(ValueError):
            recipe.configuration(final_i_gain=value)
    with pytest.raises(ValueError):
        recipe.configuration(ramp_ms=0.5)
    assert recipe.configuration()["control"] is not cfg["control"]


def test_shared_input_and_physical_weights():
    cfg = recipe.configuration(
        n_e=8, n_i=4, n_external=8, baseline_ms=1, ramp_ms=1, hold_ms=1
    )
    graph = recipe.author_network(cfg).graph
    projections = {p["id"]: p for p in graph["projections"]}
    assert len(projections) == 6
    assert projections["X_to_E"]["source"] == projections["X_to_I"]["source"]
    assert cfg["jump_means_us"]["X_to_I"] == pytest.approx(0.00108)
    assert cfg["jump_means_us"]["I_to_E"] == pytest.approx(0.008)
    values = dataset.encode_drive(cfg)
    rng = np.random.default_rng(cfg["seed"] + cfg["external_seed_offset"])
    expected = (
        rng.random(values.shape, dtype=np.float32)
        < cfg["external_rate_hz"] * cfg["dt_ms"] / 1000
    )
    np.testing.assert_equal(values.numpy(), expected)


def test_native_continuation_and_constant_control():
    torch.set_num_threads(1)
    cfg = recipe.configuration(
        n_e=8, n_i=4, n_external=8, baseline_ms=1, ramp_ms=2, hold_ms=1, final_i_gain=1
    )
    bundle = recipe.author_network(cfg)
    external = dataset.encode_drive(cfg)
    external[:100] = 1
    model = executor(bundle, cfg)
    with torch.no_grad():
        full = model({"X": external}, diagnostics=False)
        arrays, audit, split = execute(bundle, external, cfg)
    assert audit["unchanged_other_weights"] and audit["final_gain_verified"]
    for p in ("E", "I"):
        times, cells = np.nonzero(full.outputs[p + "_spikes"][:, 0].numpy())
        np.testing.assert_equal(arrays[p + "_time_ms"], times * cfg["dt_ms"])
        np.testing.assert_equal(arrays[p + "_cell"], cells)
    for p in full.runtime_state.voltages:
        torch.testing.assert_close(
            split.runtime_state.voltages[p],
            full.runtime_state.voltages[p],
            rtol=0,
            atol=0,
        )
    cfg["control"]["final_gain"] = 0.25
    _, audit, _ = execute(bundle, external, cfg)
    assert (
        audit["final_weight_sha256"]["X_to_I.weight"]
        != audit["initial_weight_sha256"]["X_to_I.weight"]
    )


def test_cycle_measurement_and_bad_events():
    cfg = recipe.configuration()
    times = np.arange(1060, 1550, 25, dtype=float)
    events = {
        p + "_" + q: arr.copy()
        for p in ("E", "I")
        for q, arr in (("time_ms", times), ("cell", np.zeros(len(times), dtype=int)))
    }
    result = measure(events, cfg)
    assert result["hold_cycles"]["E"]["frequency_hz"] == pytest.approx(40)
    assert result["windows"][0]["E_spikes"] == 0
    events["E_cell"][0] = cfg["n_e"]
    with pytest.raises(ValueError, match="outside its population"):
        validate_events(events, cfg)


def test_no_cross_experiment_or_scratch_dependency():
    root = Path(__file__).parent
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            assert not any(
                name.startswith("experiments.exp")
                and not name.startswith("experiments.exp148")
                for name in names
            )
        if path.name != "test.py":
            assert ".scratch" not in path.read_text()
