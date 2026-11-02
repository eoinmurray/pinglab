"""Non-simulation validation of the migration and preserved evidence contract."""

import ast
from copy import deepcopy
from pathlib import Path

import pytest
from experiments.exp167 import recipe
from experiments.exp167.analyse import validate_metrics


def test_protocol_matches_retained_scratch_overrides():
    cfg = recipe.configuration()
    assert cfg["measurement_ms"] == [50.0, 1050.0]
    assert cfg["duration_ms"] == 1050.0
    assert cfg["seed"] == 42
    assert (cfg["n_e"], cfg["n_i"], cfg["n_external"]) == (4000, 1000, 4000)
    ai = recipe.condition_configuration(cfg, "AI")
    ping = recipe.condition_configuration(cfg, "PING")
    assert ai["control"]["final_gain"] == 1.0
    assert ping["control"]["final_gain"] == 0.0
    assert ping["control"]["ramp_start_ms"] == 50.0
    assert ping["control"]["ramp_end_ms"] == 50.05
    ping["control"]["final_gain"] = 1.0
    assert ping == ai
    ai["measurement_ms"][0] = 0
    assert cfg["measurement_ms"][0] == 50.0
    with pytest.raises(ValueError):
        recipe.condition_configuration(cfg, "unknown")
    with pytest.raises(ValueError):
        recipe.configuration(seed=-1)


def test_missing_cv_requires_no_qualifying_cells():
    fields = [
        "rate_hz",
        "median_cv",
        "cv_cells",
        "peak_hz",
        "peak_contrast",
        "max_bin_fraction",
        "rhythmicity",
    ]
    values = {
        label: {
            "condition": label,
            "mean_correlation": 0.0,
            "correlation_sd": 0.0,
            **{f"{pop}_{field}": 1.0 for pop in ("E", "I") for field in fields},
        }
        for label in ("AI", "PING")
    }
    valid = deepcopy(values)
    valid["AI"].update(E_median_cv=None, E_cv_cells=0)
    assert validate_metrics(valid) is valid
    for edits in ({"E_median_cv": None}, {"E_rate_hz": float("nan")}, {"extra": 3}):
        broken = deepcopy(values)
        broken["AI"].update(edits)
        with pytest.raises(ValueError):
            validate_metrics(broken)


def test_stages_do_not_import_scratch_or_other_experiments():
    for path in Path(__file__).parent.glob("*.py"):
        if path.name == "test.py":
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("experiments.exp148")
                assert "scratch" not in (node.module or "")
        # Importing stage modules must never dispatch simulation or create runs.
        for node in tree.body:
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                assert ast.unparse(node.value.func) not in {
                    "simulate",
                    "compute",
                    "analyse",
                    "present",
                    "stage_run",
                }
