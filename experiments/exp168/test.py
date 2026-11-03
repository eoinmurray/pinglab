"""Focused checks; no simulation or upstream stage execution."""

import json
from pathlib import Path

import pytest
from experiments.exp168 import recipe


def test_configuration_contract():
    on_ai = recipe.configuration("AI", recurrence="on")
    on_ping = recipe.configuration("PING", recurrence="on")
    assert (on_ai["dt_ms"], on_ai["duration_ms"]) == (0.05, 2050.0)
    assert {k for k in on_ai if on_ai[k] != on_ping[k]} == {
        "condition",
        "external_rate_hz",
        "input_connection_probability",
        "input_i_gain",
    }
    assert on_ai["initializer_parent_means_us"]["E_to_E"] > 0
    assert on_ai["initializer_parent_means_us"]["I_to_I"] > 0
    for bad in (
        {"condition": "gamma"},
        {"condition": "AI", "recurrence": "masked"},
        {"condition": "AI", "device": "bad"},
        {"condition": "AI", "seed": -1},
    ):
        with pytest.raises(ValueError):
            recipe.configuration(**bad)


def test_native_on_evidence_contract():
    run_id = "exp168-r029-compute"
    record = json.loads((Path(".pingstore/runs") / run_id / "run.json").read_text())
    configurations = record["execution"]["configuration"]["conditions"]
    assert set(configurations) == {"AI", "PING"}
    for state in recipe.CONDITIONS:
        snapshot = json.loads(
            (
                Path(".pingstore/runs")
                / run_id
                / "export"
                / f"{state}--start/snapshot.json"
            ).read_text()
        )
        projections = {p["id"] for p in snapshot["network"]["projections"]}
        assert {"E_to_E", "I_to_I"} <= projections
