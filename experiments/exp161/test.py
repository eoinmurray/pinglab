"""Phase discrimination, cue routing, resolved settings and standalone imports."""

import ast
import copy
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from experiments.exp161 import recipe
from experiments.exp161.analyse import measure
from snnlab.meanfield import Trajectory


@pytest.fixture
def cfg():
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
    return recipe.configuration(reference_network=n, incoming_weights_us=weights)


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_numeric_overrides_fail(cfg, value):
    for key in ("coupling_us", "cue_hz", "background_hz"):
        with pytest.raises(ValueError):
            recipe.configuration(
                reference_network=cfg["network"],
                incoming_weights_us=cfg["incoming_weights_us"],
                **{key: value},
            )


def test_configuration_does_not_mutate_imported_settings(cfg):
    n = copy.deepcopy(cfg["network"])
    weights = copy.deepcopy(cfg["incoming_weights_us"])
    changed = recipe.configuration(reference_network=n, incoming_weights_us=weights)
    changed["network"]["populations"]["E"]["size"] = 4
    changed["incoming_weights_us"]["PING_B_E_to_E"] = 100
    assert n["populations"]["E"]["size"] == 8
    assert weights["PING_B_E_to_E"] == 0.395


def test_native_architecture_is_symmetric_and_cue_routes_complementarily(cfg):
    bundle, weights = recipe.author_network(cfg)
    graph = bundle.graph
    assert (
        len(graph["populations"]) == 6
        and len(graph["inputs"]) == 5
        and len(graph["projections"]) == 21
    )
    paths = {p["id"]: p for p in graph["projections"]}
    for c in "AB":
        for p in "EI":
            assert weights[c + "_E_to_C_" + p] == cfg["coupling_us"]
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
    assert paths["cue_A"]["modulation"]["before"] == 1
    assert paths["cue_B"]["modulation"]["after"] == 1


def test_phase_metric_accepts_switch_and_rejects_persistent_old_winner(cfg):
    t = np.arange(0, 2400.1, 0.1)
    switch = cfg["switch_ms"]
    pa = np.where(
        t < switch, 40 * t / 1000, 40 * switch / 1000 + 35 * (t - switch) / 1000
    )
    pb = np.where(
        t < switch, 35 * t / 1000, 35 * switch / 1000 + 40 * (t - switch) / 1000
    )
    pc = 40 * t / 1000
    for actual_c, expected in ((pc, True), (pa, False)):
        state = {
            c + "_E.rate": 20 + 10 * np.cos(2 * np.pi * p)
            for c, p in [("A", pa), ("B", pb), ("C", actual_c)]
        }
        results, _ = measure(Trajectory(t, state), cfg)
        assert results["switched"] is expected
        if expected:
            assert all(r["senders"][r["cued"]]["plv"] > 0.99 for r in results["epochs"])


def test_imports_are_standalone_and_have_no_filesystem_side_effects(tmp_path):
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
                    assert module.split(".")[1] == "exp161"
        if path.name != "test.py":
            assert ".scratch/" not in path.read_text()
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from experiments.exp161 import recipe,compute,analyse,present",
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
