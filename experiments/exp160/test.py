"""Matched inputs, native conductance timing, phase metrics and stage independence."""

import ast
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from experiments.exp160 import analyse, recipe
from snnlab.meanfield import Trajectory
from snnlab.sim import (
    DenseArrayBinding,
    Performance,
    RecordingSpec,
    SignalRecording,
    simulate,
)


@pytest.fixture
def cfg():
    network = dict(
        dt_ms=0.1,
        network_seed=85,
        input_channels=1024,
        mean_input_hz=25.0,
        base_coupling_us=0.12,
        onset_ms=1.0,
        resting_mv=-65.0,
        threshold_mv=-50.0,
        reset_mv=-65.0,
        populations=dict(
            E=dict(size=80, capacitance_nf=1.0, leak_us=0.05, refractory_ms=3.0),
            I=dict(size=20, capacitance_nf=0.5, leak_us=0.1, refractory_ms=1.5),
        ),
        sensory=dict(mean_us=0.2, std_us=0.03, gain=9.2, tau_ms=2.0),
        local=dict(
            E_to_I=dict(mean_us=0.75, std_us=0.075, tau_ms=1.0),
            I_to_E=dict(mean_us=1.5, std_us=0.15, tau_ms=9.0),
            E_to_E=dict(total_us=0.4, tau_ms=2.0),
            I_to_I=dict(total_us=0.8, tau_ms=6.0),
        ),
        cross=dict(tau_ms=0.5, delay_ms=2.0, zero_fraction=0.9),
    )
    return recipe.configuration(network=network)


def test_inputs_are_reproducible_private_and_unchanged_by_coupling(cfg):
    cfg["duration_ms"] = 20
    a = recipe.make_inputs(cfg)
    cfg["coupling_us"] = 0.12
    b = recipe.make_inputs(cfg)
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])
    assert not np.array_equal(a["drive_A"], a["drive_B"])
    assert a["drive_A"].dtype == np.uint8


def test_native_pair_retains_delayed_conductance_and_preserves_sender(cfg):
    cfg["duration_ms"] = 8
    bundle = recipe.author_network(cfg)
    assert len(bundle.graph["projections"]) == 12
    assert not any(
        p["source"].startswith("PING_B") and p["target"].startswith("PING_A")
        for p in bundle.graph["projections"]
    )
    bindings = tuple(
        DenseArrayBinding(
            f"drive_{c}", torch.ones(80, 1, cfg["network"]["input_channels"])
        )
        for c in ("A", "B")
    )
    results = []
    for k in (0, 0.24):
        result = simulate(
            graph=bundle.graph,
            seed=85,
            device="cpu",
            input_bindings=bindings,
            inference_overrides={"projection_scales": recipe.scales(cfg, k)},
            performance=Performance(compile_mode=None),
            recording=RecordingSpec(
                signals=[SignalRecording(s) for s in cfg["recording_signals"]]
            ),
        )
        results.append(result)
    np.testing.assert_array_equal(
        results[0].diagnostics["A_E"], results[1].diagnostics["A_E"]
    )
    gx = results[1].recorded_signals[cfg["recording_signals"][0]].numpy()
    assert (
        gx.shape == (80, 1, cfg["network"]["populations"]["E"]["size"])
        and np.isfinite(gx).all()
        and np.min(gx) >= 0
    )
    source = results[1].diagnostics["A_E"].numpy().sum(axis=(1, 2))
    first_source = np.flatnonzero(source)[0]
    first_arrival = np.flatnonzero(gx.sum(axis=(1, 2)))[0]
    assert first_arrival - first_source in (20, 21)
    assert not results[0].recorded_signals[cfg["recording_signals"][0]].any()


def test_phase_metrics_distinguish_drift_and_locking_and_retain_late_markers(cfg):
    t = np.arange(0, 1500, 0.1)
    for frequency, locked in ((50, True), (51, False)):
        state = {
            "PING_A_E.rate": 20 + 10 * np.cos(2 * np.pi * 50 * t / 1000),
            "PING_B_E.rate": 20 + 10 * np.cos(2 * np.pi * frequency * t / 1000 + 0.3),
            "PING_B_I_to_E.conductance": 0.1
            + 0.03 * np.cos(2 * np.pi * frequency * t / 1000),
        }
        metrics, diagnostics = analyse.measure(Trajectory(t, state), cfg)
        assert metrics["locked"] is locked
        assert diagnostics["arrivals"].max() > 1450


def test_imports_have_no_side_effects_or_cross_experiment_code(tmp_path):
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
                    assert module.split(".")[1] == "exp160"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from experiments.exp160 import recipe,compute,analyse,present",
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


def test_alignment_rejects_coincident_or_previous_cycle_output(cfg):
    from types import SimpleNamespace

    dt = cfg["network"]["dt_ms"]
    t = np.arange(0, 1500, dt)
    centers = np.arange(1010, 1450, 25)
    gi = 0.1 + 0.9 * (1 + np.cos(2 * np.pi * (t - 1000) / 25)) / 2
    gx = sum(np.exp(-(((t - c) / 0.2) ** 2)) for c in centers)
    # Place the input in a low-inhibition part of each cycle.
    gi[:] = 1
    for c in centers:
        gi[np.abs(t - c) < 3] = 0.1
    solution = SimpleNamespace(
        time_ms=t,
        state={
            "PING_B_I_to_E.conductance": gi,
            "PING_A_E_to_PING_B_E_K_EE.conductance": gx,
        },
    )
    for lag, passed in ((1.0, True), (0.0, False), (-1.0, False)):
        spikes = np.zeros((len(t), 1, 8), dtype=np.uint8)
        for c in centers:
            spikes[round((c + lag) / dt), 0, :] = 1
        result = analyse.alignment(solution, {"B_E": spikes}, cfg)
        assert result["timing_passed"] is passed
        np.testing.assert_allclose(
            [r["output_lag_ms"] for r in result["per_volley"]], lag, atol=1e-10
        )


def test_selected_configuration_and_weights_match_recorded_execution():
    """Authenticate the trimmed recipe without running another simulation."""
    from pingstore.library import source_run
    from snnlab.sim import NetworkSnapshot

    root = Path(__file__).resolve().parents[2]
    reference = source_run(
        root / ".pingstore", "exp159-r001-compute", experiment="exp159", stage="compute"
    )
    selected = source_run(
        root / ".pingstore", "exp160-r068-compute", experiment="exp160", stage="compute"
    )
    cfg = recipe.configuration(
        network=reference.record["execution"]["configuration"]["network"]
    )
    assert cfg == selected.record["execution"]["configuration"]
    original = NetworkSnapshot.read(
        reference.file(recipe.REFERENCE_SNAPSHOT)
    ).load_parameters()
    before = {key: value.clone() for key, value in original.items()}
    base = recipe.reference_parameters(original, cfg)
    for name, strength in (("control", 0), ("coupled", cfg["coupling_us"])):
        recorded = NetworkSnapshot.read(
            selected.file(f"{name}-execution/start/snapshot.json")
        ).load_parameters()
        scales = recipe.scales(cfg, strength)
        for key, value in base.items():
            scale = scales.get(key.removesuffix(".weight"), 1)
            torch.testing.assert_close(
                value * scale, recorded[key], rtol=1e-6, atol=1e-9
            )
            torch.testing.assert_close(original[key], before[key], rtol=0, atol=0)
    assert ((base["PING_A_input_to_E.weight"] > 0).sum(0) == 128).all()
    assert ((base["PING_A_E_to_I.weight"] > 0).sum(0) == 80).all()
    reference.check_unchanged()
    selected.check_unchanged()
