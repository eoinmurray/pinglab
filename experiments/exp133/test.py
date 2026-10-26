"""Scientific diagnostics and native paired-grid execution invariants.

Run with pytest from the repository root. Small synthetic inputs check the
scientific invariants and native runtime integration without replacing the
report's pinned historical evidence.
"""

import numpy as np
import pytest
import torch
from experiments.exp133 import analyse, recipe
from experiments.exp133.present import edges
from snnlab.sim import DenseArrayBinding, NetworkSnapshot, Performance, simulate


def native(bundle, drives, coupling, state=None):
    return simulate(
        graph=bundle.graph,
        seed=recipe.NETWORK_SEED,
        device="cpu",
        input_bindings=tuple(
            DenseArrayBinding(k, torch.from_numpy(v).float())
            for (k, v) in drives.items()
        ),
        inference_overrides={"projection_scales": recipe.coupling_scales(coupling)},
        runtime_state=state,
        performance=Performance(compile_mode=None),
    )


def test_native_zero_prefix_and_step_gain_match_monolithic_execution(monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(recipe, "COUPLING_ONSET_MS", 0.4)
    bundle = recipe.author_network()
    drive = {
        key: np.ones((100, 2, recipe.N_INPUT), dtype=np.uint8)
        for key in ("drive_A", "drive_B")
    }
    full = native(bundle, drive, 0.06)
    prefix = native(bundle, {key: value[:4] for key, value in drive.items()}, 0.0)
    saved = prefix.runtime_state.detached()
    preserved = {key: value.clone() for key, value in saved.voltages.items()}
    split = native(
        bundle, {key: value[4:] for key, value in drive.items()}, 0.06, saved.detached()
    )
    for name in full.diagnostics:
        torch.testing.assert_close(
            torch.cat([prefix.diagnostics[name], split.diagnostics[name]]),
            full.diagnostics[name],
            rtol=0,
            atol=0,
        )
    for name in full.runtime_state.voltages:
        torch.testing.assert_close(
            full.runtime_state.voltages[name],
            split.runtime_state.voltages[name],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            saved.voltages[name], preserved[name], rtol=0, atol=0
        )
    assert sum(value.sum().item() for value in full.diagnostics.values()) > 0
    for name, value in full.parameters.items():
        if any(name == f"{cross}.weight" for cross in recipe.cross_ids()):
            np.testing.assert_array_equal((value.numpy() != 0).sum(axis=0), 8)
            np.testing.assert_allclose(value.numpy().sum(axis=0), 0.06, atol=1e-7)
        else:
            torch.testing.assert_close(value, prefix.parameters[name], rtol=0, atol=0)


def test_input_pairing_and_signed_grid():
    first, second = recipe.make_inputs(steps=100), recipe.make_inputs(steps=100)
    rates = recipe.rates()
    np.testing.assert_allclose((rates["drive_A"] + rates["drive_B"]) / 2, 25)
    np.testing.assert_allclose(
        rates["drive_A"] - rates["drive_B"], recipe.INPUT_DIFFERENCES_HZ
    )
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])
        assert first[name].shape == (100, 11, 1024)
    assert np.all(np.diff(first["drive_A"].astype(int), axis=1) >= 0)
    assert np.all(np.diff(first["drive_B"].astype(int), axis=1) <= 0)
    with pytest.raises(ValueError):
        recipe.coupling_scales(-0.01)


def pulse_train(period):
    spikes = np.zeros((round(recipe.T_MS / recipe.DT_MS), recipe.N_E), dtype=np.uint8)
    spikes[100::period] = 1
    return spikes


def test_phase_estimator_distinguishes_fixed_offset_detuning_and_silence():
    first = pulse_train(200)
    locked, _ = analyse.measure_pair(first, np.roll(first, 30, axis=0))
    assert locked["valid"] and locked["locked"]
    assert locked["concentration"] > 0.999
    assert abs(locked["drift_hz"]) < 1e-8
    drifting, _ = analyse.measure_pair(first, pulse_train(210))
    assert drifting["valid"] and not drifting["locked"]
    assert drifting["drift_hz"] == pytest.approx(50 - 1000 / 21, abs=0.02)
    quiet, trace = analyse.measure_pair(np.zeros_like(first), first)
    assert not quiet["valid"] and not quiet["locked"]
    assert quiet["concentration"] is None
    assert len(trace["time_ms"]) == 0


def test_concentration_and_cancelled_drift_do_not_hide_a_complete_excursion():
    time = np.arange(10000) * recipe.DT_MS
    difference = np.zeros_like(time)
    difference[5000] = 2 * np.pi
    metrics = analyse.phase_metrics(time, difference)
    assert metrics["concentration"] == pytest.approx(1)
    assert abs(metrics["drift_hz"]) < 0.25
    assert not metrics["locked"]
    assert metrics["phase_span_cycles"] == pytest.approx(1)


def test_measured_coordinate_edges_reject_duplicates_and_preserve_nonuniform_spacing():
    np.testing.assert_allclose(edges([-3, -1, 4]), [-4, -2, 1.5, 6.5])
    with pytest.raises(ValueError, match="ordered"):
        edges([0, 0, 1])


def test_native_grid_survives_v4_storage_and_checks_actual_pairing(
    tmp_path, monkeypatch
):
    import copy
    import json
    import subprocess
    from dataclasses import replace

    from experiments.exp133 import compute
    from experiments.exp133.custom import evidence
    from pingstore.contracts import PingstoreError
    from pingstore.stages import source_run

    monkeypatch.setattr(recipe, "INPUT_DIFFERENCES_HZ", np.array([-20.0, 20.0]))
    monkeypatch.setattr(recipe, "K_VALUES_US", np.array([0.0, 0.06]))
    monkeypatch.setattr(recipe, "COUPLING_ONSET_MS", 0.4)
    monkeypatch.setattr(recipe, "T_MS", 10.0)
    monkeypatch.setattr(compute, "REPO", tmp_path)
    monkeypatch.setattr(
        "pingstore.stages.memberships", lambda repo: {recipe.SLUG: "demo"}
    )
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "Native grid fixture",
            "--allow-empty",
        ],
        cwd=tmp_path,
        check=True,
    )
    identity = compute.compute()
    source = source_run(tmp_path / ".pingstore", identity)
    assert evidence.compute_contract(source) == recipe.configuration()
    corrupted = copy.deepcopy(source.record)
    corrupted["execution"]["simulations"]["coupling-060"]["batches"] = 999
    with pytest.raises(PingstoreError, match="coverage"):
        evidence.compute_contract(replace(source, record=corrupted))
    mapping = source.record["scientific_artifacts"]["native_paths"]
    assert [key for key in mapping if key.endswith(".svg")] == [
        "paired-ping.bundle/reports/circuit.svg"
    ]
    assert not any("/end/" in key for key in mapping)
    path = source.export / mapping["coupling-060-execution/start/snapshot.json"]
    metadata = json.loads(path.read_text())
    corrupted = copy.deepcopy(metadata)
    corrupted["parameter_file_sha256"] = "0" * 64
    path.write_text(json.dumps(corrupted))
    with pytest.raises(ValueError, match="parameter file digest mismatch"):
        evidence.compute_contract(source)
    metadata["request"]["input_bindings"][0]["value"]["sha256"] = "0" * 64
    path.write_text(json.dumps(metadata))
    with pytest.raises(PingstoreError, match="paired inputs"):
        evidence.compute_contract(source)


def test_sender_receiver_reciprocal_coupling_switches_without_reset(monkeypatch):
    from experiments.exp133.custom import sender_receiver

    torch.set_num_threads(1)
    monkeypatch.setattr(sender_receiver, "ONSET_MS", 10.0)
    coupled = sender_receiver.author_network()
    monkeypatch.setattr(sender_receiver, "ONSET_MS", 100.0)
    uncoupled = sender_receiver.author_network()
    drives = {
        f"drive_{name}": torch.ones(400, 1, sender_receiver.N_INPUT)
        for name in ("S", "R")
    }

    def run(bundle):
        return simulate(
            graph=bundle.graph,
            seed=recipe.NETWORK_SEED,
            device="cpu",
            input_bindings=tuple(
                DenseArrayBinding(key, value) for (key, value) in drives.items()
            ),
            performance=Performance(compile_mode=None),
        )

    enabled, control = run(coupled), run(uncoupled)
    for name in ("S", "R"):
        for pop in ("E", "I"):
            torch.testing.assert_close(
                enabled.diagnostics[f"{name}_{pop}"][:100],
                control.diagnostics[f"{name}_{pop}"][:100],
                rtol=0,
                atol=0,
            )
        assert any(
            not torch.equal(
                enabled.diagnostics[f"{name}_{pop}"],
                control.diagnostics[f"{name}_{pop}"],
            )
            for pop in ("E", "I")
        )
        for target in ("E", "I"):
            origin = "R" if name == "S" else "S"
            weights = enabled.parameters[
                f"PING_{origin}_E_to_PING_{name}_{target}.weight"
            ].numpy()
            np.testing.assert_allclose(weights.sum(axis=0), 0.09, atol=1e-7)
            np.testing.assert_array_equal((weights != 0).sum(axis=0), 8)


def test_raster_frequencies_use_exact_first_and_last_windows():
    spikes = np.zeros((30000, 1, recipe.N_E), dtype=np.uint8)
    spikes[100:5000:200] = 1
    spikes[10100:15000:250] = 1
    spikes[25100:30000:160] = 1
    measured = analyse.raster_frequency_windows(
        {"S_E": spikes, "R_E": np.zeros_like(spikes)}
    )
    assert measured["S"]["first"]["frequency_hz"] == pytest.approx(50)
    assert measured["S"]["last"]["frequency_hz"] == pytest.approx(62.5)
    assert measured["S"]["first"]["valid"]
    assert measured["S"]["last"]["valid"]
    assert measured["R"]["first"]["frequency_hz"] is None
    assert not measured["R"]["last"]["valid"]


def test_input_expansion_preserves_recurrent_weights_and_summed_drive(
    tmp_path, monkeypatch
):
    import subprocess

    from experiments.exp133 import compute
    from experiments.exp133.custom import evidence
    from pingstore.stages import source_run

    monkeypatch.setattr(recipe, "INPUT_DIFFERENCES_HZ", np.array([-4.0, 4.0]))
    monkeypatch.setattr(recipe, "K_VALUES_US", np.array([0.0, 0.09, 0.12]))
    monkeypatch.setattr(recipe, "COUPLING_ONSET_MS", 0.4)
    monkeypatch.setattr(recipe, "T_MS", 10.0)
    monkeypatch.setattr(recipe, "N_INPUT", 128)
    monkeypatch.setattr(compute, "REPO", tmp_path)
    monkeypatch.setattr(
        "pingstore.stages.memberships", lambda repo: {recipe.SLUG: "demo"}
    )
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "Input expansion fixture",
            "--allow-empty",
        ],
        cwd=tmp_path,
        check=True,
    )
    baseline = compute.compute()
    original = evidence.parameter_reference(
        source_run(tmp_path / ".pingstore", baseline)
    )
    monkeypatch.setattr(recipe, "N_INPUT", 1024)
    expanded = compute.compute(parameter_source=baseline)
    current = source_run(tmp_path / ".pingstore", expanded)
    evidence.compute_contract(current)
    values = {
        name: tensor.numpy()
        for name, tensor in NetworkSnapshot.read(
            current.file("coupling-120-execution", "start", "snapshot.json")
        )
        .load_parameters()
        .items()
    }
    for key, value in original.items():
        if "input_to_E" in key:
            assert values[key].shape == (1024, recipe.N_E)
            np.testing.assert_allclose(
                values[key].sum(axis=0, dtype=np.float64),
                value.sum(axis=0, dtype=np.float64),
                atol=1e-7,
            )
        else:
            np.testing.assert_array_equal(values[key], value)

    from experiments.exp133.custom import sender_receiver

    monkeypatch.setattr(sender_receiver, "DURATION_MS", 10.0)
    transferred = compute.compute(protocol="sender-receiver", parameter_source=expanded)
    sr = source_run(tmp_path / ".pingstore", transferred)
    sender_receiver.compute_contract(sr)
    actual = {
        name: tensor.numpy()
        for name, tensor in NetworkSnapshot.read(
            sr.file("sender-receiver-execution", "start", "snapshot.json")
        )
        .load_parameters()
        .items()
    }
    expected = sender_receiver.transferred_parameters(current)
    assert set(actual) == set(expected)
    for key in expected:
        np.testing.assert_allclose(actual[key], expected[key], rtol=1e-6, atol=1e-9)
    assert sr.record["execution"]["configuration"]["onset_ms"] == 500.0
    assert len(actual) == 10
