"""Verify physical weight conversion, shared-source wiring and native recording.

Tests use small populations while retaining the one-second measurement window after 50 ms burn-in.
They do not launch the full 5,000-cell experiment or allocate scientific runs.
"""

import numpy as np
import pytest
import torch
from experiments.exp145 import dataset, recipe
from experiments.exp145.compute import recording_signals
from snnlab.sim import (
    DenseArrayBinding,
    NPZRecordingSink,
    Performance,
    RecordingData,
    RecordingSpec,
    SnapshotSpec,
    simulate,
)


def test_recipe_and_area_conversion():
    cfg = recipe.configuration()
    assert (cfg["n_e"], cfg["n_i"], cfg["n_external"]) == (4000, 1000, 4000)
    assert cfg["duration_ms"] - cfg["burn_in_ms"] == 1000
    assert cfg["burn_in_ms"] == 50
    assert cfg["jump_means_us"]["I_to_E"] == pytest.approx(0.008)
    for name, jump in cfg["jump_means_us"].items():
        assert jump * cfg["tau_synapse_ms"] == pytest.approx(
            cfg["reference_coefficients_ns"][name] * 0.001
        )
    other = recipe.configuration()
    cfg["analysis"]["correlation_bins_ms"].append(99)
    assert 99 not in other["analysis"]["correlation_bins_ms"]


def test_shared_source_and_direct_weights():
    cfg = recipe.configuration(n_e=8, n_i=4, n_external=8)
    bundle = recipe.author_network(cfg)
    projections = {p["id"]: p for p in bundle.graph["projections"]}
    assert len(projections) == 6
    assert projections["X_to_E"]["source"] == projections["X_to_I"]["source"]
    assert all(
        p["initialization_scaling"] == "direct" for p in bundle.graph["parameters"]
    )
    inputs = dataset.encode_drive(cfg)
    torch.testing.assert_close(inputs, dataset.encode_drive(cfg), rtol=0, atol=0)
    assert inputs.shape == (21000, 1, 8)


def test_long_window_and_chunked_input_preserve_stream():
    cfg = recipe.configuration(n_e=4, n_i=2, n_external=4, measurement_ms=30000)
    assert cfg["duration_ms"] == 30050
    values = dataset.encode_drive(cfg).numpy()
    rng = np.random.default_rng(cfg["seed"] + cfg["external_seed_offset"])
    expected = rng.random(values.shape, dtype=np.float32)
    np.testing.assert_array_equal(
        values, expected < cfg["external_rate_hz"] * cfg["dt_ms"] / 1000
    )
    for invalid in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="measurement_ms"):
            recipe.configuration(measurement_ms=invalid)


def test_native_conductance_recording(tmp_path):
    torch.set_num_threads(1)
    cfg = recipe.configuration(n_e=4, n_i=2, n_external=4)
    bundle = recipe.author_network(cfg)
    bundle.write(tmp_path / "network.bundle", visualise=False)
    external = dataset.encode_drive(cfg)
    external[:20] = 1  # Known shared events exercise synaptic conductance recording.
    sink = NPZRecordingSink(tmp_path / "recording")
    result = simulate(
        bundle=bundle,
        input_bindings=(DenseArrayBinding("X", external),),
        seed=cfg["seed"],
        device="cpu",
        diagnostics=False,
        performance=Performance(
            compile_mode=None, precision="highest", cuda_graphs="off"
        ),
        recording=RecordingSpec(
            signals=recording_signals(cfg),
            sink=sink,
            sink_id=sink.identity,
            block_steps=1000,
        ),
        snapshot=SnapshotSpec(path=tmp_path / "start", stage="start", parameters=True),
        retain_outputs=False,
        retain_parameters=False,
        retain_final_state=False,
        retain_runtime_state=False,
    )
    recording = RecordingData.read(tmp_path / "recording").select_batch(0)
    from experiments.exp145.custom.recording import read_recording

    compact = read_recording(tmp_path / "recording", tmp_path / "reader")
    assert compact.signals["E.spikes"].dtype == np.bool_
    for name in recording.signals:
        np.testing.assert_array_equal(compact.signals[name], recording.signals[name])
        assert compact.info[name].step_indices == recording.info[name].step_indices
        assert compact.info[name].cell_ids == recording.info[name].cell_ids
    assert recording.signals["E.spikes"].shape == (21000, 4)
    assert "start" in result.snapshots
    assert np.isfinite(recording.signals["X_to_E.conductance"]).all()
    assert recording.signals["X_to_E.conductance"].shape == (21000, 4)


def test_stage_pipeline_in_isolated_store(tmp_path, monkeypatch):
    """Exercise real stage finalization and explicit ancestry on tiny populations."""
    from argparse import Namespace
    from pathlib import Path
    from shutil import copyfile

    from experiments.exp145 import analyse, compute, present
    from pingstore import core

    cfg = recipe.configuration(n_e=4, n_i=2, n_external=4)
    report = tmp_path / "experiments" / "exp145" / "report" / "report.mdx"
    report.parent.mkdir(parents=True)
    copyfile(Path(__file__).parent / "report" / "report.mdx", report)
    monkeypatch.setattr(
        core, "capture_code", lambda *args: dict(git_commit="fixture", dirty=False)
    )
    for module in (compute, analyse, present):
        monkeypatch.setattr(module, "REPO", tmp_path)
    monkeypatch.setattr(recipe, "configuration", lambda **kwargs: cfg)
    compute_id = compute.compute(
        Namespace(seed=42, device="cpu", measurement_ms=1000.0, run_id=None)
    )
    analysis_id = analyse.analyse(Namespace(source=compute_id, run_id=None))
    present_id = present.present(Namespace(source=analysis_id, run_id=None))
    output = core.source_run(
        tmp_path / ".pingstore", present_id, experiment="exp145", stage="present"
    )
    assert output.file("onset.png").is_file()
    assert output.file("ai-diagnostics.png").is_file()
    # New geometry must retain every executed projection and its math annotation.
    import xml.etree.ElementTree as ET

    computed = core.source_run(
        tmp_path / ".pingstore", compute_id, experiment="exp145", stage="compute"
    )
    graph = core.load_json(computed.file("simulation", "start", "snapshot.json"))[
        "network"
    ]
    expected = {row["id"] for row in graph["projections"]}
    styled = ET.parse(output.file("circuit.svg"))
    labels = {
        element.get("data-snnlab-id")
        for element in styled.iter()
        if element.tag.endswith("}svg")
    }
    assert expected <= labels
    refreshed_id = present.redraw_networks(present_id)
    refreshed = core.source_run(tmp_path / ".pingstore", refreshed_id, stage="present")
    assert refreshed.record["inputs"]["figure_source"] == output.reference
    assert refreshed.record["inputs"]["analyse"] == output.record["inputs"]["analyse"]
    for name in ("onset.png", "ai-diagnostics.png", "numbers.json"):
        assert refreshed.file(name).read_bytes() == output.file(name).read_bytes()
    output.check_unchanged()
    assert output.record["inputs"]["compute"]["run_id"] == compute_id


def test_pair_covariance_matches_direct_lagged_pairs():
    from experiments.exp145.analyse import pair_covariance

    rng = np.random.default_rng(7)
    a = rng.normal(size=(40, 4))
    b = rng.normal(size=(40, 4))
    actual = pair_covariance(a, b, 3)
    a -= a.mean(axis=0)
    b -= b.mean(axis=0)
    expected = []
    for lag in range(-3, 4):
        left = a[max(0, -lag) : min(40, 40 - lag)]
        right = b[max(0, lag) : min(40, 40 + lag)]
        expected.append(
            np.mean(
                [
                    np.mean(left[:, i] * right[:, j])
                    for i in range(4)
                    for j in range(4)
                    if i != j
                ]
            )
        )
    np.testing.assert_allclose(actual, expected, atol=1e-14)


def test_sliding_correlations_match_direct_window_sums():
    """Check overlap, window endpoints and exclusion of constant-count cells."""
    from experiments.exp145.analyse import pair_correlations

    cfg = recipe.configuration(n_e=3, n_i=2, n_external=4)
    cfg["analysis"]["correlation_bins_ms"] = [0.2, 0.4, 0.6]
    cfg["analysis"]["correlation_step_ms"] = 0.1
    joined = (np.random.default_rng(8).random((80, 5)) < 0.2).astype(int)
    joined[:, 0] = 0
    joined[:, 4] = 1
    summary, arrays = pair_correlations({"E": joined[:, :3], "I": joined[:, 3:]}, cfg)
    for width in cfg["analysis"]["correlation_bins_ms"]:
        steps = round(width / cfg["dt_ms"])
        counts = np.stack(
            [joined[t : t + steps].sum(0) for t in range(0, 81 - steps, 2)]
        )
        centered = counts - counts.mean(0)
        expected = {"EE": [], "II": [], "EI": []}
        for i in range(5):
            for j in range(i + 1, 5):
                norm = np.sqrt(
                    np.sum(centered[:, i] ** 2) * np.sum(centered[:, j] ** 2)
                )
                if norm:
                    pair = "EE" if j < 3 else "II" if i >= 3 else "EI"
                    expected[pair].append(
                        np.sum(centered[:, i] * centered[:, j]) / norm
                    )
        for pair, values in expected.items():
            np.testing.assert_allclose(
                arrays[f"correlation_{pair}_{width:g}"], values, atol=1e-14
            )
            assert summary[str(width)][pair]["pairs"] == len(values)
            if values:
                assert summary[str(width)][pair]["mean"] == pytest.approx(
                    np.mean(values)
                )
            else:
                assert summary[str(width)][pair]["mean"] is None


def test_current_reconstruction_and_exact_cancellation():
    from experiments.exp145.analyse import current_cancellation, reconstruct_currents

    cfg = recipe.configuration(n_e=4, n_i=2, n_external=4)
    steps = round(cfg["duration_ms"] / cfg["dt_ms"])
    shared = np.broadcast_to(
        (0.001 * (1 + np.sin(np.arange(steps) / 20)))[:, None], (steps, 4)
    ).copy()
    voltage = np.full_like(shared, -60.0)
    signals = {
        "E.voltage": voltage,
        "E_to_E.conductance": shared,
        "X_to_E.conductance": np.zeros_like(shared),
        "I_to_E.conductance": shared * 3,
    }
    e, i = reconstruct_currents(signals, cfg, "E")
    # First current uses the configured resting voltage, later currents use the preceding sample.
    np.testing.assert_allclose(e[0], shared[0] * 70)
    np.testing.assert_allclose(e[1:], shared[1:] * 60)
    np.testing.assert_allclose(i[1:], -shared[1:] * 60)
    summary, curves = current_cancellation(signals, cfg)
    assert summary["zero_lag_nA2"]["EE"] > 0
    assert summary["zero_lag_nA2"]["EI"] < 0
    np.testing.assert_allclose(curves["current_total"], 0, atol=1e-20)
    np.testing.assert_allclose(
        curves["current_total"],
        sum(curves[f"current_{name}"] for name in ["EE", "II", "EI", "IE"]),
        atol=1e-14,
    )


def test_population_scaling_independent_and_shared_activity():
    from experiments.exp145.analyse import population_scaling
    from scipy.linalg import hadamard

    cfg = recipe.configuration(n_e=32, n_i=8, n_external=16)
    # Repeating the same train in every cell produces a plateau Q(K)=1.
    shared = np.tile(np.tile([0, 1], 10000)[:, None], (1, 32)).astype(float)
    # Alternate count blocks rather than equal-count fine bins.
    shared[:10000] = 0
    summary, curves = population_scaling(shared, cfg, seed=42)
    np.testing.assert_allclose(curves["q_median"], 1.0)
    assert abs(summary["loglog_slope"]) < 1e-12

    # Orthogonal balanced count trains have exactly zero cross-cell covariance.
    orthogonal = np.zeros((64 * 400, 32))
    orthogonal[::400] = hadamard(64)[:, 1:33] > 0
    summary, curves = population_scaling(orthogonal, cfg, seed=42)
    np.testing.assert_allclose(
        curves["q_median"], 1 / curves["group_sizes"], atol=1e-14
    )
    assert summary["loglog_slope"] == pytest.approx(-1.0)


def test_sparse_initializer_preserves_connected_physical_strength(tmp_path):
    """Check actual runtime weights, including the small lower-clamping bias."""
    from math import erf, exp, pi, sqrt

    cfg = recipe.configuration(n_e=128, n_i=64, n_external=128)
    bundle = recipe.author_network(cfg)
    result = simulate(
        bundle=bundle,
        input_bindings=(DenseArrayBinding("X", torch.zeros(1, 1, 128)),),
        seed=cfg["seed"],
        device="cpu",
        diagnostics=False,
        performance=Performance(
            compile_mode=None, precision="highest", cuda_graphs="off"
        ),
        snapshot=SnapshotSpec(path=tmp_path / "start", stage="start", parameters=True),
        retain_outputs=False,
        retain_parameters=False,
        retain_final_state=False,
        retain_runtime_state=False,
    )
    relative_sd = cfg["weight_relative_sd"]
    alpha = 1 / relative_sd
    phi = exp(-alpha * alpha / 2) / sqrt(2 * pi)
    cdf = (1 + erf(alpha / sqrt(2))) / 2
    positive_mean_factor = 1 + relative_sd * phi / cdf
    for name, target in cfg["jump_means_us"].items():
        weights = result.snapshots["start"].parameters[name + ".weight"]
        positive = weights[weights > 0]
        assert float(positive.mean()) == pytest.approx(
            target * positive_mean_factor, rel=0.06
        )
        assert float((weights > 0).float().mean()) == pytest.approx(
            cfg["connection_probability"] * cdf, abs=0.02
        )
