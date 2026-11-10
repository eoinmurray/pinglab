"""Focused non-simulation checks of scientific boundaries and input generation."""

import ast
from pathlib import Path

import numpy as np
import pytest
import torch
from experiments.exp171 import recipe
from experiments.exp171.analyse import summarise
from experiments.exp171.compute import (
    artificial,
    calibrate,
    digest,
    filtered,
    sensory,
    source_indices,
    timing_endpoints,
    timing_events,
)
from snnlab.sim import plan_graph


def fixed_configuration(**overrides):
    """Historical fixed-count fixtures must retain their recorded semantics."""
    overrides.setdefault("inhibition_method", "count_matched_timing_v2")
    return recipe.configuration(**overrides)


def test_poisson_protocol_is_default_for_every_ceiling():
    from experiments.exp171.compute import check_calibration_configuration

    for cfg in recipe.sweep_configurations():
        assert cfg["inhibition_method"] == recipe.POISSON_METHOD
        assert cfg["target_events_per_source"] is None
    with pytest.raises(ValueError, match="fixed count"):
        recipe.configuration(target_events_per_source=24)
    old, new = fixed_configuration(), recipe.configuration()
    for cfg in (old, new):
        cfg.update(source_checkpoints={}, dataset_sha256={})
    with pytest.raises(ValueError, match="inhibition_method"):
        check_calibration_configuration(old, new)


def test_poisson_calibration_matches_stationary_moments_without_timing_search():
    from experiments.exp171.compute import poisson_expected_moments

    cfg = recipe.configuration(n_e=3, bracket_attempts=1)
    target = np.zeros((cfg["steps"], 3))
    target[300:320, 1] = 1
    target[1000:1005, 2] = 2
    rates, amplitude, mean, sd, fitted_mean, fitted_sd, error = calibrate(
        target, cfg, 17
    )
    assert rates[0] == amplitude[0] == 0
    assert np.all(rates[1:] > 0) and np.all(amplitude[1:] > 0)
    expected_mean, expected_sd = poisson_expected_moments(rates, amplitude, cfg)
    np.testing.assert_allclose(expected_mean, mean, rtol=1e-6)
    np.testing.assert_allclose(expected_sd, sd, rtol=1e-6)
    np.testing.assert_array_equal(fitted_mean, expected_mean)
    np.testing.assert_array_equal(fitted_sd, expected_sd)
    assert error < 1e-6
    with pytest.raises(ValueError, match="Constant positive"):
        calibrate(np.ones((100, 3)), cfg, 17)


@pytest.mark.parametrize("history_ms", [0, 0.1, 60])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_poisson_draws_preserve_counts_seeds_and_native_initial_state(
    history_ms, device
):
    from experiments.exp171.compute import artificial_device

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("Poisson GPU check requires CUDA")
    cfg = recipe.configuration(n_e=4, inhibition_prehistory_ms=history_ms)
    rates = np.tile([0.0, 20.0, 2000.0, 500.0], (3, 1))
    amp = np.tile(np.array([0.0, 0.02, 0.01, 0.03], dtype=np.float32), (3, 1))
    ids = np.array([17, 42, 66])
    full, g, pending = artificial_device(
        rates, amp, ids, cfg, device, draw=0, retain_history=True
    )
    scored, compact_g, compact_pending = artificial_device(
        rates, amp, ids, cfg, device, draw=0
    )
    h = cfg["inhibition_prehistory_steps"]
    torch.testing.assert_close(scored, full[h:], rtol=0, atol=0)
    torch.testing.assert_close(g, compact_g, rtol=0, atol=0)
    torch.testing.assert_close(pending, compact_pending, rtol=0, atol=0)
    expected_g = filtered(full.cpu().numpy()[:h], cfg)[-1] if h else np.zeros((3, 4))
    np.testing.assert_allclose(g.cpu(), expected_g, rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(
        pending.cpu(), full[h - 1 : h].cpu() if h else np.zeros((1, 3, 4))
    )
    counts = (scored[:, :, 2] / amp[0, 2]).round().cpu().numpy()
    assert counts.max() > 1 and np.unique(counts.sum(0)).size > 1
    assert counts.mean() == pytest.approx(0.2, rel=0.1)
    assert counts.var() == pytest.approx(0.2, rel=0.1)
    assert not full[:, :, 0].any()
    subset = artificial_device(
        rates[[2, 0]], amp[[2, 0]], ids[[2, 0]], cfg, device, draw=0
    )[0]
    torch.testing.assert_close(subset, scored[:, [2, 0]], rtol=0, atol=0)
    assert not torch.equal(
        scored, artificial_device(rates, amp, ids, cfg, device, epoch=1)[0]
    )


def test_poisson_dataset_calibration_and_replay_keep_multiplicity(tmp_path):
    from types import SimpleNamespace

    from experiments.exp171.analyse import (
        native_replay_events,
        poisson_rate_summary,
        validate_native_dataset,
    )
    from experiments.exp171.compute import (
        artificial_device,
        poisson_expected_moments,
        reuse_calibration,
    )

    cfg = recipe.configuration(train_images=1, validation_images=1, n_e=2, n_i=2)
    cfg.update(source_checkpoints={}, dataset_sha256={})
    rates = np.array([[0.0, 1000.0], [1200.0, 1500.0]])
    amp = np.array([[0.0, 0.03], [0.01, 0.02]], dtype=np.float32)
    mean, sd = poisson_expected_moments(rates, amp, cfg)
    data = dict(
        ids=np.array([17, 42]),
        labels=[1, 2],
        features=np.zeros((2, 784), dtype=np.uint8),
        source_train=[17],
        source_validation=[42],
        poisson_rate_hz=rates,
        amplitude=amp,
    )
    cal = dict(
        ids=data["ids"],
        target_mean=mean,
        target_sd=sd,
        expected_mean=mean,
        expected_sd=sd,
        max_relative_sd_error=np.zeros(2),
        teacher_i_spike_count=np.full((2, 2), 10),
        teacher_i_rate_hz=np.full((2, 2), 50.0),
    )
    np.savez(tmp_path / "dataset.npz", **data)
    np.savez(tmp_path / "calibration.npz", **cal)
    computed = SimpleNamespace(file=lambda name: tmp_path / name)
    assert validate_native_dataset(computed, cfg) == (0, 0)
    assert poisson_rate_summary(computed, cfg)["validation"]["mean"] == 1350
    restored = []
    computed.record = {"execution": {"configuration": dict(cfg)}}
    computed.restore_bundle = lambda *args: restored.append(args)
    computed.check_unchanged = lambda: None
    pixels = np.zeros((43, 784), dtype=np.uint8)
    labels = np.zeros(43, dtype=np.int64)
    labels[data["ids"]] = data["labels"]
    cached = reuse_calibration(
        computed, cfg, data["ids"], pixels, labels, tmp_path / "teacher"
    )
    np.testing.assert_array_equal(cached[0], rates)
    np.testing.assert_array_equal(cached[1], amp)
    assert len(restored) == 1
    events = artificial_device(
        rates, amp, data["ids"], cfg, "cpu", draw=0, retain_history=True
    )[0].numpy()
    counts = np.rint(
        np.divide(events, amp[None], out=np.zeros_like(events), where=amp[None] > 0)
    ).astype(np.uint32)
    assert counts.max() > 1
    np.testing.assert_array_equal(
        native_replay_events(data, {"batch_artificial_counts": counts}, cfg, 0, 2),
        events,
    )
    data["poisson_rate_hz"] = rates + 1
    np.savez(tmp_path / "dataset.npz", **data)
    with pytest.raises(ValueError, match="Invalid Poisson"):
        validate_native_dataset(computed, cfg)


def test_population_mean_counts_round_half_up_without_a_floor_of_one():
    from experiments.exp171.compute import teacher_event_counts

    cfg = recipe.configuration(n_i=4)
    counts = np.array(
        [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1], [2, 3, 2, 3], [20, 21, 20, 21]],
        dtype=np.int32,
    )
    actual = teacher_event_counts(counts, cfg)
    np.testing.assert_array_equal(actual, [0, 0, 1, 3, 21])
    assert actual.dtype == np.int32
    rate_error = (actual - counts.mean(1)) / (cfg["duration_ms"] / 1000)
    assert np.max(np.abs(rate_error)) <= 2.5
    for invalid in (counts[:, :2], counts.astype(float) + 0.1, counts - 1):
        with pytest.raises(ValueError):
            teacher_event_counts(invalid, cfg)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_variable_counts_reach_device_draws_and_prehistory(device):
    from experiments.exp171.compute import artificial_device

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA event-count check requires a GPU")
    cfg = recipe.configuration(inhibition_method=recipe.TEACHER_COUNT_METHOD, n_e=3)
    counts = np.array([0, 1, 7, 31], dtype=np.int32)
    ids = np.array([12, 81, 24, 51])
    alpha, amplitude = np.full((4, 3), 1.4), np.full((4, 3), 0.02, dtype=np.float32)
    amplitude[0] = 0
    full, conductance, pending = artificial_device(
        alpha, amplitude, ids, cfg, device, counts=counts, draw=0, retain_history=True
    )
    history = cfg["inhibition_prehistory_steps"]
    np.testing.assert_array_equal(
        (full[history:] > 0).sum(0).cpu(), np.repeat(counts[:, None], 3, axis=1)
    )
    assert not full[:, 0].any() and not conductance[0].any() and not pending[:, 0].any()
    np.testing.assert_allclose(
        conductance.cpu(),
        filtered(full.cpu().numpy()[:history], cfg)[-1],
        rtol=1e-5,
        atol=1e-7,
    )
    order = np.array([3, 1])
    subset = artificial_device(
        alpha[order],
        amplitude[order],
        ids[order],
        cfg,
        device,
        counts=counts[order],
        draw=0,
        retain_history=True,
    )[0]
    torch.testing.assert_close(subset, full[:, order], rtol=0, atol=0)
    numpy_events = artificial(alpha, amplitude, ids, cfg, counts=counts, draw=0)
    np.testing.assert_array_equal(
        (numpy_events[history:] > 0).sum(0), np.repeat(counts[:, None], 3, axis=1)
    )
    with pytest.raises(ValueError, match="requires saved"):
        artificial_device(alpha, amplitude, ids, cfg, device, draw=0)


@pytest.mark.parametrize("count", [7, 29])
def test_calibration_fits_the_requested_teacher_derived_count(count):
    from experiments.exp171.compute import clustered_endpoint, rng, timing_trial

    cfg = recipe.configuration(inhibition_method=recipe.TEACHER_COUNT_METHOD, n_e=3)
    gen = rng(cfg, "calibration", 17)
    endpoints = timing_endpoints(gen, cfg["steps"], 3, count)
    past = timing_endpoints(gen, cfg["steps"], 3, count)
    clusters = tuple(clustered_endpoint(gen, cfg["steps"], 3, count) for _ in range(2))
    events = timing_trial(*endpoints, np.full(3, 2.0), cfg, past, clusters)
    target = filtered(events, cfg)[cfg["inhibition_prehistory_steps"] :] * np.array(
        [0.01, 0.02, 0.03]
    )
    alpha, q, mu, sd, fm, fs, error = calibrate(target, cfg, 17, event_count=count)
    np.testing.assert_allclose(mu, fm, rtol=1e-6)
    assert np.max(np.abs(fs / sd - 1)) < 0.01
    assert np.all(q > 0) and np.all(error < 0.01)


def test_zero_count_is_silent_and_cannot_fake_positive_conductance():
    from experiments.exp171.compute import relative_moment_error

    cfg = recipe.configuration(inhibition_method=recipe.TEACHER_COUNT_METHOD, n_e=3)
    zero = np.zeros((cfg["steps"], 3))
    fitted = calibrate(zero, cfg, 17, event_count=0)
    assert all(not np.any(a) for a in fitted)
    assert not relative_moment_error(np.zeros(3), np.zeros(3)).any()
    with pytest.raises(ValueError, match="rounds to zero"):
        calibrate(zero + 0.001, cfg, 17, event_count=0)


def test_analysis_checks_derived_counts_and_reports_rounding(tmp_path):
    from types import SimpleNamespace

    from experiments.exp171.analyse import event_count_summary, validate_native_dataset

    cfg = recipe.configuration(
        inhibition_method=recipe.TEACHER_COUNT_METHOD,
        train_images=2,
        validation_images=1,
        n_i=2,
        n_e=3,
    )
    counts = np.array([[4, 2], [0, 0], [5, 6]], dtype=np.int32)
    event_count = np.array([3, 0, 6], dtype=np.int32)
    moments = np.ones((3, 3))
    moments[1] = 0
    dataset = dict(
        ids=[12, 17, 23],
        labels=[1, 2, 3],
        features=np.zeros((3, 784)),
        source_train=[12, 17],
        source_validation=[23],
        timing_interpolation=np.full((3, 3), 0.5),
        amplitude=moments * 0.02,
        event_count=event_count.copy(),
    )
    cal = dict(
        ids=dataset["ids"],
        teacher_i_spike_count=counts,
        teacher_i_rate_hz=counts.astype(np.float32) / 0.2,
        target_mean=moments.copy(),
        target_sd=moments.copy(),
        fitted_mean=moments.copy(),
        fitted_sd=moments.copy(),
        event_count=event_count.copy(),
    )
    computed = SimpleNamespace(file=lambda name: tmp_path / name)

    def save():
        np.savez(tmp_path / "dataset.npz", **dataset)
        np.savez(tmp_path / "calibration.npz", **cal)

    save()
    assert validate_native_dataset(computed, cfg) == (0, 0)
    summary = event_count_summary(computed, cfg)
    assert summary["train"]["mean"] == 1.5
    assert summary["validation"]["mean"] == 6
    assert summary["validation"]["max_rate_rounding_error_hz"] == 2.5
    dataset["event_count"][2] = 24
    save()
    with pytest.raises(AssertionError):
        validate_native_dataset(computed, cfg)
    dataset["event_count"][2] = 6
    cal["target_mean"][1] = 0.1
    save()
    with pytest.raises(ValueError, match="Zero-count images"):
        validate_native_dataset(computed, cfg)


def test_dependency_boundary():
    root = Path(__file__).parent
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            names = (
                [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else []
            )
            for name in names:
                assert not name.startswith("experiments.exp") or name.startswith(
                    "experiments.exp171"
                )
        assert not any(
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "__import__"
            for n in ast.walk(tree)
        )
    cfg = fixed_configuration()
    assert cfg["source_run"] == "exp022-r015-compute"
    assert cfg["rate_penalty"] == 0


@pytest.mark.parametrize("seed", recipe.SOURCE_SEEDS)
def test_matched_sweep_sources_and_native_penalties(seed):
    configurations = recipe.sweep_configurations(seed=seed)
    assert [c["rate_target_hz"] for c in configurations] == list(recipe.RATE_TARGETS_HZ)
    for cfg in configurations:
        target = cfg["rate_target_hz"]
        label = recipe.target_label(target)
        assert cfg["coba_cell"] == f"coba__{label}__seed{seed}"
        assert cfg["ping_cell"] == f"ping__{label}__seed{seed}"
        regularizers = recipe.author_network(cfg).training["regularizers"]
        assert not recipe.author_network(cfg, teacher=True).training
        if target is None:
            assert not regularizers and cfg["rate_penalty"] == 0
        else:
            assert regularizers == [
                {
                    "kind": "spike_budget",
                    "signals": ["E.spikes"],
                    "strength": 0.041,
                    "config": {
                        "ceiling": {"value": target, "unit": "Hz"},
                        "penalty": "squared_hinge",
                        "aggregation": "mean_presentations_then_layers_of_population_mean_rate",
                    },
                }
            ]
            assert cfg["rate_penalty"] == 0.041


@pytest.mark.parametrize(
    "overrides",
    [
        {"rate_target_hz": 0},
        {"rate_target_hz": True},
        {"rate_target_hz": float("nan")},
        {"rate_target_hz": 5, "rate_penalty": 0},
        {"seed": 45},
        {"rate_target_hz": 5, "ping_cell": "ping__off__seed42"},
    ],
)
def test_rejects_unmatched_sweep_settings(overrides):
    with pytest.raises(ValueError):
        fixed_configuration(**overrides)


def test_sweep_rejects_missing_duplicate_or_unpaired_conditions():
    from experiments.exp171.analyse import validate_sweep_configurations

    configs = recipe.sweep_configurations()
    validate_sweep_configurations(configs)
    for changed in (
        configs[:-1],
        configs[:-1] + [configs[0]],
        [*configs[:-1], dict(configs[-1], epochs=20)],
        [*configs[:-1], dict(configs[-1], seed=43)],
        [*configs[:-1], dict(configs[-1], ping_cell="ping__off__seed42")],
    ):
        with pytest.raises(ValueError):
            validate_sweep_configurations(changed)


def test_penalised_history_preserves_classification_loss():
    from experiments.exp171.analyse import native_history

    row = dict(
        epoch=0,
        validation_loss=50.0,
        validation_accuracy=0.75,
        validation_cross_entropies={"objective[0]": 0.5},
        observations={"validation": {"population_rates_hz": {"E": 20.0}}},
    )
    measured = native_history({"epochs": [row]}, penalised=True)
    assert measured[0]["cross_entropy"] == 0.5
    del row["validation_cross_entropies"]
    with pytest.raises(ValueError, match="separate cross-entropy"):
        native_history({"epochs": [row]}, penalised=True)


def test_calibration_reuse_preserves_arrays_and_rejects_changes(tmp_path):
    from types import SimpleNamespace

    from experiments.exp171.compute import reuse_calibration

    cfg = fixed_configuration(
        train_images=1, validation_images=1, n_e=2, n_i=2, n_input=4, rate_target_hz=5
    )
    cfg.update(
        source_checkpoints={"ping": {"sha256": "teacher"}},
        dataset_sha256={"pixels": "fixture"},
    )
    ids = np.array([1, 0])
    pixels = np.zeros((2, 4), dtype=np.uint8)
    labels = np.array([1, 2])
    shape = (2, 2)
    alpha, amplitude = np.full(shape, 0.5), np.full(shape, 0.01)
    np.savez(
        tmp_path / "dataset.npz",
        ids=ids,
        features=pixels[ids],
        labels=labels[ids],
        timing_interpolation=alpha,
        amplitude=amplitude,
    )
    counts = np.full(shape, 3, dtype=np.int32)
    np.savez(
        tmp_path / "calibration.npz",
        ids=ids,
        target_mean=np.ones(shape),
        target_sd=np.ones(shape),
        fitted_mean=np.ones(shape),
        fitted_sd=np.ones(shape),
        max_relative_sd_error=np.zeros(2),
        teacher_i_spike_count=counts,
        teacher_i_rate_hz=counts.astype(np.float32) / 0.2,
    )
    restored = []
    source = SimpleNamespace(
        record={"execution": {"configuration": dict(cfg)}},
        file=lambda name: tmp_path / name,
        restore_bundle=lambda *args: restored.append(args),
        check_unchanged=lambda: None,
    )
    arrays = reuse_calibration(source, cfg, ids, pixels, labels, tmp_path / "teacher")
    np.testing.assert_array_equal(arrays[0], alpha)
    np.testing.assert_array_equal(arrays[1], amplitude)
    np.testing.assert_array_equal(arrays[-2], counts)
    assert len(restored) == 1

    for key, value in (
        ("ping_cell", "ping__off__seed42"),
        ("device", "cuda"),
        ("encoding_method", "numpy_v1"),
        ("sensory_seed", 1),
    ):
        with pytest.raises(ValueError, match="Calibration reuse mismatch"):
            reuse_calibration(
                source, dict(cfg, **{key: value}), ids, pixels, labels, tmp_path / "bad"
            )
    with pytest.raises(AssertionError):
        reuse_calibration(source, cfg, ids[::-1], pixels, labels, tmp_path / "bad")
    assert len(restored) == 1


def test_sweep_analysis_pins_every_condition_and_orders_the_grid(monkeypatch, tmp_path):
    from contextlib import contextmanager
    from types import SimpleNamespace

    from experiments.exp171 import analyse as stage

    np.savez(
        tmp_path / "dataset.npz",
        ids=[1, 2],
        features=[[0], [1]],
        labels=[0, 1],
        source_train=[1],
        source_validation=[2],
    )
    sources = {}
    for index, cfg in enumerate(recipe.sweep_configurations()):
        sources[str(index)] = SimpleNamespace(
            record={"execution": {"configuration": cfg}},
            file=lambda *args: tmp_path / "dataset.npz",
            check_unchanged=lambda: None,
        )
    monkeypatch.setattr(
        stage, "source_run", lambda root, identity, **kwargs: sources[identity]
    )
    monkeypatch.setattr(
        stage, "validate", lambda source: dict(timings={}, history=[{"epoch": 10}])
    )
    monkeypatch.setattr(
        stage, "source_controls", lambda source: {"fixture": "controls"}
    )
    recorded = {}

    @contextmanager
    def staged(*args, **kwargs):
        recorded.update(kwargs)
        yield SimpleNamespace(
            export=tmp_path, run_id="analysis", record={"execution": {}}
        )

    monkeypatch.setattr(stage, "stage_run", staged)
    assert stage.analyse(["5", "4", "3", "2", "1", "0"]) == "analysis"
    import json

    measured = json.loads((tmp_path / "results.json").read_text())
    assert [c["target_hz"] for c in measured["sweep"]] == list(recipe.RATE_TARGETS_HZ)
    assert len(recorded["inputs"]) == 6
    assert recorded["inputs"]["compute"] is sources["0"]
    for case in measured["sweep"]:
        assert case["compute_role"] in recorded["inputs"]


def test_frontier_uses_achieved_rates_and_curves_keep_final_training_only():
    import matplotlib.pyplot as plt
    from experiments.exp171.present import frontier_figure, learning_figure

    cases = []
    for index, target in enumerate(recipe.RATE_TARGETS_HZ):
        row = dict(
            epoch=10, cross_entropy=0.4, accuracy_pct=80 + index, e_rate_hz=20 + index
        )
        cases.append(
            dict(
                target_hz=target,
                history=[dict(row, epoch=0), row],
                final_training=dict(row),
                controls={
                    "coba": dict(accuracy_pct=90, e_rate_hz=60 + index),
                    "ping": dict(accuracy_pct=89, e_rate_hz=10 + index),
                },
            )
        )
    results = dict(cases[0], sweep=cases)
    curves = learning_figure(results)
    assert len(curves.axes[0].lines) == 12
    assert all(list(line.get_xdata()) == [10] for line in curves.axes[0].lines[6:])
    frontier = frontier_figure(cases)
    np.testing.assert_array_equal(
        frontier.axes[0].lines[2].get_xdata(), np.arange(20, 26)
    )
    plt.close(curves)
    plt.close(frontier)


def test_delay_and_decay():
    cfg = fixed_configuration()
    events = np.zeros((6, 2))
    events[0] = [1, 2]
    g = filtered(events, cfg)
    np.testing.assert_array_equal(g[0], [0, 0])
    np.testing.assert_array_equal(g[1], [1, 2])
    np.testing.assert_allclose(g[2], np.exp(-0.1 / 6) * np.array([1, 2]))


def test_independent_reproducible_draws_and_fixed_encoding():
    cfg = fixed_configuration()
    cfg.update(n_e=3, n_input=4, steps=100, inhibition_prehistory_steps=60)
    p = np.full((2, 3), 0.1)
    q = np.full((2, 3), 0.02)
    indices = np.array([12, 17])
    first = artificial(p, q, indices, cfg, epoch=1)
    assert first.shape == (cfg["inhibition_prehistory_steps"] + cfg["steps"], 2, 3)
    np.testing.assert_array_equal(first, artificial(p, q, indices, cfg, epoch=1))
    assert digest(first) != digest(artificial(p, q, indices, cfg, epoch=2))
    assert digest(first) != digest(artificial(p, q, indices, cfg, draw=1))
    np.testing.assert_array_equal(
        first[:, :1], artificial(p[:1], q[:1], indices[:1], cfg, epoch=1)
    )
    np.testing.assert_array_equal(
        sensory(np.ones(4), 12, cfg), sensory(np.ones(4), 12, cfg)
    )


def test_calibration_uses_only_conductance():
    cfg = fixed_configuration()
    cfg.update(n_e=3, steps=2000)
    regular, random = timing_endpoints(np.random.default_rng(9), 2000, 3, 24)
    events = timing_events(regular * 0.3 + random * 0.7, 2000)
    target = filtered(events, cfg) * np.array([0.01, 0.02, 0.03])
    p, q, mu, sd, fm, fs, error = calibrate(target, cfg, 17)
    assert np.all(p > 0) and np.all(q > 0)
    np.testing.assert_allclose(mu, fm, rtol=1e-6)
    assert np.max(np.abs(fs / sd - 1)) < 0.03 and error < 0.03


def test_metrics_average_draws_without_ensemble_voting():
    scores = np.array(
        [[[4.0, 0.0], [0.0, 4.0]], [[0.0, 4.0], [0.0, 4.0]]], dtype=np.float32
    )
    measured, loss = summarise(scores, np.array([0, 1]), np.ones((2, 2)) * 5)
    assert measured["accuracy_pct"] == 75
    assert measured["e_rate_hz"] == 5
    assert measured["cross_entropy"] == pytest.approx(loss.mean())


def test_native_authoring_and_trainable_membership():
    cfg = fixed_configuration()
    graph = recipe.author_network(cfg).graph
    assert {p["id"] for p in graph["projections"]} == {
        "input_to_E",
        "readout_projection",
        "E_to_I",
        "I_to_E",
        "artificial_to_E",
    }
    teacher = recipe.author_network(cfg, teacher=True).graph
    assert "i_count" in {row["id"] for row in teacher["outputs"]}
    assert "i_count" not in {row["id"] for row in graph["outputs"]}
    assert set(recipe.TRAINABLE) == {"input_to_E.weight", "readout_projection.weight"}
    assert "artificial" not in {
        p["id"] for p in recipe.author_network(cfg, teacher=True).graph["inputs"]
    }
    with pytest.raises(ValueError):
        fixed_configuration(epochs=0)


def test_ten_epoch_default_and_explicit_epoch_overrides():
    assert fixed_configuration()["epochs"] == 10
    assert fixed_configuration(epochs=3)["epochs"] == 3
    assert fixed_configuration(epochs=20)["epochs"] == 20
    assert fixed_configuration()["train_images"] == 6300
    assert fixed_configuration()["validation_images"] == 700
    assert fixed_configuration()["batch_size"] == 256


def test_cuda_configuration_without_allocating_device():
    from experiments.exp171.compute import performance

    cfg = fixed_configuration(device="cuda")
    assert cfg["calibration_batch_size"] == 64
    assert cfg["precision"] == "auto"
    assert cfg["compile_mode"] == "default"
    assert cfg["microbatch_size"] is None
    assert performance(cfg).cuda_graphs == "auto"
    assert performance(cfg).compile_mode == "default"
    assert fixed_configuration()["compile_mode"] is None
    assert fixed_configuration(device="cuda", compile_mode=None)["compile_mode"] is None
    for invalid in (0, -1, 257, True):
        with pytest.raises(ValueError):
            fixed_configuration(microbatch_size=invalid)
    assert cfg["epochs"] == 10
    assert fixed_configuration()["calibration_batch_size"] == 1


def test_raster_outputs_preserve_network_dynamics():
    cfg = fixed_configuration()
    original = recipe.author_network(cfg).graph
    observed = recipe.author_network(cfg, record_spikes=True).graph
    for key in ("populations", "projections", "parameters", "timebase"):
        assert original[key] == observed[key]
    assert {row["id"] for row in observed["outputs"]} >= {"E_spikes", "I_spikes"}


def test_history_configuration_and_legacy_draws():
    cfg = fixed_configuration()
    assert cfg["inhibition_prehistory_ms"] == 60
    assert cfg["inhibition_prehistory_steps"] == 600
    assert cfg["target_events_per_source"] == 24
    with pytest.raises(ValueError):
        fixed_configuration(inhibition_prehistory_ms=201)
    for invalid in (-1, 0.15, float("nan")):
        with pytest.raises(ValueError):
            fixed_configuration(inhibition_prehistory_ms=invalid)
    # Archived configurations with no history fields retain their original RNG draws.
    cfg.update(n_e=2, steps=10)
    cfg.pop("inhibition_prehistory_steps")
    cfg.pop("inhibition_method")
    ev = artificial(np.full((1, 2), 0.5), np.ones((1, 2)), np.array([17]), cfg, draw=0)
    assert ev.shape == (10, 1, 2)


@pytest.mark.parametrize(
    "method",
    ["count_matched_timing_v2", recipe.TEACHER_COUNT_METHOD, recipe.POISSON_METHOD],
)
@pytest.mark.parametrize("encoding", ["numpy_v1", "torch_device_v1"])
def test_native_coordinated_binding(tmp_path, encoding, method):
    from experiments.exp171.compute import (
        artificial_device,
        encode_native,
        sensory_device,
    )
    from snnlab import extensions, sim

    cfg = fixed_configuration(encoding_method=encoding, inhibition_method=method)
    bundle = recipe.author_network(cfg)
    private = next(row for row in bundle.graph["inputs"] if row["id"] == "artificial")
    assert private["signal_type"] == "continuous"
    weight = next(
        row
        for row in bundle.graph["parameters"]
        if row["id"] == "artificial_to_E.weight"
    )
    assert weight["shape"] == [cfg["n_e"], 1]
    name = "tests.exp171." + encoding + method + "/v1"
    extensions.register_encoder(name, encode_native)
    path = tmp_path / "dataset.npz"
    ids = [17, 42, 66]
    np.savez(
        path,
        ids=ids,
        features=np.full((3, 784), 200, dtype=np.uint8),
        labels=[1, 2, 3],
        event_count=np.array([1, 7, 3], dtype=np.int32),
        poisson_rate_hz=np.full((3, 1024), 200.0, dtype=np.float64),
        timing_interpolation=np.full((3, 1024), 0.5, dtype=np.float32),
        amplitude=np.full((3, 1024), 0.02, dtype=np.float32),
    )
    encoder = sim.DatasetEncoder(
        "custom",
        duration_ms=cfg["duration_ms"],
        definition=name,
        rng_policy="sample_v1",
        config={
            "recipe": cfg,
            "signature": sim.runtime_state_signature(plan_graph(bundle.graph)),
        },
    )
    binding = sim.DatasetSnapshotBinding(
        path,
        "drive",
        "test",
        "validation",
        encoder,
        target_id="digit",
        sample_ids_key="ids",
        sample_indices=(2, 0),
    )
    resolved, (target,) = sim.resolve_dataset_snapshot_binding(bundle.graph, binding)
    torch.testing.assert_close(target.value, torch.tensor([3, 1]))
    np.testing.assert_array_equal(
        resolved.tensors["drive"][:, 0],
        sensory_device(np.full((1, 784), 200), [66], cfg, "cpu")[:, 0]
        if encoding == "torch_device_v1"
        else sensory(np.full(784, 200 / 255), 66, cfg),
    )
    with np.load(path) as dataset:
        events = (artificial_device if encoding == "torch_device_v1" else artificial)(
            dataset[
                "poisson_rate_hz"
                if method == recipe.POISSON_METHOD
                else "timing_interpolation"
            ][[2, 0]],
            dataset["amplitude"][[2, 0]],
            [66, 17],
            cfg,
            **(
                {"device": "cpu", "retain_history": True}
                if encoding == "torch_device_v1"
                else {}
            ),
            draw=0,
            counts=dataset["event_count"][[2, 0]]
            if method == recipe.TEACHER_COUNT_METHOD
            else None,
        )
    if encoding == "torch_device_v1":
        events = events[0].numpy()
    history = cfg["inhibition_prehistory_steps"]
    state = resolved.tensors.initial_state
    np.testing.assert_array_equal(resolved.tensors["artificial"], events[history:])
    np.testing.assert_allclose(
        state.conductances["artificial_to_E"],
        filtered(events[:history], cfg)[-1],
        rtol=1e-6,
    )
    np.testing.assert_array_equal(
        state.input_histories["artificial"], events[history - 1 : history]
    )
    assert not state.currents
    assert sim.Batching(microbatch_size=16).sizes(6300, cfg["batch_size"]) == [
        256
    ] * 24 + [156]


def test_count_matched_draws_preserve_all_24_deliverable_events():
    cfg = fixed_configuration()
    cfg.update(n_e=4)
    # Endpoints and interior timing fits must all preserve the hard count.
    alpha = np.array([[0.0, 0.7, 1.3, 2.0]], dtype=np.float32)
    q = np.full_like(alpha, 0.02)
    for coordinate in (dict(epoch=1), dict(epoch=2), dict(draw=0), dict(draw=1)):
        events = artificial(alpha, q, [17], cfg, **coordinate)
        np.testing.assert_array_equal(
            (events[cfg["inhibition_prehistory_steps"] :] > 0).sum(0),
            np.full((1, 4), 24),
        )
        assert not events[-1].any()
        assert events.shape == (2600, 1, 4)
        np.testing.assert_array_equal(np.unique(events), [0, np.float32(0.02)])


def test_impossible_count_matched_targets_fail_explicitly():
    cfg = fixed_configuration()
    with pytest.raises(ValueError, match="Regular timing endpoint"):
        calibrate(
            np.ones((cfg["steps"], 2)) + np.arange(cfg["steps"])[:, None] * 1e-8,
            cfg,
            17,
        )
    for invalid in (0, 1.5, 2000):
        with pytest.raises(ValueError):
            fixed_configuration(target_events_per_source=invalid)


def test_archived_bernoulli_draws_keep_exact_rng_semantics():
    from experiments.exp171.compute import rng

    cfg = fixed_configuration()
    cfg.pop("inhibition_method")
    cfg.update(n_e=3, steps=10, inhibition_prehistory_steps=6)
    p, q = np.full((1, 3), 0.4), np.full((1, 3), 0.02)
    expected = (
        rng(cfg, "validation", 0, 17).random((16, 3), dtype=np.float32) < p[0]
    ).astype(np.float32) * q[0]
    np.testing.assert_array_equal(artificial(p, q, [17], cfg, draw=0)[:, 0], expected)


def test_high_variability_target_needs_more_than_original_search_budget():
    # Synthetic conductance with CV near the image that exhausted 100 draws.
    target = np.zeros((2000, 1024), dtype=np.float64)
    target[:1020] = 1.0
    short = fixed_configuration(
        inhibition_method="count_matched_timing_v1", bracket_attempts=100
    )
    with pytest.raises(
        ValueError,
        match=r"channels after 100 random draws; unbracketed target CV range",
    ):
        calibrate(target, short, 22078)
    cfg = fixed_configuration()
    alpha, amplitude, mu, sd, fitted_mean, fitted_sd, error = calibrate(
        target, cfg, 22078
    )
    np.testing.assert_allclose(fitted_mean, mu, rtol=1e-6)
    assert error < cfg["calibration_std_tolerance"]
    assert np.max(np.abs(fitted_sd / sd - 1)) < cfg["calibration_std_tolerance"]
    events = artificial(alpha[None], amplitude[None], [22078], cfg, draw=0)
    scored = events[cfg["inhibition_prehistory_steps"] :]
    np.testing.assert_array_equal((scored > 0).sum(0), np.full((1, 1024), 24))


def test_clustered_endpoint_brackets_variability_beyond_uniform_search():
    target = np.zeros((2000, 8), dtype=np.float64)
    target[:838] = np.linspace(0.5, 1.5, 8)
    cfg = fixed_configuration()
    cfg.update(n_e=8)
    alpha, amplitude, mu, sd, fm, fs, error = calibrate(target, cfg, 7363)
    assert np.any(alpha > 1)
    assert error < cfg["calibration_std_tolerance"]
    np.testing.assert_allclose(fm, mu, rtol=1e-6)
    np.testing.assert_allclose(fs, sd, rtol=cfg["calibration_std_tolerance"])
    for coordinate in ({"epoch": 1}, {"epoch": 2}, {"draw": 0}):
        ev = artificial(alpha[None], amplitude[None], [7363], cfg, **coordinate)
        np.testing.assert_array_equal((ev[600:] > 0).sum(0), np.full((1, 8), 24))
        assert not ev[-1].any()


def test_extended_timing_quantization_keeps_adjacent_half_bin_events():
    times = np.array([[10.5], [11.5], [12.5]])
    events = timing_events(times, 100, half_up=True)
    np.testing.assert_array_equal(np.flatnonzero(events[:, 0]), [11, 12, 13])


def test_original_partition_selection_is_deterministic():
    labels = np.tile(np.arange(10), 700)
    train, validation = source_indices(labels)
    assert len(train) == 6300 and len(validation) == 700
    assert not np.intersect1d(train, validation).size
    for actual, expected in zip(
        (train, validation), source_indices(labels), strict=True
    ):
        np.testing.assert_array_equal(actual, expected)


def test_removed_modes_are_rejected_without_execution():
    import subprocess
    import sys

    entrypoint = Path(__file__).with_name("compute.py")
    for flag in ("--benchmark", "--full"):
        result = subprocess.run(
            [sys.executable, str(entrypoint), flag], capture_output=True, text=True
        )
        assert result.returncode == 2
        assert "unrecognized arguments" in result.stderr


def test_native_history_uses_fixed_weight_metrics():
    from experiments.exp171.analyse import native_history

    measured = native_history(
        {
            "epochs": [
                {
                    "epoch": 0,
                    "validation_loss": 2.0,
                    "validation_accuracy": 0.25,
                    "observations": {"validation": {"population_rates_hz": {"E": 3.0}}},
                }
            ]
        }
    )
    assert measured == [
        dict(
            epoch=0,
            training_loss=None,
            cross_entropy=2.0,
            accuracy_pct=25.0,
            e_rate_hz=3.0,
        )
    ]


@pytest.mark.parametrize(
    "method",
    ["count_matched_timing_v2", recipe.TEACHER_COUNT_METHOD, recipe.POISSON_METHOD],
)
@pytest.mark.parametrize("target", [None, 5.0])
def test_compute_delegates_training_to_native_api(
    monkeypatch, tmp_path, target, method
):
    from contextlib import contextmanager
    from types import SimpleNamespace

    from experiments.exp171 import compute as stage
    from snnlab import sim

    cfg = fixed_configuration(
        inhibition_method=method,
        rate_target_hz=target,
        train_images=4,
        validation_images=2,
        epochs=2,
        batch_size=3,
        microbatch_size=2,
        n_input=4,
        n_e=3,
        n_i=2,
        n_output=2,
        duration_ms=10.0,
        inhibition_prehistory_ms=3.0,
    )
    built = sim.build(
        bundle=recipe.author_network(cfg),
        device="cpu",
        performance=sim.Performance(compile_mode=None),
    )
    params = {
        name: torch.zeros_like(value)
        for name, value in built.model.parameter_map().items()
        if name != "artificial_to_E.weight"
    }
    monkeypatch.setattr(recipe, "configuration", lambda **kwargs: dict(cfg))
    monkeypatch.setattr(
        stage,
        "load_source",
        lambda config: (
            SimpleNamespace(check_unchanged=lambda: None),
            {"coba": params, "ping": params},
        ),
    )
    monkeypatch.setattr(
        stage,
        "MNIST",
        lambda **kwargs: SimpleNamespace(
            targets=torch.arange(7000) % 10,
            data=torch.zeros((7000, 4), dtype=torch.uint8),
        ),
    )
    monkeypatch.setattr(stage, "parameter_file", lambda *args: None)
    arrays = [np.full((6, 3), value) for value in (0.7, 0.01, 0.2, 0.1, 0.2, 0.1)] + [
        np.zeros(6),
        np.zeros((6, 2), dtype=np.int32),
        np.zeros((6, 2)),
    ]
    if method == recipe.TEACHER_COUNT_METHOD:
        arrays[-2] = np.full((6, 2), 4, dtype=np.int32)
        arrays[-1] = arrays[-2].astype(np.float32) / (cfg["duration_ms"] / 1000)
    if method == recipe.POISSON_METHOD:
        arrays[0] = np.full((6, 3), 100.0)
        arrays[2], arrays[3] = stage.poisson_expected_moments(arrays[0], arrays[1], cfg)
        arrays[4], arrays[5] = arrays[2].copy(), arrays[3].copy()
    monkeypatch.setattr(stage, "calibrate_dataset", lambda *args: arrays)
    monkeypatch.setattr(
        stage, "record_selected", lambda *args: {"simulation_seconds": 0.0}
    )
    export, scratch = tmp_path / "export", tmp_path / "scratch"
    export.mkdir()
    scratch.mkdir()

    @contextmanager
    def staged(*args, **kwargs):
        yield SimpleNamespace(
            export=export, scratch=scratch, run_id="test", record={"execution": {}}
        )

    monkeypatch.setattr(stage, "stage_run", staged)
    calls = []

    def train(**kwargs):
        calls.append(kwargs)
        weights = {
            name: value.clone() + (0.1 if name in recipe.TRAINABLE else 0)
            for name, value in kwargs["weights"].source.items()
        }
        return SimpleNamespace(
            model=SimpleNamespace(parameter_map=lambda: weights),
            selected_checkpoint=SimpleNamespace(
                parameters=weights, data_state={"epoch": 2}
            ),
            metrics={
                "batch_counts": {"optimizer_updates": 4, "examples": 8},
                "epochs": [],
                "training_timing": {"total_s": 0.0},
                "performance": {},
                "selection_record": {},
            },
        )

    final_calls = []

    def evaluate(*args):
        final_calls.append(args)
        return {"seconds": 0.0}

    monkeypatch.setattr(stage, "evaluate_final_training", evaluate)
    monkeypatch.setattr(sim, "train", train)
    assert stage.compute(epochs=2) == "test"
    if method == recipe.TEACHER_COUNT_METHOD:
        for filename in ("dataset.npz", "calibration.npz"):
            with np.load(export / filename) as saved:
                np.testing.assert_array_equal(saved["event_count"], np.full(6, 4))
    if method == recipe.POISSON_METHOD:
        with np.load(export / "dataset.npz") as saved:
            assert "poisson_rate_hz" in saved and "event_count" not in saved
        with np.load(export / "calibration.npz") as saved:
            assert "expected_mean" in saved and "fitted_mean" not in saved
    assert len(calls) == 1
    assert len(final_calls) == 1
    assert final_calls[0][1] == export / "final.checkpoint"
    assert calls[0]["metrics"].train == "none"
    call = calls[0]
    assert (
        call["bundle"].training["regularizers"]
        == recipe.author_network(cfg).training["regularizers"]
    )
    assert call["epochs"] == 2 and call["batch_size"] == 3
    assert call["batching"].microbatch_size == 2
    assert tuple(call["input_bindings"][0].sample_indices) == (0, 1, 2, 3)
    assert tuple(call["validation"].input_bindings[0].sample_indices) == (4, 5)
    assert call["snapshot"].stage == "both"
    assert call["save_final_checkpoint"].name == "final.checkpoint"


def test_historical_report_evidence_remains_readable():
    from experiments.exp171.analyse import validate
    from pingstore.library import source_run

    root = Path(__file__).resolve().parents[2]
    identity = "exp171-r012-compute"
    if not (root / ".pingstore/runs" / identity).exists():
        pytest.skip("Historical pilot evidence is not local")
    result = validate(
        source_run(root / ".pingstore", identity, experiment="exp171", stage="compute")
    )
    assert result["update_count"] == 200
    assert result["history"][-1]["accuracy_pct"] == pytest.approx(75.0)


def test_native_checkpoint_state_restores_from_pingstore(tmp_path):
    from pingstore.library import source_run
    from snnlab import sim

    root = Path(__file__).resolve().parents[2]
    identity = "exp171-r023-compute"
    if not (root / ".pingstore/runs" / identity).exists():
        pytest.skip("Retained native checkpoint is not local")
    source = source_run(
        root / ".pingstore", identity, experiment="exp171", stage="compute"
    )
    checkpoint = sim.load_training_checkpoint(
        source.restore_bundle("final.checkpoint", tmp_path / "checkpoint")
    )
    assert checkpoint.completed_updates == 4
    assert checkpoint.data_state == {"epoch": 2, "batch": 0}
    assert checkpoint.best_checkpoint is not None
    assert checkpoint.optimizer_state and checkpoint.rng_state.numel()


def test_native_analysis_authenticates_dataset_calibration_identity(tmp_path):
    from types import SimpleNamespace

    from experiments.exp171.analyse import validate_native_dataset

    cfg = fixed_configuration(train_images=4, validation_images=2, n_e=3)
    ids = np.arange(6)
    np.savez(
        tmp_path / "dataset.npz",
        ids=ids,
        source_train=ids[:4],
        source_validation=ids[4:],
        timing_interpolation=np.full((6, 3), 1.3),
        amplitude=np.full((6, 3), 0.02),
    )
    moments = dict(
        teacher_i_spike_count=np.zeros((6, cfg["n_i"]), dtype=np.int32),
        teacher_i_rate_hz=np.zeros((6, cfg["n_i"]), dtype=np.float32),
        target_mean=np.ones((6, 3)),
        fitted_mean=np.ones((6, 3)),
        target_sd=np.full((6, 3), 0.5),
        fitted_sd=np.full((6, 3), 0.5005),
    )
    np.savez(tmp_path / "calibration.npz", ids=ids, **moments)
    computed = SimpleNamespace(file=lambda name: tmp_path / name)
    mean_error, sd_error = validate_native_dataset(computed, cfg)
    assert mean_error == 0 and sd_error == pytest.approx(0.001)
    np.savez(tmp_path / "calibration.npz", ids=ids[::-1], **moments)
    with pytest.raises(AssertionError):
        validate_native_dataset(computed, cfg)


def test_final_training_evaluation_uses_full_split_final_weights(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from experiments.exp171 import compute as stage
    from snnlab import sim

    cfg = fixed_configuration(train_images=3, validation_images=1, batch_size=2)
    path = tmp_path / "dataset.npz"
    np.savez(path, labels=np.array([0, 1, 1, 0]))
    calls = []

    def simulate(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            outputs={
                "scores": torch.tensor([[3.0, 0.0], [0.0, 3.0], [3.0, 0.0]]),
                "e_count": torch.full((3, cfg["n_e"]), 2.0),
            },
            metrics={"performance": {}},
        )

    monkeypatch.setattr(sim, "simulate", simulate)
    checkpoint = tmp_path / "final.checkpoint"
    bundle = recipe.author_network(cfg)
    measured = stage.evaluate_final_training(
        bundle,
        checkpoint,
        path,
        sim.DatasetEncoder(
            "custom", definition="pinglab.exp171/v1", duration_ms=cfg["duration_ms"]
        ),
        cfg,
    )
    assert len(calls) == 1
    assert calls[0]["checkpoint"] == checkpoint
    assert calls[0]["batch_size"] == 2
    assert tuple(calls[0]["input_bindings"][0].sample_indices) == (0, 1, 2)
    assert measured["examples"] == 3 and measured["epoch"] == 10
    assert measured["accuracy_pct"] == pytest.approx(200 / 3)
    assert measured["e_rate_hz"] == 10
    assert measured["cross_entropy"] == pytest.approx(
        float(
            torch.nn.functional.cross_entropy(
                torch.tensor([[3.0, 0.0], [0.0, 3.0], [3.0, 0.0]]),
                torch.tensor([0, 1, 1]),
            )
        )
    )


@pytest.mark.parametrize(
    "method",
    ["count_matched_timing_v2", recipe.TEACHER_COUNT_METHOD, recipe.POISSON_METHOD],
)
def test_teacher_pass_extracts_activity_in_image_order(monkeypatch, tmp_path, method):
    from threading import Event
    from types import SimpleNamespace

    from experiments.exp171 import compute as stage
    from experiments.exp171.analyse import validate_teacher_activity
    from snnlab import sim

    cfg = fixed_configuration(
        inhibition_method=method,
        train_images=2,
        validation_images=1,
        n_input=4,
        n_e=3,
        n_i=2,
        duration_ms=10,
        inhibition_prehistory_ms=0,
        calibration_batch_size=2,
    )
    calls = []
    counts = np.array([[1, 0], [3, 2], [0, 4]], dtype=np.int32)

    fit_started, next_teacher = Event(), Event()

    class Teacher:
        def __call__(self, inputs, **kwargs):
            assert kwargs["batch_offset"] == 2
            assert fit_started.wait(5), "First CPU fit never started"
            next_teacher.set()
            return outputs(inputs["drive"])

    model = Teacher()
    initial_calls = []

    def simulate(**kwargs):
        initial_calls.append(kwargs)
        return outputs(kwargs["input_bindings"][0].value)

    def outputs(drive):
        start = sum(calls)
        batch = drive.shape[1]
        calls.append(batch)
        return SimpleNamespace(
            model=model,
            outputs={
                "g_ping": torch.ones((cfg["steps"], batch, cfg["n_e"])),
                "i_count": torch.from_numpy(
                    counts[start : start + batch].astype(np.float32)
                ),
            },
        )

    monkeypatch.setattr(sim, "simulate", simulate)

    def fit(g, config, index, *, event_count=None):
        if method == recipe.TEACHER_COUNT_METHOD:
            assert event_count == {5: 1, 1: 3, 3: 2}[int(index)]
        if index == 5:
            fit_started.set()
            assert next_teacher.wait(5), "Teacher did not overlap CPU fitting"
        return (np.full(3, index),) * 7

    monkeypatch.setattr(stage, "calibrate", fit)
    extracted = stage.calibrate_dataset(
        None,
        None,
        np.zeros((6, 4), dtype=np.uint8),
        np.array([5, 1, 3]),
        cfg,
        tmp_path / "snapshot",
    )
    assert calls == [2, 1]
    assert len(initial_calls) == 1
    assert initial_calls[0]["snapshot"].stage == "start"
    np.testing.assert_array_equal(extracted[0][:, 0], [5, 1, 3])
    np.testing.assert_array_equal(extracted[-2], counts)
    np.testing.assert_allclose(extracted[-1], counts * 100)
    saved = dict(teacher_i_spike_count=extracted[-2], teacher_i_rate_hz=extracted[-1])
    validate_teacher_activity(saved, cfg)
    saved["teacher_i_rate_hz"] = extracted[-1] + 1
    with pytest.raises(AssertionError):
        validate_teacher_activity(saved, cfg)


@pytest.mark.parametrize("history_ms", [0, 0.1, 4, 10])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_device_encoding_preserves_counts_history_and_sample_streams(
    history_ms, device
):
    from experiments.exp171.compute import artificial_device, sensory_device

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA encoder conformance requires a CUDA device")
    cfg = fixed_configuration(
        n_input=4,
        n_e=4,
        duration_ms=10,
        inhibition_prehistory_ms=history_ms,
    )
    alpha = np.array([[0.0, 0.7, 1.3, 2.0], [0.4, 0.8, 1.2, 1.6]], dtype=np.float32)
    amplitude = np.array([[0.01, 0.02, 0.03, 0.04]] * 2, dtype=np.float32)
    ids = [17, 42]
    pixels = np.full((2, 4), 200, dtype=np.uint8)
    state_before = torch.random.get_rng_state().clone()
    extended, g, pending = artificial_device(
        alpha,
        amplitude,
        ids,
        cfg,
        device,
        draw=0,
        retain_history=True,
    )
    scored, compact_g, compact_pending = artificial_device(
        alpha,
        amplitude,
        ids,
        cfg,
        device,
        draw=0,
    )
    h = cfg["inhibition_prehistory_steps"]
    torch.testing.assert_close(scored, extended[h:], rtol=0, atol=0)
    torch.testing.assert_close(g, compact_g, rtol=0, atol=0)
    torch.testing.assert_close(pending, compact_pending, rtol=0, atol=0)
    torch.testing.assert_close(
        (scored > 0).sum(0), torch.full((2, 4), 24, device=device)
    )
    assert not scored[-1].any()
    values = extended.cpu().numpy()
    expected_g = filtered(values[:h], cfg)[-1] if h else np.zeros((2, 4))
    np.testing.assert_allclose(g.cpu(), expected_g, rtol=1e-6, atol=1e-8)
    np.testing.assert_array_equal(
        pending.cpu(), values[h - 1 : h] if h else np.zeros((1, 2, 4))
    )
    reordered = artificial_device(
        alpha[::-1].copy(), amplitude, ids[::-1], cfg, device, draw=0
    )[0]
    torch.testing.assert_close(scored[:, [1, 0]], reordered, rtol=0, atol=0)
    separate = artificial_device(
        alpha[:1], amplitude[:1], ids[:1], cfg, device, draw=0
    )[0]
    torch.testing.assert_close(scored[:, :1], separate, rtol=0, atol=0)
    changed = artificial_device(alpha, amplitude, ids, cfg, device, draw=1)[0]
    assert not torch.equal(scored, changed)
    drive = sensory_device(pixels, ids, cfg, device)
    torch.testing.assert_close(
        drive[:, :1],
        sensory_device(pixels[:1], ids[:1], cfg, device),
        rtol=0,
        atol=0,
    )
    assert drive.device.type == device and g.device.type == device
    assert not drive.requires_grad and not g.requires_grad
    torch.testing.assert_close(
        state_before, torch.random.get_rng_state(), rtol=0, atol=0
    )


@pytest.mark.parametrize(
    "method",
    ["count_matched_timing_v2", recipe.TEACHER_COUNT_METHOD],
)
def test_device_replay_evidence_supports_cpu_analysis(method):
    from experiments.exp171.analyse import native_replay_events
    from experiments.exp171.compute import artificial_device

    cfg = fixed_configuration(
        n_e=4, duration_ms=10, inhibition_prehistory_ms=4, inhibition_method=method
    )
    ids = np.array([17, 42])
    alpha = np.full((2, 4), 1.3)
    amplitude = np.full((2, 4), 0.02, dtype=np.float32)
    counts = (
        np.array([1, 5], dtype=np.int32)
        if method == recipe.TEACHER_COUNT_METHOD
        else None
    )
    events = artificial_device(
        alpha, amplitude, ids, cfg, "cpu", draw=0, retain_history=True, counts=counts
    )[0].numpy()
    data = dict(
        ids=ids, amplitude=amplitude, timing_interpolation=alpha, event_count=counts
    )
    packed = dict(batch_artificial_packed=np.packbits(events > 0, axis=0))
    np.testing.assert_array_equal(native_replay_events(data, packed, cfg, 0, 2), events)
    flags = events > 0
    flags[cfg["inhibition_prehistory_steps"] :, 0, 0] = False
    packed["batch_artificial_packed"] = np.packbits(flags, axis=0)
    with pytest.raises(ValueError, match="event counts"):
        native_replay_events(data, packed, cfg, 0, 2)
    legacy = {**cfg, "encoding_method": "numpy_v1"}
    np.testing.assert_array_equal(
        native_replay_events(data, {}, legacy, 0, 2),
        artificial(alpha, amplitude, ids, legacy, draw=0, counts=counts),
    )


def test_encoder_registration_is_idempotent_and_rejects_collisions(monkeypatch):
    from types import SimpleNamespace

    from experiments.exp171.compute import dataset_encoder
    from snnlab import extensions

    cfg = fixed_configuration()
    bundle = recipe.author_network(cfg)
    first = dataset_encoder(bundle, cfg)
    second = dataset_encoder(bundle, cfg)
    assert first == second
    monkeypatch.setattr(
        extensions,
        "definitions",
        lambda category: (
            SimpleNamespace(name=first.definition, function=lambda: None),
        ),
    )
    with pytest.raises(ValueError, match="another function"):
        dataset_encoder(bundle, cfg)


@pytest.mark.parametrize("method", ["count_matched_timing_v2", recipe.POISSON_METHOD])
@pytest.mark.parametrize("position", [0, 1])
def test_selected_replay_uses_native_presentation_state(
    monkeypatch, tmp_path, position, method
):
    from types import SimpleNamespace

    from experiments.exp171 import compute as stage
    from snnlab import sim

    cfg = fixed_configuration(
        inhibition_method=method,
        train_images=2,
        validation_images=2,
        replay_batch_size=2,
        n_input=4,
        n_e=4,
        n_i=2,
        n_output=2,
        duration_ms=10,
        inhibition_prehistory_ms=4,
        final_training_draw=9,
    )
    ids = np.array([12, 17, 42, 66])
    alpha = np.full((4, 4), 2000.0 if method == recipe.POISSON_METHOD else 1.3)
    amplitude = np.full((4, 4), 0.02, dtype=np.float32)
    path = tmp_path / "dataset.npz"
    np.savez(
        path,
        ids=ids,
        features=np.full((4, 4), 200, dtype=np.uint8),
        labels=np.array([0, 1, 0, 1]),
        **{stage.generator_key(cfg): alpha},
        amplitude=amplitude,
    )
    events = stage.artificial_device(
        alpha[2:],
        amplitude[2:],
        ids[2:],
        cfg,
        "cpu",
        draw=0,
        retain_history=True,
    )[0].numpy()
    h = cfg["inhibition_prehistory_steps"]
    expected_g = filtered(events, cfg)[h:]
    calls = []

    def simulate(**kwargs):
        calls.append(kwargs)
        assert "runtime_state" not in kwargs
        assert kwargs["batch_size"] == 2
        assert kwargs["snapshot"].stage == "end"
        assert kwargs["performance"].precision == cfg["precision"]
        binding = kwargs["input_bindings"][0]
        assert tuple(binding.sample_indices) == (2, 3)
        resolved, _ = sim.resolve_dataset_snapshot_binding(
            kwargs["bundle"].graph, binding
        )
        state = resolved.tensors.initial_state
        assert isinstance(state, sim.PresentationState)
        np.testing.assert_array_equal(resolved.tensors["artificial"], events[h:])
        np.testing.assert_allclose(
            np.exp(-cfg["dt_ms"] / cfg["tau_gaba_ms"])
            * state.conductances["artificial_to_E"]
            + state.input_histories["artificial"][0],
            expected_g[0],
            rtol=1e-6,
        )
        return SimpleNamespace(
            outputs={
                "g_artificial": torch.from_numpy(expected_g),
                "E_spikes": torch.zeros((cfg["steps"], 2, 4)),
                "I_spikes": torch.zeros((cfg["steps"], 2, 2)),
                "scores": torch.zeros((2, 2)),
                "e_count": torch.zeros((2, 4)),
            }
        )

    monkeypatch.setattr(sim, "simulate", simulate)
    monkeypatch.setattr(stage, "parameter_file", lambda *args: None)
    run = SimpleNamespace(export=tmp_path, scratch=tmp_path)
    summary = stage.record_selected(run, {}, path, cfg, 3, position=position)
    assert len(calls) == 1
    assert summary["epoch"] == 3 and summary["validation_draw"] == 0
    assert summary["precision_policy"] == cfg["precision"]
    assert summary["original_index"] == ids[2 + position]
    assert summary["batch_position"] == position
    assert cfg["final_training_draw"] == 9
    with np.load(tmp_path / "raster.npz") as saved:
        np.testing.assert_array_equal(saved["batch_indices"], ids[2:])
        if method == recipe.POISSON_METHOD:
            from experiments.exp171.analyse import native_replay_events

            with np.load(path) as data:
                np.testing.assert_array_equal(
                    native_replay_events(data, saved, cfg, 2, 4), events
                )
            assert saved["batch_artificial_counts"].max() > 1
            assert "batch_artificial_packed" not in saved
        else:
            np.testing.assert_array_equal(
                saved["prehistory_events"], events[:h, position] > 0
            )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_sensory_fast_path_matches_native_without_tensor_truth_checks(
    monkeypatch, device
):
    from experiments.exp171.compute import device_generator, sensory_device
    from snnlab import sim

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA encoder parity requires a CUDA device")
    cfg = fixed_configuration(n_input=256, duration_ms=10, inhibition_prehistory_ms=0)
    pixels = np.tile(np.arange(256, dtype=np.uint8), (2, 1))
    ids = [17, 42]
    expected = torch.cat(
        [
            sim.encode_images_poisson(
                torch.as_tensor(
                    pixels[row : row + 1], device=device, dtype=torch.float32
                )
                / 255,
                cfg["steps"],
                cfg["dt_ms"],
                cfg["input_rate_hz"],
                generator=device_generator(cfg, "sensory", index, device=device),
            )
            for row, index in enumerate(ids)
        ],
        dim=1,
    )

    def reject_truth_check(self):
        raise AssertionError("Encoding must not branch on a tensor")

    with monkeypatch.context() as guard:
        guard.setattr(torch.Tensor, "__bool__", reject_truth_check)
        actual = sensory_device(pixels, ids, cfg, device)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for invalid in (-1.0, float("nan"), float("inf"), 20000.0):
        with pytest.raises(ValueError, match="sensory rate"):
            sensory_device(pixels, ids, {**cfg, "input_rate_hz": invalid}, device)


def test_calibration_certifies_persisted_float32_values():
    from experiments.exp171.compute import clustered_endpoint, rng, timing_trial

    cfg = fixed_configuration(n_e=3)
    regular, random = timing_endpoints(np.random.default_rng(9), cfg["steps"], 3, 24)
    target = filtered(timing_events(regular * 0.3 + random * 0.7, cfg["steps"]), cfg)
    target *= np.array([0.01, 0.02, 0.03])
    alpha, q, mu, sd, fitted_mean, fitted_sd, error = calibrate(target, cfg, 17)
    assert alpha.dtype == q.dtype == np.float32
    gen = rng(cfg, "calibration", 17)
    regular, random = timing_endpoints(gen, cfg["steps"], 3, 24)
    past = timing_endpoints(gen, cfg["steps"], 3, 24)
    clusters = tuple(clustered_endpoint(gen, cfg["steps"], 3, 24) for _ in range(2))
    events = timing_trial(regular, random, alpha, cfg, past, clusters) * q
    measured = filtered(events, cfg)[cfg["inhibition_prehistory_steps"] :]
    np.testing.assert_allclose(measured.mean(0), fitted_mean, rtol=1e-12)
    np.testing.assert_allclose(measured.std(0), fitted_sd, rtol=1e-12)
    assert error == pytest.approx(
        float(np.max(np.abs(measured.std(0) / sd - 1))), abs=1e-12
    )
    with pytest.raises(ValueError, match="Saved-parameter calibration failed"):
        calibrate(target * 1e-50, cfg, 17)


@pytest.mark.parametrize("method", ["count_matched_timing_v2", recipe.POISSON_METHOD])
@pytest.mark.parametrize("target", [None, 5.0])
def test_native_validation_only_rate_policy_passes_preflight_without_simulation(
    monkeypatch, tmp_path, target, method
):
    from experiments.exp171.compute import dataset_binding, dataset_encoder
    from snnlab import sim

    cfg = fixed_configuration(
        inhibition_method=method,
        rate_target_hz=target,
        n_input=4,
        n_e=4,
        n_i=2,
        n_output=2,
        duration_ms=10,
        inhibition_prehistory_ms=4,
        train_images=1,
        validation_images=1,
    )
    path = tmp_path / "dataset.npz"
    np.savez(
        path,
        ids=[17, 42],
        features=np.full((2, 4), 200, dtype=np.uint8),
        labels=[0, 1],
        poisson_rate_hz=np.full((2, 4), 200.0),
        timing_interpolation=np.full((2, 4), 1.3),
        amplitude=np.full((2, 4), 0.02, dtype=np.float32),
    )
    bundle = recipe.author_network(cfg)
    encoder = dataset_encoder(bundle, cfg)

    class ReachedForward(Exception):
        pass

    def stop_before_simulation(model, inputs, **kwargs):
        assert isinstance(kwargs["initial_state"], sim.PresentationState)
        raise ReachedForward

    monkeypatch.setattr(sim.GraphExecutor, "_forward", stop_before_simulation)
    with pytest.raises(ReachedForward):
        sim.train(
            bundle=bundle,
            input_bindings=[dataset_binding(path, encoder, "train", [0])],
            validation=sim.ValidationSpec(
                input_bindings=[dataset_binding(path, encoder, "validation", [1])],
                encoding_seeds=(0, 1, 2),
            ),
            epochs=1,
            batch_size=1,
            device="cpu",
            diagnostics=False,
            metrics=sim.TrainingMetrics(train="none"),
            observations=sim.EpochObservations(population_rates=("E",)),
            performance=sim.Performance(compile_mode=None),
        )


def test_correct_raster_selection_uses_first_correct_and_rejects_empty_selection():
    from experiments.exp171.compute import first_correct_position

    labels = np.array([1, 0, 1])
    scores = np.array([[2, 0], [2, 0], [0, 2]], dtype=float)
    assert first_correct_position(labels, scores) == 1
    with pytest.raises(ValueError, match="No correctly classified"):
        first_correct_position(1 - scores.argmax(axis=1), scores)


@pytest.mark.parametrize("target", recipe.RATE_TARGETS_HZ)
def test_compact_source_export_preserves_weights_and_detects_corruption(
    tmp_path, monkeypatch, target
):
    import json
    import subprocess
    from types import SimpleNamespace

    from experiments.exp022 import analyse as source_analysis
    from experiments.exp171 import analyse as analysis
    from experiments.exp171 import compute as training
    from pingstore import core
    from pingstore.library import file_sha256, source_run, stage_run

    monkeypatch.setattr(core, "memberships", lambda _: {"exp022": "fixture"})
    monkeypatch.setattr(source_analysis, "REPO", tmp_path)
    monkeypatch.setattr(training, "ROOT", tmp_path)
    monkeypatch.setattr(analysis, "ROOT", tmp_path)
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.com",
            "commit",
            "--allow-empty",
            "-qm",
            "fixture",
        ],
        cwd=tmp_path,
        check=True,
    )
    cfg = recipe.configuration(rate_target_hz=target)
    conf = dict(
        dt=0.1,
        t_ms=200.0,
        max_samples=7000,
        input_rate=25.0,
        fr_reg_upper_strength=cfg["rate_penalty"],
        fr_reg_upper_target_hz=target or 0.0,
        readout_mode="mem-mean",
        seed=42,
        n_in=784,
        n_hidden=1024,
        n_inh=256,
        n_out=10,
        tau_gaba_ms=6.0,
        tau_ampa_ms=2.0,
        v_grad_dampen=1000.0,
        epochs=50,
        dataset_split={"validation_samples": 700},
        validation_encoder_draws={"count": 3},
    )
    cells = [cfg[kind + "_cell"] for kind in ("coba", "ping")]
    with stage_run(tmp_path, "exp022", "compute") as bank:
        provenance = {}
        for cell in cells:
            directory = bank.export / cell
            directory.mkdir()
            (directory / "config.json").write_text(json.dumps(conf))
            (directory / "metrics.json").write_text(
                json.dumps(
                    dict(
                        config=conf,
                        training_run_id="TR-02",
                        epochs=[
                            dict(ep=50, acc=85.0, test_rate_e=12.0, test_rate_i=30.0)
                        ],
                    )
                )
            )
            weights = {key: torch.zeros(1, 1) for key in recipe.MAPPING.values()}
            weights["W_ff.0"] = torch.ones(1, 1)
            torch.save(weights, directory / "weights_final.pth")
            provenance[cell] = {
                "checkpoint": dict(
                    role="final_epoch",
                    epoch=50,
                    filename="weights_final.pth",
                    sha256=file_sha256(directory / "weights_final.pth"),
                )
            }
        (bank.export / "unused-recording.npz").write_bytes(b"unused")
        bank.record["bank_reuse"] = {"diagnostic_provenance": provenance}
    original = source_run(tmp_path / ".pingstore", bank.run_id)
    cfg.update(
        source_run=bank.run_id, source_digest=original.reference["payload_digest"]
    )
    _, expected = training.load_source(cfg)

    def computed(pin):
        return SimpleNamespace(
            record={
                "inputs": {"bank": pin},
                "execution": {"configuration": cfg},
            }
        )

    expected_controls = analysis.source_controls(computed(original.reference))
    identity = source_analysis.export_finetuning(bank.run_id, cells)
    compact = source_run(tmp_path / ".pingstore", identity)
    assert compact.record["inputs"]["bank"] == original.reference
    assert not (compact.export / "unused-recording.npz").exists()
    cfg.update(source_run=identity, source_digest=compact.reference["payload_digest"])
    _, actual = training.load_source(cfg)
    assert analysis.source_controls(computed(compact.reference)) == expected_controls
    assert analysis.source_bank(cfg, root=tmp_path)[1] == original.reference

    class ReachedNativeArtifacts(Exception):
        pass

    def stop_at_artifacts(*parts):
        assert parts == ("network.bundle", "training.json")
        raise ReachedNativeArtifacts

    evidence = computed(compact.reference)
    evidence.file = stop_at_artifacts
    with pytest.raises(ReachedNativeArtifacts):
        analysis.validate_native(evidence)
    for kind in expected:
        for name in expected[kind]:
            torch.testing.assert_close(actual[kind][name], expected[kind][name])
    (compact.export / cells[0] / "config.json").write_text("{}")
    with pytest.raises(core.PingstoreError, match="checksum"):
        training.load_source(cfg)


def test_compact_sweep_requires_common_original_bank(monkeypatch):
    from experiments.exp171 import analyse as analysis

    configs = recipe.sweep_configurations()
    for index, cfg in enumerate(configs):
        cfg.update(
            source_run=f"exp022-r{100 + index:03d}-analyse",
            source_digest="sha256:" + f"{index:064x}",
        )
    parent = {"run_id": "exp022-r015-compute", "payload_digest": "sha256:" + "a" * 64}
    monkeypatch.setattr(analysis, "source_bank", lambda cfg, **kwargs: (None, parent))
    analysis.validate_sweep_configurations(configs)

    def changed_parent(cfg, **kwargs):
        return None, dict(parent, payload_digest=cfg["source_digest"])

    monkeypatch.setattr(analysis, "source_bank", changed_parent)
    with pytest.raises(ValueError, match="same original bank"):
        analysis.validate_sweep_configurations(configs)
