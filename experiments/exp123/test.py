"""Scientific pairing and first-crossing checks for the shared sweep.

Run with pytest from the repository root. Small synthetic inputs check the
scientific invariants and native runtime integration without replacing the
report's pinned historical evidence.
"""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from experiments.exp123 import recipe
from experiments.exp123.custom import accuracy


def test_shared_sample_is_uniform_and_anchor_decays_are_exact():
    labels = np.repeat(np.arange(10), 60)
    cfg = recipe.shared_configuration(labels)
    sweep = cfg["sweep"]
    assert len(sweep["image_indices"]) == 500
    assert np.array_equal(np.bincount(labels[list(sweep["image_indices"])]), [50] * 10)
    assert len(sweep["seeds"]) == 10
    assert len(sweep["tau_gaba_ms"]) == 19
    assert np.all(np.diff(sweep["tau_gaba_ms"]) > 0)
    assert all(tau in sweep["tau_gaba_ms"] for tau in (3, 6, 12, 30))
    assert set(cfg["illustrative_image_indices"]) <= set(sweep["image_indices"])
    assert sweep["input_rates_hz"] == (25.0,)
    assert cfg["presentations"] == 95000


def test_reused_measurements_require_the_exact_compute_payload():
    from experiments.exp123.custom.shared_analysis import analyse_shared

    source = SimpleNamespace(
        reference={"run_id": "compute", "digest": "new"},
        record={"execution": {"configuration": {"sweep": {}}}},
    )
    previous = SimpleNamespace(
        record={
            "execution": {"configuration": {"schema": "exp123.shared.measurement/v1"}},
            "inputs": {"compute": {"run_id": "compute", "digest": "old"}},
        }
    )
    with pytest.raises(ValueError, match="exact compute payload"):
        analyse_shared(source, reuse=previous)


def test_network_budget_samples_first_crossing_and_preserves_latency(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        accuracy,
        "volley_times",
        lambda *args: (np.array([10.0, 20.0, 30.0]), True, 100.0),
    )
    cfg = recipe.configuration("benchmark", (0, 1))
    cfg.update(seeds=(42, 43), shared_sweep=True)
    root = tmp_path / "source"
    root.mkdir()
    for image in cfg["image_indices"]:
        unit = root / recipe.recording_unit(image, 25, "benchmark")
        unit.mkdir()
        np.savez_compressed(
            unit / "input.npz", label=1, spikes=np.ones((2000, 2, 1), dtype=bool)
        )
        for tau, width in zip(cfg["tau_gaba_ms"], (10, 20, 40)):
            output = np.zeros((2000, 2, 10), dtype=bool)
            output[9, :, 1] = True
            np.savez_compressed(
                unit / f"tau{tau:g}--spikes.npz",
                spk_e=np.ones((2000, 2, width), dtype=bool),
                spk_i=np.zeros((2000, 2, 1), dtype=bool),
                spk_out=output,
            )
    destination = tmp_path / "export"
    destination.mkdir()
    report = accuracy.measure_accuracy(
        SimpleNamespace(unit=lambda name: root / name),
        SimpleNamespace(export=destination),
        cfg,
    )
    budget100 = report["curves"]["network"]["grid"].index(100)
    assert [curve[budget100] for curve in report["curves"]["network"]["accuracy"]] == [
        1.0,
        0.0,
        0.0,
    ]
    assert np.allclose(
        [row[budget100] for row in report["mean_network_budget_latency_ms"]],
        [1.0, 0.5, 0.3],
    )
    assert [row[1] for row in report["mean_network_spikes_by_time"]] == [
        101.0,
        201.0,
        401.0,
    ]
    assert report["curves"]["network"]["trial_coverage"][201] == 0
    assert all(row[201] is None for row in report["curves"]["network"]["accuracy"])
    assert (
        json.loads((destination / "accuracy.json").read_text())["presentations"] == 12
    )


def test_one_source_supplies_all_analysis_and_six_figures(tmp_path, monkeypatch):
    from contextlib import contextmanager

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from experiments.exp123.custom import shared_analysis, shared_plots
    from experiments import theme

    theme.set_paper_mode(True)
    theme.apply()

    cfg = recipe.configuration("benchmark", tuple(range(9)))
    cfg.update(seeds=(42, 43), shared_sweep=True, tau_gaba_ms=recipe.SWEEP_TAUS_MS)
    root = tmp_path / "compute"
    root.mkdir()
    for image in cfg["image_indices"]:
        unit = root / recipe.recording_unit(image, 25, "benchmark")
        unit.mkdir()
        np.savez_compressed(
            unit / "input.npz", label=image, spikes=np.ones((2000, 2, 1), dtype=bool)
        )
        for ti, tau in enumerate(cfg["tau_gaba_ms"]):
            e = np.zeros((2000, 2, 1024), dtype=bool)
            i = np.zeros((2000, 2, 256), dtype=bool)
            out = np.zeros((2000, 2, 10), dtype=bool)
            e[10::60, :, :256] = True
            i[30::60] = True
            out[15::60, :, image] = True
            if image == 0 and ti == 2:
                i[:, 1] = False
            np.savez_compressed(
                unit / f"tau{tau:g}--spikes.npz", spk_e=e, spk_i=i, spk_out=out
            )
    definition = dict(
        schema="exp123.compute/v3", sweep=cfg, illustrative_image_indices=list(range(9))
    )
    source = SimpleNamespace(
        unit=lambda name: root / name,
        record={"execution": {"configuration": definition}},
    )
    export = tmp_path / "analysis"
    export.mkdir()

    @contextmanager
    def synthetic_stage(*args, **kwargs):
        assert kwargs["inputs"] == {"compute": source}
        yield SimpleNamespace(export=export, run_id="synthetic-analysis")

    monkeypatch.setattr(shared_analysis, "stage_run", synthetic_stage)
    assert shared_analysis.analyse_shared(source, workers=2) == "synthetic-analysis"
    pooled = json.loads((export / "pooled_decisions.json").read_text())
    assert pooled["presentations"] == 342
    assert pooled["missing_frequency"] == 1
    assert pooled["stable_with_frequency"] == 341
    assert pooled["mapped_with_frequency"] == 340
    assert np.sum(pooled["time_counts"]) == 341
    assert np.sum(pooled["volley_counts"]) == 340
    with np.load(export / "image00000/trajectories.npz") as data:
        assert data["completed_volley_coverage"][0, 0] == 1
        assert data["completed_volley_coverage"][2, 0] == 0.5
        assert np.isfinite(data["true_score_by_completed_volley"][0]).all()
    analysis_source = SimpleNamespace(
        file=lambda name: export / name, unit=lambda name: export / name
    )
    figures = tmp_path / "figures"
    figures.mkdir()

    def save(fig, destination):
        fig.savefig(destination.with_suffix(".png"), dpi=100)
        plt.close(fig)

    shared_plots.present_shared(analysis_source, SimpleNamespace(export=figures), save)
    assert len(list(figures.glob("*.png"))) == 6
    colours = json.loads((figures / "frequency_colours.json").read_text())
    assert set(colours["mean_frequency_hz_by_image"]) == {str(i) for i in range(9)}
    assert np.allclose(colours["mean_frequency_hz_by_image"]["1"], 1000 / 6)
    assert all(p.stat().st_size > 1000 for p in figures.glob("*.png"))
    # Moving raster events into analysis must preserve the rendered overview.
    from experiments.exp123.custom.overview_plot import plot_inhibitory_overview

    dynamics_report = json.loads((export / "metrics.json").read_text())
    plot_inhibitory_overview(
        dynamics_report, source, tmp_path / "direct_overview", save
    )
    assert (tmp_path / "direct_overview.png").read_bytes() == (
        figures / "inhibitory_overview.png"
    ).read_bytes()
    tradeoffs = json.loads((export / "tradeoffs.json").read_text())
    assert len(tradeoffs["rows"]) == 19 * 201
    # Parallel scheduling must preserve the scientific result of serial evaluation.
    settings = recipe.dynamics_configuration(cfg)
    units = {
        index: str(source.unit(recipe.recording_unit(index, 25, "benchmark")))
        for index in cfg["image_indices"]
    }
    metrics = json.loads((export / "metrics.json").read_text())
    condition = metrics["design"]["conditions"][0]
    serial, _ = shared_analysis.condition_dynamics(condition, cfg, settings, units)
    assert serial == metrics["summaries"][0]


def test_raw_image_reuse_preserves_balance_and_rejects_changed_labels(tmp_path):
    import pytest
    from experiments.exp123.compute import image_bank

    cfg = recipe.configuration("benchmark", tuple(range(500)))
    source = SimpleNamespace(
        record={
            "execution": {
                "configuration": {
                    "schema": "exp123.compute/v2",
                    "protocols": {"benchmark": cfg},
                }
            }
        },
        unit=lambda name: tmp_path / name,
    )
    for index in cfg["image_indices"]:
        unit = source.unit(recipe.recording_unit(index, 25, "benchmark"))
        unit.mkdir()
        np.savez(
            unit / "input.npz",
            pixels=np.full((28, 28), index % 256 / 255, dtype=np.float32),
            label=index % 10,
        )
    images = image_bank(source)
    assert len(images) == 500
    assert np.isclose(images[255][0][0, 0], 1.0)
    unit = source.unit(recipe.recording_unit(1, 25, "benchmark"))
    np.savez(unit / "input.npz", pixels=np.zeros((28, 28), dtype=np.float32), label=0)
    with pytest.raises(ValueError, match="50 distinct"):
        image_bank(source)


def test_installed_runtime_builds_the_frozen_network():
    from snnlab.sim import build

    bundle = recipe.author_network(3.0)
    model = build(
        graph=bundle.graph,
        device="cpu",
        seed=0,
    ).model
    assert set(model.parameter_map()) == set(recipe.PARAMETERS)


def test_two_volleys_without_complete_interior_cycle_are_undefined():
    from experiments.exp123.custom.overview_analysis import dynamics

    settings = recipe.dynamics_configuration()
    e = np.zeros((2000, 1024), dtype=bool)
    i = np.zeros((2000, 256), dtype=bool)
    e[10::60, :256] = True
    i[[300, 1000]] = True
    metrics, diagnostic = dynamics(e, i, settings)
    assert not metrics["valid_cycles"]
    assert metrics["accepted_cycles"] == 0
    assert metrics["participation_e_pct"] is None
    assert len(diagnostic["boundaries_steps"]) == 0


def test_target_frontier_uses_first_crossing_and_two_cost_dominance():
    from experiments.exp123.custom.target_accuracy import measure_target_accuracy

    def row(tau, time, cost, accuracy):
        return dict(
            tau_gaba_ms=tau,
            deadline_ms=time,
            mean_network_spikes=cost,
            accuracy=accuracy,
        )

    tradeoffs = dict(
        design=dict(tau_gaba_ms=[1, 2, 3, 4]),
        rows=[
            row(1, 1, 5, 0.79),
            row(1, 2, 10, 0.80),
            row(1, 3, 20, 0.90),
            row(2, 3, 9, 0.81),
            row(3, 4, 11, 0.82),
            row(4, 5, 3, 0.70),
        ],
    )
    trials = dict(
        tau_gaba_ms=np.array([1, 1, 2, 3, 4]),
        frequency_hz=np.array([100, np.nan, 80, 60, 40]),
    )
    report = measure_target_accuracy(tradeoffs, trials, 80)
    assert [p["tau_gaba_ms"] for p in report["frontier"]] == [2, 1]
    assert report["points"][0]["deadline_ms"] == 2
    assert report["points"][0]["mean_network_spikes"] == 10
    assert report["points"][0]["accepted_frequency_trials"] == 1
    assert report["points"][0]["total_trials"] == 2
    assert not report["points"][2]["on_frontier"]
    assert report["unreachable_tau_gaba_ms"] == [4]
