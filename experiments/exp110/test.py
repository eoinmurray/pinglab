from pathlib import Path

import pytest
from experiments.exp033 import recipe as mean_field_recipe
from experiments.exp054 import plots as exp054_plots
from experiments.exp054 import recipe as exp054_recipe
from experiments.exp110 import plots, present, recipe
from PIL import Image
from pingstore import stages
from pingstore.contracts import write_json_atomic
from pingstore.stages import source_run


def test_figure_ownership_has_moved_from_exp054() -> None:
    assert not hasattr(exp054_plots, "build_super_compound")
    assert "onset_super_compound.png" not in exp054_recipe.FIGURES
    assert recipe.FIGURES == (
        "onset_super_compound.png",
        "onset_super_compound.pdf",
        "cycle_participation_compound.png",
        "cycle_participation_compound.pdf",
        "robustness_compound.png",
        "robustness_compound.pdf",
    )


@pytest.mark.parametrize("refreshed_theory", [False, True])
def test_present_records_exp054_analysis_and_exports_only_the_bundle(
    tmp_path: Path, monkeypatch, refreshed_theory
) -> None:
    monkeypatch.setattr(
        stages,
        "memberships",
        lambda _: {
            "exp025": "test",
            "exp038": "test",
            "exp041": "test",
            "exp046": "test",
            "exp037": "test",
            "exp044": "test",
            "exp054": "test",
            "exp110": "test",
        },
    )
    monkeypatch.setattr(
        stages, "_capture_code", lambda *args: {"git_commit": "fixture", "dirty": False}
    )
    with stages.stage_run(tmp_path, "exp054", "analyse") as run:
        (run.export / "fixture.json").write_text("{}")
    analysis = source_run(
        tmp_path / ".pingstore", run.run_id, stage="analyse", experiment="exp054"
    )
    presentations = {}
    source_analyses = {}
    for experiment, name in (
        ("exp041", recipe.RATE_FREQUENCY_SOURCE),
        ("exp046", recipe.CYCLE_COUNT_SOURCE),
        ("exp037", recipe.PERTURBATION_SOURCE),
        ("exp044", recipe.TIMESTEP_SOURCE),
    ):
        sources = {}
        if experiment in ("exp041", "exp046"):
            with stages.stage_run(tmp_path, experiment, "analyse") as saved:
                (saved.export / "results.json").write_text("{}")
            source_analyses[experiment] = source_run(tmp_path / ".pingstore", saved.run_id)
            sources["analysis"] = source_analyses[experiment]
        with stages.stage_run(tmp_path, experiment, "present", inputs=sources) as source:
            path = source.export / name
            if path.suffix == ".svg":
                path.write_text('<svg xmlns="http://www.w3.org/2000/svg"/>')
            elif path.suffix == ".json":
                path.write_text("{}")
            else:
                Image.new("RGB", (200, 100), "white").save(path)
        presentations[experiment] = source_run(
            tmp_path / ".pingstore",
            source.run_id,
            stage="present",
            experiment=experiment,
        )
    source_recipe = exp054_recipe.configuration(smoke=True)
    if refreshed_theory:
        source_recipe = exp054_recipe.refresh_configuration(
            source_recipe, mean_field_recipe.configuration(version=2)
        )
    coordinates = {
        "grid": [],
        "mean_field": {
            "sweep": [],
            "hopf": {},
            "criticality": {},
            "frequency_vs_tau_gaba": [],
            "spiking_exp041": {},
        },
    }
    monkeypatch.setattr(present, "REPO", tmp_path)
    monkeypatch.setattr(
        present,
        "analysis_source",
        lambda repo, identity: (analysis, source_recipe, coordinates, {}),
    )

    def render(*args) -> None:
        destination = args[-1]
        destination.with_suffix(".png").write_bytes(b"png")
        destination.with_suffix(".pdf").write_bytes(b"pdf")

    monkeypatch.setattr(plots, "build_onset_super_compound", render)
    monkeypatch.setattr(
        present,
        "build_cycle_participation_compound",
        render,
    )
    monkeypatch.setattr(
        present,
        "build_robustness_compound",
        render,
    )
    identity = present.present(
        analysis.record["run_id"],
        presentations["exp041"].record["run_id"],
        presentations["exp046"].record["run_id"],
        presentations["exp037"].record["run_id"],
        presentations["exp044"].record["run_id"],
    )
    output = source_run(
        tmp_path / ".pingstore", identity, stage="present", experiment="exp110"
    )
    assert output.record["inputs"] == {
        "exp054_analysis": analysis.reference,
        "exp041_presentation": presentations["exp041"].reference,
        "exp046_presentation": presentations["exp046"].reference,
        "exp041_analysis": source_analyses["exp041"].reference,
        "exp046_analysis": source_analyses["exp046"].reference,
        "exp037_presentation": presentations["exp037"].reference,
        "exp044_presentation": presentations["exp044"].reference,
    }
    assert sorted(path.name for path in output.export.iterdir()) == sorted(
        recipe.FIGURES
    )
    assert output.record["execution"]["configuration"]["source_recipe"] == source_recipe

def test_cycle_participation_equal_width_and_no_percentages(tmp_path, monkeypatch):
    import pytest

    rate = tmp_path / "rate.json"
    cycles = tmp_path / "cycles.json"
    taus = (4.5, 6, 9, 12, 18, 27)
    write_json_atomic(rate, {
        "schema": "exp041.analysis/v1",
        "aggregate": [
            {"tau_gaba_ms": tau, **{
                key: {"mean": value, "sem": 0.5}
                for key, value in (("f_gamma_hz", 100 / tau), ("e_rate_hz", 20 / tau), ("acc", 90))
            }} for tau in taus
        ],
        "fit": {"p_affine": 0.2, "a_affine": 0, "r2_affine": 1},
    })
    write_json_atomic(cycles, {
        "schema": "exp046.analysis/v1",
        "per_tau": {f"tau_{tau:g}": {
            "frac_zero": 0.7, "frac_one": 0.28, "frac_two": 0.015, "frac_three_plus": 0.005,
        } for tau in taus},
    })
    save = present.save_figure
    def inspect(fig, stem, **kwargs):
        a, b, *bottom = fig.axes
        assert a.get_position().width == pytest.approx(b.get_position().width)
        assert a.get_position().y0 == pytest.approx(b.get_position().y0)
        assert a.get_xlim() == b.get_xlim()
        assert [text.get_text() for ax in bottom for text in ax.texts] == list("CDEFGH")
        for ax in bottom:
            assert [bar.get_height() for bar in ax.patches] == pytest.approx([0.7, 0.28, 0.015, 0.005])
        save(fig, stem, **kwargs)
    monkeypatch.setattr(present, "save_figure", inspect)
    present.build_cycle_participation_compound(rate, cycles, tmp_path / "combined")
    assert (tmp_path / "combined.png").is_file()
    assert (tmp_path / "combined.pdf").is_file()


@pytest.mark.parametrize("relative", [False, True])
def test_robustness_composite_uses_equal_width_row_panels(
    tmp_path: Path, monkeypatch, relative,
) -> None:
    perturbation = tmp_path / "perturbation.json"
    timestep = tmp_path / "timestep.json"
    curve = {
        "x": [0.0, 100.0],
        "mean": [90.0, 10.0],
        "lo": [89.0, 9.0],
        "hi": [91.0, 11.0],
    }
    write_json_atomic(
        perturbation,
        {
            "schema": "exp037.analysis/v2" if relative else "exp037.analysis/v1",
            "plot_data": {
                "use_pct": True,
                "relative_test_baseline": relative,
                "panels": {
                    mode: {model: {**curve, "x": [0.0, 200.0] if mode == "add" else curve["x"]} for model in ("coba", "ping")}
                    for mode in ("drop", "add")
                },
            },
        },
    )
    write_json_atomic(
        timestep,
        {
            "schema": "exp044.analysis/v1",
            "aggregate": [
                {
                    "dt_ms": value,
                    "e_rate_hz": {"mean": 15.0, "sem": 0.1},
                    "acc": {"mean": 90.0, "sem": 0.2},
                }
                for value in (0.05, 0.1, 0.2, 0.3, 0.6)
            ],
        },
    )
    save = present.save_figure
    def inspect(fig, stem, **kwargs):
        drop, add, rate, accuracy = fig.axes
        assert list(rate.lines[0].get_xdata()) == [0.05, 0.1, 0.2, 0.3, 0.6]
        assert list(accuracy.lines[0].get_xdata()) == [0.05, 0.1, 0.2, 0.3, 0.6]
        assert list(add.lines[0].get_xdata()) == [0.0, 200.0]
        assert add.get_xlim()[1] > 200
        assert ("baseline E rate" if relative else "reference E rate") in add.get_xlabel()
        assert "Spike insertion" in add.get_title(loc="left")
        assert "±1 SD band" in [text.get_text() for text in fig.legends[0].get_texts()]
        save(fig, stem, **kwargs)
    monkeypatch.setattr(present, "save_figure", inspect)
    present.build_robustness_compound(perturbation, timestep, tmp_path / "combined")
    with Image.open(tmp_path / "combined.png") as combined:
        assert combined.width > 2 * combined.height
    assert (tmp_path / "combined.pdf").is_file()
