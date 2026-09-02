from pathlib import Path

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


def test_present_records_exp054_analysis_and_exports_only_the_bundle(
    tmp_path: Path, monkeypatch
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
    for experiment, name in (
        ("exp041", recipe.RATE_FREQUENCY_SOURCE),
        ("exp046", recipe.CYCLE_COUNT_SOURCE),
        ("exp037", recipe.PERTURBATION_SOURCE),
        ("exp044", recipe.TIMESTEP_SOURCE),
    ):
        with stages.stage_run(tmp_path, experiment, "present") as source:
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
        lambda *args: render(*args[:-1]),
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
        "exp037_presentation": presentations["exp037"].reference,
        "exp044_presentation": presentations["exp044"].reference,
    }
    assert sorted(path.name for path in output.export.iterdir()) == sorted(
        recipe.FIGURES
    )

def test_cycle_participation_composite_stacks_and_relabels(
    tmp_path: Path, monkeypatch
) -> None:
    rate = tmp_path / "rate.svg"
    cycles = tmp_path / "cycles.svg"
    Image.new("RGB", (200, 140), "white").save(rate, format="PNG")
    Image.new("RGB", (200, 60), "white").save(cycles, format="PNG")

    def rasterize(source: Path, destination: Path, *, width: int = 2070) -> None:
        destination.write_bytes(source.read_bytes())

    monkeypatch.setattr(present, "_rasterize_svg", rasterize)
    present.build_cycle_participation_compound(
        rate, cycles, tmp_path / "combined", tmp_path
    )
    with Image.open(tmp_path / "combined.png") as combined:
        assert combined.size == (200, 202)
    assert (tmp_path / "combined.pdf").is_file()


def test_robustness_composite_uses_equal_width_row_panels(
    tmp_path: Path,
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
            "schema": "exp037.analysis/v1",
            "plot_data": {
                "use_pct": True,
                "panels": {
                    mode: {model: curve for model in ("coba", "ping")}
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
                for value in (0.05, 0.1, 0.25, 0.5, 1.0)
            ],
        },
    )
    present.build_robustness_compound(perturbation, timestep, tmp_path / "combined")
    with Image.open(tmp_path / "combined.png") as combined:
        assert combined.width > 2.5 * combined.height
    assert (tmp_path / "combined.pdf").is_file()
