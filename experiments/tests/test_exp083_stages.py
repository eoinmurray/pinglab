"""Synthetic stage-contract probes; no production simulation or archive import."""

import importlib
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from experiments.exp083 import (
    analyse,
    compute,
    evidence,
    inputs,
    measurements,
    present,
    recipe,
)
from pingstore import stages
from pingstore.contracts import (
    PingstoreError,
    load_json,
    payload_digest,
    validate_operational_run_directory,
    write_json_atomic,
)
from pingstore.discovery import discover_runs

REPO = Path(__file__).resolve().parents[2]


def forbid(*args, **kwargs):
    pytest.fail("stage crossed an execution boundary")


def directory(repo, identity):
    return repo / ".pingstore/runs" / identity


def synthetic_simulate(graph, spikes):
    steps, trials, _ = spikes.shape
    e = np.zeros((steps, trials, recipe.N_E), dtype=np.uint8)
    i = np.zeros((steps, trials, recipe.N_I), dtype=np.uint8)
    if spikes.any():
        for trial in range(trials):
            e[trial::300, trial] = 1
            i[trial + 30 :: 300, trial] = 1
    return {"e_spikes": e, "i_spikes": i}


@pytest.fixture(scope="module")
def seed_store(tmp_path_factory):
    repo = tmp_path_factory.mktemp("exp083-synthetic")
    with pytest.MonkeyPatch.context() as patch:
        for module in (compute, analyse):
            patch.setattr(module, "REPO", repo)
        patch.setattr(stages, "memberships", lambda _: {"exp083": "demo"})
        patch.setattr(
            stages,
            "_capture_code",
            lambda *a: {"git_commit": "fixture", "dirty": False},
        )
        patch.setattr(compute, "simulate", synthetic_simulate)
        compute_id = compute.compute()
        analysis_id = analyse.analyse(compute_id)
    return repo, compute_id, analysis_id


@pytest.fixture
def lab(tmp_path, monkeypatch, seed_store):
    original, compute_id, analysis_id = seed_store
    shutil.copytree(original / ".pingstore", tmp_path / ".pingstore")
    for module in (compute, analyse, present):
        monkeypatch.setattr(module, "REPO", tmp_path)
    monkeypatch.setattr(
        stages, "memberships", lambda _: {"exp083": "demo", "exp084": "demo"}
    )
    monkeypatch.setattr(
        stages, "_capture_code", lambda *a: {"git_commit": "fixture", "dirty": False}
    )
    monkeypatch.setattr(compute, "simulate", forbid)
    return SimpleNamespace(repo=tmp_path, compute=compute_id, analysis=analysis_id)


def rewrite_record(path, edit):
    record = load_json(path / "run.json")
    edit(record)
    record["payload_digest"] = payload_digest(path)
    write_json_atomic(path / "run.json", record)


def test_recipe_preserves_grid_pairing_and_measurement_policy():
    cfg = recipe.configuration()
    assert cfg["trial_seeds"] == [8300, 8301, 8302, 8303, 8304]
    assert cfg["config"]["rates_hz"] == [0, 25, 50, 75, 100, 125, 150, 200]
    assert cfg["config"]["network_seed"] == 83
    assert cfg["rate_sd_ddof"] == 1
    assert cfg["representative_rates_hz"] == [25, 75, 150]
    assert cfg["display_trial"] == 0
    assert cfg["lag_bin_ms"] == 1 and cfg["lag_max_ms"] == 20
    low, high = recipe.make_inputs(25), recipe.make_inputs(150)
    assert np.all(low <= high)
    cfg["config"]["rates_hz"].clear()
    assert len(recipe.configuration()["config"]["rates_hz"]) == 8


def test_compute_retains_every_trial_population_and_input(lab):
    run = inputs.source(lab.repo, lab.compute, "compute")
    metadata, graph, manifest = evidence.compute_payload(run)
    assert run.record["schema"] == "pingstore.run/v3"
    assert run.record["inputs"] == {}
    assert run.record["collection"] == "demo"
    assert metadata["graph"]["digest"] == manifest["graph_digest"]
    assert graph == recipe.author_network().graph
    for condition in recipe.conditions():
        arrays = evidence.recording(run.export / condition["file"])
        np.testing.assert_array_equal(
            arrays["input_spikes"], recipe.make_inputs(condition["input_rate_hz"])
        )
    assert not list(run.export.rglob("*.svg"))
    assert not (run.export / "numbers.json").exists()
    assert not (lab.repo / ".artifacts").exists()


def test_analysis_matches_original_estimators_and_raster_selection(lab):
    source = inputs.source(lab.repo, lab.compute, "compute")
    analysis = inputs.source(lab.repo, lab.analysis, "analyse")
    result, _, _ = evidence.analysis_payload(analysis, source)
    assert analysis.record["inputs"] == {"compute": source.reference}
    saved = evidence.display_arrays(analysis, result["rasters"], "rasters")
    for condition, summary in zip(
        recipe.conditions(), result["conditions"], strict=True
    ):
        arrays = evidence.recording(source.export / condition["file"])
        rate = condition["input_rate_hz"]
        estimate = analyse.estimate_gamma_from_raster(
            arrays["e_spikes"], dt_ms=recipe.DT_MS, config=recipe.FREQUENCY_CONFIG
        )
        rows = measurements._trial_rows(
            rate, arrays["e_spikes"], arrays["i_spikes"], estimate
        )
        assert summary == measurements.summarize_condition(rate, rows)
        if rate in saved:
            for population in ("e", "i"):
                t, cells = np.nonzero(
                    arrays[f"{population}_spikes"][:, recipe.DISPLAY_TRIAL]
                )
                np.testing.assert_array_equal(saved[rate][f"{population}_t"], t)
                np.testing.assert_array_equal(saved[rate][f"{population}_cells"], cells)
    assert result["conditions"][0]["rhythm_frequency_median_hz"] is None
    assert result["conditions"][0]["rhythmicity_score_median"] == 0
    assert not list(analysis.export.rglob("*.png"))


def test_present_draws_only_saved_analysis_and_has_flat_export(lab, monkeypatch):
    monkeypatch.setattr(recipe, "author_network", forbid)
    monkeypatch.setattr(recipe, "make_inputs", forbid)
    monkeypatch.setattr(analyse, "analyse", forbid)
    monkeypatch.setattr(analyse, "estimate_gamma_from_raster", forbid)
    monkeypatch.setattr(measurements, "_trial_rows", forbid)
    monkeypatch.setattr(measurements, "summarize_condition", forbid)
    before = {
        p.name: payload_digest(p) for p in (lab.repo / ".pingstore/runs").iterdir()
    }
    identity = present.present(lab.analysis)
    run = inputs.source(lab.repo, identity, "present")
    assert identity == "exp083-r003-present"
    assert run.record["inputs"] == {
        "analysis": inputs.source(lab.repo, lab.analysis, "analyse").reference
    }
    assert {p.name for p in run.export.iterdir()} == {
        "network.svg",
        "representative_rasters.png",
        "response.png",
        "spectra.png",
        "numbers.json",
        "protocol.json",
        "_manifest.json",
    }
    assert all(p.is_file() for p in run.export.iterdir())
    result = load_json(directory(lab.repo, lab.analysis) / "export/results.json")
    numbers = load_json(run.export / "numbers.json")
    for key in (
        "conditions",
        "config",
        "graph",
        "frequency_analysis",
        "representative_rates_hz",
        "question",
    ):
        assert numbers[key] == result[key]
    assert len(discover_runs(lab.repo / ".pingstore/runs")) == 1
    assert before == {
        name: payload_digest(directory(lab.repo, name)) for name in before
    }
    assert not (lab.repo / ".artifacts").exists()


def test_analysis_never_authors_graph_generates_inputs_or_plots(lab, monkeypatch):
    monkeypatch.setattr(recipe, "author_network", forbid)
    monkeypatch.setattr(recipe, "make_inputs", forbid)
    monkeypatch.setattr(present, "present", forbid)
    for name in ("plot_response", "plot_representative_rasters", "plot_spectra"):
        monkeypatch.setattr(present.plots, name, forbid)
    identity = analyse.analyse(lab.compute)
    inputs.source(lab.repo, identity, "analyse")


@pytest.mark.parametrize(
    "stage,source", [("analyse", "analysis"), ("present", "compute")]
)
def test_wrong_stage_rejected(lab, stage, source):
    with pytest.raises(PingstoreError):
        getattr(globals()[stage], stage)(getattr(lab, source))


@pytest.mark.parametrize(
    "mutation", ["payload", "root", "symlink", "schema", "manifest", "recipe", "roles"]
)
def test_rejects_invalid_compute_or_changed_ancestor(lab, mutation):
    root = directory(lab.repo, lab.compute)
    if mutation == "payload":
        (root / "export/evidence.json").write_text("{}")
    elif mutation == "root":
        (root / "unexpected.txt").write_text("unexpected")
    elif mutation == "symlink":
        (root / "export/alias").symlink_to("evidence.json")
    else:

        def edit(record):
            if mutation == "schema":
                record["schema"] = "pingstore.run/v2"
            elif mutation == "manifest":
                record["execution"]["extra"] = "changed"
            elif mutation == "recipe":
                record["execution"]["configuration"]["trial_seeds"] = [1]
            else:
                record["inputs"] = {
                    "invalid": inputs.source(
                        lab.repo, lab.analysis, "analyse"
                    ).reference
                }

        rewrite_record(root, edit)
    with pytest.raises((PingstoreError, OSError)):
        present.present(lab.analysis)
    assert not any(p.name.endswith("present") for p in root.parent.iterdir())


@pytest.mark.parametrize(
    "fault", ["missing", "shape", "dtype", "binary", "keys", "grid", "graph"]
)
def test_analysis_rejects_semantically_invalid_but_rehashed_evidence(lab, fault):
    root = directory(lab.repo, lab.compute)
    path = root / "export/conditions/rate-0.npz"
    if fault == "missing":
        path.unlink()
    elif fault == "grid":
        record = load_json(root / "export/evidence.json")
        record["conditions"].pop()
        write_json_atomic(root / "export/evidence.json", record)
    elif fault == "graph":
        write_json_atomic(
            root / "export/network.bundle/graph.json", {"name": "changed"}
        )
    else:
        arrays = evidence.recording(path)
        if fault == "shape":
            arrays["e_spikes"] = arrays["e_spikes"][:-1]
        elif fault == "dtype":
            arrays["e_spikes"] = arrays["e_spikes"].astype(np.float32)
        elif fault == "binary":
            arrays["e_spikes"][0, 0, 0] = 2
        else:
            del arrays["input_spikes"]
        np.savez_compressed(path, **arrays)
    rewrite_record(root, lambda _: None)
    with pytest.raises((PingstoreError, OSError)):
        analyse.analyse(lab.compute)


@pytest.mark.parametrize(
    "fault", ["trial_grid", "raster_bounds", "spectrum_nan", "display_path"]
)
def test_present_rejects_semantically_invalid_but_rehashed_analysis(lab, fault):
    root = directory(lab.repo, lab.analysis)
    if fault in ("trial_grid", "display_path"):
        result = load_json(root / "export/results.json")
        if fault == "trial_grid":
            result["conditions"][0]["trials"].pop()
        else:
            result["rasters"][0]["file"] = "../../escape.npz"
        write_json_atomic(root / "export/results.json", result)
    elif fault == "raster_bounds":
        np.savez_compressed(
            root / "export/rasters/rate-25.npz",
            e_t=[-1],
            e_cells=[0],
            i_t=[0],
            i_cells=[0],
        )
    else:
        np.savez_compressed(
            root / "export/spectra/rate-25.npz", frequencies_hz=[1.0], mean_psd=[np.nan]
        )
    rewrite_record(root, lambda _: None)
    with pytest.raises(PingstoreError):
        present.present(lab.analysis)


def test_source_mutation_during_analysis_cannot_complete(lab, monkeypatch):
    original = analyse.estimate_gamma_from_raster
    count = 0

    def mutate(*args, **kwargs):
        nonlocal count
        count += 1
        if count == 1:
            (directory(lab.repo, lab.compute) / "README.md").write_text(
                "changed during analysis"
            )
        return original(*args, **kwargs)

    monkeypatch.setattr(analyse, "estimate_gamma_from_raster", mutate)
    with pytest.raises(PingstoreError):
        analyse.analyse(lab.compute)
    assert directory(lab.repo, ".exp083-r003-analyse.tmp").exists()
    assert not directory(lab.repo, "exp083-r003-analyse").exists()


def test_ancestor_mutation_during_presentation_cannot_complete(lab, monkeypatch):
    def mutate(*args, **kwargs):
        (directory(lab.repo, lab.compute) / "README.md").write_text("changed ancestor")

    monkeypatch.setattr(present.plots, "plot_response", mutate)
    monkeypatch.setattr(present.plots, "plot_spectra", lambda *a: None)
    monkeypatch.setattr(present.plots, "plot_representative_rasters", lambda *a: None)
    with pytest.raises(PingstoreError):
        present.present(lab.analysis)
    assert directory(lab.repo, ".exp083-r003-present.tmp").exists()
    assert not directory(lab.repo, "exp083-r003-present").exists()


def test_compute_failure_stays_hidden_and_cannot_resume(lab, monkeypatch):
    calls = 0

    def fail(graph, spikes):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("synthetic failure")
        return synthetic_simulate(graph, spikes)

    monkeypatch.setattr(compute, "simulate", fail)
    identity = stages.reserve_stage(lab.repo / ".pingstore", "exp083", "compute")
    with pytest.raises(RuntimeError, match="synthetic failure"):
        compute.compute(run_id=identity)
    temporary = directory(lab.repo, f".{identity}.tmp")
    assert (temporary / "export/conditions/rate-0.npz").is_file()
    assert not directory(lab.repo, identity).exists()
    with pytest.raises(PingstoreError, match="interrupted"):
        compute.compute(run_id=identity)


def test_explicit_reservation_and_wrong_reservation(lab, monkeypatch):
    identity = stages.reserve_stage(lab.repo / ".pingstore", "exp083", "analyse")
    with pytest.raises(PingstoreError, match="reservation does not match"):
        compute.compute(run_id=identity)
    assert analyse.analyse(lab.compute, run_id=identity) == identity
    with pytest.raises(PingstoreError, match="unused reserved"):
        analyse.analyse(lab.compute, run_id=identity)
    with pytest.raises(PingstoreError, match="unused reserved"):
        compute.compute(run_id="exp083-r999-compute")


def test_new_compute_is_atomic_and_does_not_analyse(lab, monkeypatch):
    seen = []

    def check(graph, spikes):
        visible = [
            p
            for p in (lab.repo / ".pingstore/runs").iterdir()
            if not p.name.startswith(".")
        ]
        assert len(visible) == 2
        seen.append(graph)
        return synthetic_simulate(graph, spikes)

    monkeypatch.setattr(compute, "simulate", check)
    monkeypatch.setattr(analyse, "estimate_gamma_from_raster", forbid)
    monkeypatch.setattr(measurements, "_trial_rows", forbid)
    monkeypatch.setattr(present, "present", forbid)
    identity = compute.compute()
    assert len(seen) == 8
    assert all(graph is seen[0] for graph in seen)
    validate_operational_run_directory(directory(lab.repo, identity))
    assert not directory(lab.repo, f".{identity}.tmp").exists()


def test_real_execution_adapter_preserves_seed_and_drive(monkeypatch):
    import torch

    seen = []

    class Tensor:
        def cpu(self):
            return self

        def numpy(self):
            return np.zeros((2, 1, 1))

    def run(spec):
        seen.append(spec)
        return SimpleNamespace(
            recordings={"population_0": Tensor(), "population_1": Tensor()}
        )

    monkeypatch.setitem(
        sys.modules,
        "execution",
        SimpleNamespace(ExecutionSpec=SimpleNamespace, simulate=run),
    )
    graph = {"fixture": True}
    spikes = np.ones((2, 1, 1), dtype=np.uint8)
    result = compute.simulate(graph, spikes)
    assert seen[0].graph is graph and seen[0].seed == 83
    assert seen[0].kind == "simulate" and seen[0].executor == "graph"
    assert seen[0].inputs["drive"].dtype == torch.float32
    assert result["e_spikes"].dtype == np.uint8


@pytest.mark.parametrize("stage", ["compute", "analyse", "present"])
@pytest.mark.parametrize("form", ["module", "script"])
def test_cli_help_does_not_execute(stage, form):
    target = (
        ["-m", f"experiments.exp083.{stage}"]
        if form == "module"
        else [str(REPO / f"experiments/exp083/{stage}.py")]
    )
    result = subprocess.run(
        [sys.executable, *target, "--help"], cwd=REPO, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "--run-id" in result.stdout
    if stage != "compute":
        assert "--source" in result.stdout


def test_cli_requires_explicit_source_and_retires_combined_runner():
    for target in (
        ["-m", "experiments.exp083.analyse"],
        ["-m", "experiments.exp083.present"],
        ["-m", "experiments.exp083"],
        [str(REPO / "experiments/exp083.py")],
    ):
        result = subprocess.run(
            [sys.executable, *target], cwd=REPO, capture_output=True, text=True
        )
        assert result.returncode != 0


def test_package_import_has_no_execution_or_plotting_dependencies():
    script = "import sys; import experiments.exp083; assert 'execution' not in sys.modules; assert 'matplotlib.pyplot' not in sys.modules"
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=REPO, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_legacy_exp084_interface_is_preserved():
    package = importlib.import_module("experiments.exp083")
    for name in (
        "DT_MS",
        "T_MS",
        "BURN_MS",
        "N_INPUT",
        "N_E",
        "N_I",
        "TRIAL_SEEDS",
        "NETWORK_SEED",
        "FREQUENCY_CONFIG",
        "make_inputs",
        "_trial_rows",
        "summarize_condition",
        "estimate_gamma_from_raster",
    ):
        assert hasattr(package, name)


def test_lag_sign_and_sample_standard_deviation_are_preserved():
    e = np.zeros((10_000, recipe.N_E), dtype=np.uint8)
    i = np.zeros((10_000, recipe.N_I), dtype=np.uint8)
    e[2_500::1000] = 1
    i[2_530::1000] = 1
    assert measurements._phase_lag_ms(e, i) == -3
    rows = [
        {
            "e_rate_hz": float(n),
            "i_rate_hz": float(n * 2),
            "e_i_peak_lag_ms": None,
            "rhythmicity_contrast": None,
            "frequency": {"resolved": False},
        }
        for n in range(5)
    ]
    summary = measurements.summarize_condition(0, rows)
    assert summary["e_rate_std_hz"] == np.std(np.arange(5), ddof=1)


def test_wrong_experiment_and_hidden_sources_are_rejected(lab):
    with stages.stage_run(
        lab.repo, "exp084", "compute", configuration=recipe.configuration()
    ) as run:
        (run.export / "evidence.json").write_text("{}")
    with pytest.raises(PingstoreError, match="does not belong"):
        analyse.analyse(run.run_id)
    reservation = stages.reserve_stage(lab.repo / ".pingstore", "exp083", "compute")
    with pytest.raises((PingstoreError, OSError)):
        analyse.analyse(reservation)
    with pytest.raises((PingstoreError, OSError)):
        analyse.analyse(f".{reservation}.tmp")


def test_present_failure_never_exposes_a_completed_run(lab, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("synthetic plotting failure")

    monkeypatch.setattr(present.plots, "plot_response", fail)
    with pytest.raises(RuntimeError, match="synthetic plotting failure"):
        present.present(lab.analysis)
    assert directory(lab.repo, ".exp083-r003-present.tmp").exists()
    assert not directory(lab.repo, "exp083-r003-present").exists()
    inputs.source(lab.repo, lab.analysis, "analyse")
