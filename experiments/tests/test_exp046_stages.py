"""Synthetic cycle evidence; no training, production inference or historical import."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from experiments.exp041 import analyse as upstream_analyse
from experiments.exp041 import compute as upstream_compute
from experiments.exp046 import (
    analyse,
    collection,
    compute,
    import_gold2,
    inputs,
    measurements,
    present,
    recipe,
)
from experiments.tests import test_exp041_stages as upstream_tests
from pingstore import stages
from pingstore.contracts import (
    PingstoreError,
    file_sha256,
    load_json,
    payload_digest,
    write_json_atomic,
)


@pytest.fixture
def lab(tmp_path, monkeypatch):
    return upstream_tests.lab.__wrapped__(tmp_path, monkeypatch)


@pytest.fixture
def cycle_lab(lab, monkeypatch):
    root, bank_id, _ = lab
    monkeypatch.setattr(
        stages,
        "memberships",
        lambda _: {"exp022": "demo", "exp041": "demo", "exp046": "demo"},
    )
    upstream_compute_id = upstream_compute.compute(bank_id)
    frequency_id = upstream_analyse.analyse(upstream_compute_id)
    for module in (compute, analyse, present):
        monkeypatch.setattr(module, "REPO", root)
    calls = []

    def simulate(args, **kwargs):
        calls.append(args)

        def value(key):
            return args[args.index(key) + 1]

        assert value("--device") == "auto"
        assert args[args.index("--outputs") + 1 : args.index("--recording-mode")] == [
            "rasters",
            "per_cell_rates",
        ]
        output = Path(value("--out-dir"))
        output.mkdir(parents=True)
        config = load_json(Path(value("--load-config")))
        samples = int(value("--max-samples"))
        steps = round(config["t_ms"] / config["dt"])
        rows = {f"{p}_{key}": [] for p in ("e", "i") for key in ("trial", "t", "cell")}
        for trial in range(samples):
            for centre in range(100, steps - 30, 200):
                for prefix, times, cells in (
                    ("i", [centre] if trial else [], [0]),
                    ("e", [centre - 2, centre, centre + 2], [0, 1]),
                ):
                    for t in times:
                        for cell in cells:
                            rows[prefix + "_trial"].append(trial)
                            rows[prefix + "_t"].append(t)
                            rows[prefix + "_cell"].append(cell)
        raster = {k: np.array(v, dtype=np.int32) for k, v in rows.items()}
        raster.update(
            dt=np.float32(config["dt"]),
            n_trials=np.int32(samples),
            T=np.int32(steps),
            n_e=np.int32(4),
            n_i=np.int32(2),
        )
        np.savez_compressed(output / "rasters.npz", **raster)
        rates = (
            np.bincount(raster["e_cell"], minlength=4)
            / (samples * config["t_ms"] / 1000)
        ).astype(np.float32)
        np.savez_compressed(output / "per_cell_rates.npz", rate_e_per_cell=rates)
        write_json_atomic(
            output / "metrics.json",
            {
                "config": {
                    **config,
                    "evaluation_partition": "official_mnist_test",
                    "evaluation_samples": samples,
                },
                "n_total": samples,
                "n_correct": samples * 9 // 10,
                "best_acc": 90.0,
                "rates_hz": {"hid": float(rates.mean()), "inh": 1.0},
            },
        )
        write_json_atomic(output / "config.json", config)
        (output / "run.sh").write_text("synthetic command\n")

    monkeypatch.setattr(compute, "run_cli", simulate)
    return root, bank_id, frequency_id, calls


def resign(run):
    record = load_json(run.directory / "run.json")
    record["payload_digest"] = payload_digest(run.directory)
    write_json_atomic(run.directory / "run.json", record)


def test_independent_stages_preserve_outputs_without_publication(
    cycle_lab, monkeypatch
):
    root, bank_id, frequency_id, calls = cycle_lab
    identity = compute.compute(bank_id)
    assert len(calls) == 18
    assert all(c[c.index("--max-samples") + 1] == "100" for c in calls)
    output = inputs.source(root, identity, "compute")
    assert set(output.record["inputs"]) == {"bank"}
    assert len(list(output.export.glob("infer/*/rasters.npz"))) == 18
    assert not list(output.export.glob("infer/*/config.json"))
    monkeypatch.setattr(
        compute, "run_cli", lambda *a, **k: pytest.fail("implicit inference")
    )
    analysis_id = analyse.analyse(identity, frequency_id)
    analysis_run = inputs.source(root, analysis_id, "analyse")
    data = load_json(analysis_run.export / "results.json")
    assert len(data["results"]) == 18
    assert set(analysis_run.record["inputs"]) == {"compute", "bank", "frequencies"}
    assert data["n_cell_cycle_pairs"] == sum(
        sum(r["bucket_counts"]) for r in data["results"]
    )
    assert all(
        sum(r["bucket_counts"]) == 4 * r["n_cycles_observed"] for r in data["results"]
    )
    monkeypatch.setattr(
        measurements,
        "measure",
        lambda *a, **k: pytest.fail("presentation measured cycles"),
    )
    monkeypatch.setattr(
        measurements,
        "summarize",
        lambda *a, **k: pytest.fail("presentation fitted results"),
    )
    present_id = present.present(analysis_id)
    shown = inputs.source(root, present_id, "present")
    numbers = load_json(shown.export / "numbers.json")
    for key in ("results", "global_fracs", "per_tau", "ceiling", "n_cell_cycle_pairs"):
        assert numbers[key] == data[key]
    assert all((shown.export / name).is_file() for name in recipe.FIGURES)
    assert not (root / ".artifacts").exists()


def test_cycle_count_boundaries_and_zero_peak_trials():
    spikes = np.ones((9, 2), dtype=np.int8)
    np.testing.assert_array_equal(
        measurements.count_e_spikes_per_cycle(spikes, np.array([2, 5])),
        [[3, 3], [6, 6]],
    )
    assert measurements.count_e_spikes_per_cycle(
        spikes, np.array([], dtype=int)
    ).shape == (0, 2)
    assert measurements.detect_i_burst_steps(np.zeros((2000, 2)), 0.1, 50).size == 0


def test_summary_pools_counts_and_fits_all_networks():
    rows = [
        {
            "tau_gaba_ms": tau,
            "f_gamma_hz": f,
            "per_cell_max_rate_hz": 0.8 * f,
            "bucket_counts": buckets,
        }
        for tau, f, buckets in [
            (6, 40, [90, 10, 0, 0]),
            (6, 42, [0, 10, 0, 0]),
            (12, 25, [0, 0, 10, 0]),
        ]
    ]
    result = measurements.summarize(rows)
    assert result["per_tau"]["tau_6"]["frac_zero"] == pytest.approx(90 / 110)
    assert result["n_cell_cycle_pairs"] == 120
    assert result["ceiling"]["max_cell_slope_vs_fgamma"] == pytest.approx(0.8)
    assert result["ceiling"]["max_cell_r2"] == pytest.approx(1)


@pytest.mark.parametrize(
    "damage", ["bounds", "duplicate", "trial_count", "dt", "rates", "nan", "missing"]
)
def test_bad_recordings_fail_closed(cycle_lab, damage):
    root, bank_id, frequency_id, _ = cycle_lab
    identity = compute.compute(bank_id)
    run = inputs.source(root, identity, "compute")
    path = run.export / "infer" / recipe.cell_name(4.5, 42)
    if damage in ("rates", "nan"):
        with np.load(path / "per_cell_rates.npz") as data:
            rates = data["rate_e_per_cell"].copy()
        rates[0] = np.nan if damage == "nan" else 12345
        np.savez(path / "per_cell_rates.npz", rate_e_per_cell=rates)
    elif damage == "missing":
        (path / "rasters.npz").unlink()
    else:
        with np.load(path / "rasters.npz") as data:
            arrays = {k: data[k] for k in data.files}
        if damage == "bounds":
            arrays["e_cell"][0] = 4
        if damage == "duplicate":
            for key in ("e_trial", "e_t", "e_cell"):
                arrays[key][1] = arrays[key][0]
        if damage == "trial_count":
            arrays["n_trials"] = np.int32(99)
        if damage == "dt":
            arrays["dt"] = 0.2
        np.savez(path / "rasters.npz", **arrays)
    resign(run)
    with pytest.raises((PingstoreError, OSError)):
        analyse.analyse(identity, frequency_id)
    assert not list((root / ".pingstore/runs").glob("exp046-*-analyse"))


@pytest.mark.parametrize(
    "damage", ["missing", "duplicate", "zero", "nan", "profile", "bank"]
)
def test_frequency_dependency_must_match_complete_grid(cycle_lab, damage):
    root, bank_id, frequency_id, _ = cycle_lab
    identity = compute.compute(bank_id)
    frequency = inputs.source(root, frequency_id, "analyse", experiment="exp041")
    data = load_json(frequency.export / "results.json")
    if damage == "missing":
        data["results"].pop()
    if damage == "duplicate":
        data["results"][-1] = data["results"][0]
    if damage == "zero":
        data["results"][0]["f_gamma_hz"] = 0
    if damage == "nan":
        data["results"][0]["f_gamma_hz"] = float("nan")
    if damage == "profile":
        data["recipe"]["profile"] = "production"
    if damage == "bank":
        data["checkpoint_provenance"][0]["sha256"] = "0" * 64
    write_json_atomic(frequency.export / "results.json", data)
    resign(frequency)
    with pytest.raises(PingstoreError):
        analyse.analyse(identity, frequency_id)
    assert not list((root / ".pingstore/runs").glob("exp046-*-analyse"))


def test_source_corruption_and_wrong_stage_rejected(cycle_lab):
    root, bank_id, frequency_id, _ = cycle_lab
    with pytest.raises(PingstoreError):
        compute.compute(frequency_id)
    identity = compute.compute(bank_id)
    run = inputs.source(root, identity, "compute")
    (run.export / "evidence.json").write_text("corrupt")
    with pytest.raises(PingstoreError):
        analyse.analyse(identity, frequency_id)


def test_failed_compute_stays_hidden(cycle_lab, monkeypatch):
    root, bank_id, _, _ = cycle_lab
    monkeypatch.setattr(
        compute,
        "run_cli",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("simulator failed")),
    )
    with pytest.raises(RuntimeError, match="simulator failed"):
        compute.compute(bank_id)
    assert not list((root / ".pingstore/runs").glob("exp046-*-compute"))
    assert list((root / ".pingstore/runs").glob(".exp046-*-compute.tmp"))


@pytest.mark.parametrize("damage", ["v2", "ancestor"])
def test_historical_schema_and_changed_ancestor_are_rejected(cycle_lab, damage):
    root, bank_id, frequency_id, _ = cycle_lab
    identity = compute.compute(bank_id)
    run = inputs.source(root, identity, "compute")
    if damage == "v2":
        record = load_json(run.directory / "run.json")
        record["schema"] = "pingstore.run/v2"
        write_json_atomic(run.directory / "run.json", record)
    else:
        bank = inputs.source(root, bank_id, "compute", experiment="exp022")
        (bank.directory / "README.md").write_text("changed after pinning")
        resign(bank)
    with pytest.raises(PingstoreError):
        analyse.analyse(identity, frequency_id)
    assert not list((root / ".pingstore/runs").glob("exp046-*-analyse"))


def test_ancestor_mutation_during_analysis_prevents_completion(cycle_lab, monkeypatch):
    root, bank_id, frequency_id, _ = cycle_lab
    identity = compute.compute(bank_id)
    bank = inputs.source(root, bank_id, "compute", experiment="exp022")
    measure = measurements.measure

    def changed(*args, **kwargs):
        result = measure(*args, **kwargs)
        (bank.directory / "README.md").write_text("changed during analysis")
        return result

    monkeypatch.setattr(measurements, "measure", changed)
    with pytest.raises(PingstoreError):
        analyse.analyse(identity, frequency_id)
    assert not list((root / ".pingstore/runs").glob("exp046-*-analyse"))


def test_retired_entrypoints_fail_without_outputs(tmp_path):
    root = Path(__file__).resolve().parents[2]
    for command in (
        [sys.executable, str(root / "experiments/exp046.py")],
        [sys.executable, "-m", "experiments.exp046"],
    ):
        result = subprocess.run(command, cwd=root, capture_output=True, text=True)
        assert result.returncode != 0
        assert "explicit" in result.stderr


def test_import_has_no_storage_side_effects_and_preserves_sample_caps(tmp_path):
    assert recipe.configuration()["evaluation_samples"] == 1000
    assert recipe.configuration(smoke=True)["evaluation_samples"] == 100
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from experiments import exp046; assert not hasattr(exp046, 'RUN_PATHS'); assert exp046.cell_name(4.5,42)=='ping__tg4p5__seed42'",
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(root)},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert not list(tmp_path.iterdir())


def test_unchanged_article_renders_explicit_present_outputs(cycle_lab):
    import shutil

    from demolab_cli import _paths

    root, bank_id, frequency_id, _ = cycle_lab
    identity = compute.compute(bank_id)
    analysis_id = analyse.analyse(identity, frequency_id)
    present_id = present.present(analysis_id)
    output = inputs.source(root, present_id, "present")
    source_root = Path(__file__).resolve().parents[2]
    shutil.copytree(source_root / "writings", root / "writings")
    (root / ".demolab").mkdir()
    shutil.copy2(_paths.TYP / "lib.typ", root / ".demolab/lib.typ")
    write_json_atomic(
        root / "preview.json",
        {"exp046": {"exp046": "/" + str(output.export.relative_to(root))}},
    )
    document = root / "document.typ"
    document.write_text(
        '#set page(paper: "a4", margin: 18mm, header: [Synthetic data; article claims unchanged and not reviewed.])\n'
        '#set text(size: 10pt)\n#import "writings/exp046.typ": body\n#body\n'
    )
    command = [
        _paths.find_typst(source_root),
        "compile",
        "--root",
        str(root),
        "--input",
        "demolab-preview-file=/preview.json",
        "--format",
        "png",
        "--ppi",
        "90",
        str(document),
        str(root / "article-{p}.png"),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert list(root.glob("article-*.png"))
    (output.export / "spikes_per_cycle_distribution.svg").write_text("corrupt")
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode != 0


def test_collection_dispatch_uses_explicit_frequency_source(cycle_lab, monkeypatch):
    from experiments.collections.gamma_gated_sparsity.plan import build_plan

    root, bank_id, frequency_id, _ = cycle_lab
    plan = build_plan(root / "campaign", "fixture", smoke=True)
    plan["profile"] = "smoke"
    row = next(
        r for s in plan["stages"] for r in s["experiments"] if r["slug"] == "exp046"
    )
    monkeypatch.setattr(
        collection,
        "campaign_bank",
        lambda *a: inputs.source(root, bank_id, "compute", experiment="exp022"),
    )
    monkeypatch.setattr(
        collection,
        "campaign_frequencies",
        lambda *a: inputs.source(root, frequency_id, "analyse", experiment="exp041"),
    )
    calls = []

    def dispatch(command, **kwargs):
        from types import SimpleNamespace

        calls.append(command)

        def value(key):
            return command[command.index(key) + 1]

        stage = command[2].rsplit(".", 1)[-1]
        if stage == "compute":
            compute.compute(value("--source"), run_id=value("--run-id"))
        elif stage == "analyse":
            analyse.analyse(
                value("--source"), value("--frequency-source"), run_id=value("--run-id")
            )
        else:
            present.present(value("--source"), run_id=value("--run-id"))
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(collection.subprocess, "run", dispatch)
    collection.reserve(root, row, origin="slurm")
    refs = collection.execute(root, plan, row)
    assert len(calls) == 3
    assert calls[1][calls[1].index("--frequency-source") + 1] == frequency_id
    assert (
        inputs.source(root, refs["compute"]["run_id"], "compute").record["origin"]
        == "slurm"
    )
    collection.execute(root, plan, row)
    assert len(calls) == 3
    plan["profile"] = "production"
    with pytest.raises(PingstoreError, match="profile"):
        collection.execute(root, plan, row)
    assert len(calls) == 3


def test_collection_requires_completed_frequency_dependency(cycle_lab):
    from experiments.collections.gamma_gated_sparsity.plan import build_plan

    root, _, _, _ = cycle_lab
    plan = build_plan(root / "campaign", "fixture", smoke=True)
    with pytest.raises(PingstoreError, match="completed exp041 analysis"):
        collection.campaign_frequencies(root, plan)
    assert not list((root / ".pingstore/runs").glob("exp046-*"))


def test_legacy_collection_plan_rejected():
    with pytest.raises(PingstoreError, match="legacy"):
        collection.require_staged({"execution": {"mode": "monolithic"}})


def _gold2_fixture(cycle_lab, monkeypatch):
    root, bank_id, frequency_id, _ = cycle_lab
    identity = compute.compute(bank_id)
    analysis_id = analyse.analyse(identity, frequency_id)
    source = inputs.source(root, identity, "compute")
    result = inputs.source(root, analysis_id, "analyse")
    raw = load_json(source.export / "evidence.json")
    monkeypatch.setattr(import_gold2, "REPO", root)
    monkeypatch.setattr(recipe, "EVAL_MAX_SAMPLES", 100)
    monkeypatch.setattr(import_gold2.archive_helpers.recipe, "EVAL_MAX_SAMPLES", 100)
    configuration = recipe.configuration
    monkeypatch.setattr(
        recipe, "configuration", lambda **kwargs: configuration(smoke=True)
    )
    archive = root / "gold-2"
    write_json_atomic(
        archive / "run.json",
        {
            "run_id": "gold-2",
            "contract_version": "runstore/v1",
            "archive": {"uri": import_gold2.URI},
        },
    )
    write_json_atomic(
        archive / import_gold2.DERIVED / "numbers.json",
        load_json(result.export / "results.json"),
    )
    for name in ("lineage.json", f"{import_gold2.BASE}/collection-plan.json"):
        write_json_atomic(archive / name, {})
    write_json_atomic(
        archive / import_gold2.BASE / "run.json",
        {
            "run_id": "historical-campaign",
            "source": {"git_commit": "historical"},
        },
    )
    write_json_atomic(
        archive / import_gold2.BASE / "collection-status/exp046.json",
        {
            "experiment": "exp046",
            "state": "complete",
        },
    )
    write_json_atomic(
        archive / import_gold2.BASE / "submissions/collection-submission.json",
        {
            "jobs": [{"name": "ggs-exp046", "job_id": "fixture"}],
        },
    )
    for cell, checkpoint in zip(
        raw["training_contract"]["cells"], raw["checkpoint_provenance"], strict=True
    ):
        directory = (
            archive
            / import_gold2.STATE
            / "infer"
            / cell["cell_name"]
            / f"final_epoch__{checkpoint['sha256'][:12]}"
        )
        config = {
            **raw["training_contract"]["common"],
            "seed": cell["seed"],
            "tau_gaba": cell["tau_gaba_ms"],
            "infer": True,
            "scale_w_in": 1.0,
            "scale_w_ei": 1.0,
            "scale_w_ie": 1.0,
            "intervention": [],
            "scale_projection": [],
            "max_samples": 100,
            "load_config": f"/historical/{cell['cell_name']}/config.json",
            "load_weights": f"/historical/{cell['cell_name']}/weights_final.pth",
            "outputs": ["rasters", "per_cell_rates"],
        }
        write_json_atomic(directory / "config.json", config)
        for name in import_gold2.SIDECARS[1:]:
            (directory / name).write_text("historical execution evidence\n")
        original = source.export / "infer" / cell["cell_name"]
        metrics = load_json(original / "metrics.json")
        metrics["config"].pop("seed")
        metrics["config"].pop("tau_gaba_ms")
        metrics["config"]["load_weights"] = config["load_weights"]
        write_json_atomic(directory / "metrics.json", metrics)
        for payload in import_gold2.ARRAYS:
            with np.load(original / payload) as data:
                arrays = {k: data[k] for k in data.files}
            arrays["unused_array"] = np.ones(100)
            np.savez(directory / payload, **arrays)
    rows = [
        {
            "path": str(p.relative_to(archive)),
            "size_bytes": p.stat().st_size,
            "sha256": file_sha256(p),
        }
        for p in sorted(archive.rglob("*"))
        if p.is_file() and p != archive / "run.json"
    ]
    write_json_atomic(
        archive / "inventory.json",
        {
            "contract_version": "runstore/v1",
            "run_id": "gold-2",
            "files": rows,
            "file_count": len(rows),
            "total_size_bytes": sum(r["size_bytes"] for r in rows),
        },
    )
    plan = import_gold2.make_plan(archive, bank_id)
    write_json_atomic(root / "import-plan.json", plan)
    return archive, load_json(root / "import-plan.json")


def test_gold2_import_preserves_full_selected_arrays_and_metrics(
    cycle_lab, monkeypatch
):
    import zipfile

    archive, plan = _gold2_fixture(cycle_lab, monkeypatch)
    root, _, frequency_id, calls = cycle_lab
    previous_calls = len(calls)
    identity = import_gold2.import_subset(archive, plan)
    imported = inputs.source(root, identity, "compute")
    assert imported.record["origin"] == "local"
    assert imported.record["execution"]["operation"] == "historical-import"
    assert imported.record["historical_import"]["simulation_executed"] is False
    assert imported.record["historical_import"]["producer"]["origin"] == "slurm"
    for job in plan["jobs"]:
        original = archive / job["directory"]
        target = imported.export / "infer" / job["cell"]["cell_name"]
        for payload, keys in import_gold2.ARRAYS.items():
            with (
                zipfile.ZipFile(original / payload) as old,
                zipfile.ZipFile(target / payload) as new,
            ):
                assert set(new.namelist()) == {k + ".npy" for k in keys}
                for name in new.namelist():
                    assert new.read(name) == old.read(name)
        actual = load_json(target / "metrics.json")
        assert actual["config"].pop("seed") == job["cell"]["seed"]
        assert actual["config"].pop("tau_gaba_ms") == job["cell"]["tau_gaba_ms"]
        assert actual == load_json(original / "metrics.json")
        assert (
            imported.directory / "provenance/gold-2" / job["directory"] / "metrics.json"
        ).read_bytes() == (original / "metrics.json").read_bytes()
    present.present(analyse.analyse(identity, frequency_id))
    assert len(calls) == previous_calls
    assert not (root / ".artifacts").exists()
    import_gold2.verify_files(archive, plan)


@pytest.mark.parametrize(
    "fault", ["checksum", "plan", "checkpoint", "config", "producer", "recording"]
)
def test_gold2_import_rejects_inconsistent_evidence(cycle_lab, monkeypatch, fault):
    archive, plan = _gold2_fixture(cycle_lab, monkeypatch)
    root, _, _, _ = cycle_lab
    before = set((root / ".pingstore/runs").iterdir())
    first = archive / plan["jobs"][0]["directory"]
    if fault == "checksum":
        (first / "metrics.json").write_text("corrupt")
        with pytest.raises(PingstoreError, match="checksum"):
            import_gold2.import_subset(archive, plan)
    elif fault == "plan":
        plan["recipe"]["evaluation_samples"] += 1
        with pytest.raises(PingstoreError, match="plan changed"):
            import_gold2.import_subset(archive, plan)
    else:
        if fault == "checkpoint":
            plan["checkpoints"][0]["sha256"] = "0" * 64
        elif fault == "config":
            config = load_json(first / "config.json")
            config["tau_gaba"] += 1
            write_json_atomic(first / "config.json", config)
        elif fault == "producer":
            write_json_atomic(
                archive / import_gold2.BASE / "submissions/collection-submission.json",
                {"jobs": []},
            )
        else:
            np.savez(first / "per_cell_rates.npz", rate_e_per_cell=np.zeros(4))
        with pytest.raises(PingstoreError):
            import_gold2.validate_science(archive, plan)
    assert set((root / ".pingstore/runs").iterdir()) == before


def test_gold2_source_mutation_prevents_import_completion(cycle_lab, monkeypatch):
    archive, plan = _gold2_fixture(cycle_lab, monkeypatch)
    root, _, _, _ = cycle_lab
    before = {
        p for p in (root / ".pingstore/runs").iterdir() if not p.name.startswith(".")
    }
    extract = import_gold2.archive_helpers.extract_arrays

    def changed(*args):
        result = extract(*args)
        (archive / "lineage.json").write_text("changed during import")
        return result

    monkeypatch.setattr(import_gold2.archive_helpers, "extract_arrays", changed)
    with pytest.raises(PingstoreError, match="checksum"):
        import_gold2.import_subset(archive, plan)
    assert {
        p for p in (root / ".pingstore/runs").iterdir() if not p.name.startswith(".")
    } == before
    assert list((root / ".pingstore/runs").glob(".exp046-*-compute.tmp"))
