"""Temporary synthetic banks and mocked inference; never run scientific experiments."""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from experiments.exp041 import (
    analyse,
    collection,
    compute,
    evidence,
    inputs,
    present,
    recipe,
)
from experiments.exp044.test import _common_config
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
    for module in (compute, analyse, present):
        monkeypatch.setattr(module, "REPO", tmp_path)
    monkeypatch.setattr(
        stages, "memberships", lambda _: {"exp022": "demo", "exp041": "demo"}
    )
    monkeypatch.setattr(
        stages, "_capture_code", lambda *args: {"git_commit": "fixture", "dirty": False}
    )
    monkeypatch.setattr(recipe, "RASTER_N_E_PLOT", 2)
    monkeypatch.setattr(recipe, "RASTER_N_I_PLOT", 1)
    monkeypatch.setenv("PINGLAB_SMOKE", "1")
    with stages.stage_run(tmp_path, "exp022", "compute") as bank:
        for tau in recipe.TAU_GABA_SWEEP:
            for seed in recipe.SEEDS:
                name = recipe.cell_name(tau, seed)
                cell = bank.export / name
                cell.mkdir(parents=True)
                cfg = {
                    **_common_config(),
                    "dt": recipe.DT_TRAIN,
                    "tau_gaba_ms": tau,
                    "seed": seed,
                    "n_hidden": 4,
                    "n_inh": 2,
                    "hidden_sizes": [4],
                }
                write_json_atomic(cell / "config.json", cfg)
                checkpoints = {}
                for role, filename in (
                    ("final_epoch", "weights_final.pth"),
                    ("best_validation", "weights.pth"),
                ):
                    path = cell / filename
                    path.write_bytes(role.encode())
                    checkpoints[role] = {
                        "filename": filename,
                        "epoch": recipe.EPOCHS,
                        "sha256": file_sha256(path),
                    }
                write_json_atomic(
                    cell / "metrics.json",
                    {
                        "config": cfg,
                        "training_cell_name": name,
                        "best_epoch": recipe.EPOCHS,
                        "checkpoints": checkpoints,
                        "epochs": [
                            {
                                "ep": ep,
                                "acc": 80.0 + ep / 10,
                                "test_rate_e": ep / 2 + tau,
                            }
                            for ep in range(1, recipe.EPOCHS + 1)
                        ],
                    },
                )
    calls = []

    def simulate(args, **kwargs):
        calls.append(args)

        def arg(key):
            return args[args.index(key) + 1]

        out = Path(arg("--out-dir"))
        out.mkdir(parents=True)
        cfg = load_json(Path(arg("--load-config")))
        assert arg("--device") == "auto"
        assert Path(arg("--load-weights")).name == "weights_final.pth"
        if "--sample-index" in args:
            assert "--max-samples" not in args
            steps = round(cfg["t_ms"] / cfg["dt"])
            e, i = np.zeros((steps, 4), dtype=bool), np.zeros((steps, 2), dtype=bool)
            e[::20, :] = True
            i[::40, :] = True
            np.savez_compressed(
                out / "recording.npz", spk_e=e, spk_i=i, dt=cfg["dt"], label=3
            )
        else:
            samples = int(arg("--max-samples"))
            write_json_atomic(
                out / "metrics.json",
                {
                    "config": {
                        **cfg,
                        "evaluation_partition": "official_mnist_test",
                        "evaluation_samples": samples,
                    },
                    "best_acc": 90.0,
                    "n_correct": 9 * samples // 10,
                    "n_total": samples,
                    "rates_hz": {
                        "hid": 1.15
                        + 0.18 * (50 - cfg["tau_gaba_ms"] + (cfg["seed"] - 42)),
                        "inh": 20.0,
                    },
                },
            )
            t = np.arange(round(cfg["t_ms"] / cfg["dt"])) * cfg["dt"] / 1000
            wave = 0.1 + 0.05 * np.sin(2 * np.pi * (50 - cfg["tau_gaba_ms"]) * t)
            np.savez_compressed(
                out / "pop_traces.npz", pop_e=np.tile(wave, (samples, 1)), dt=cfg["dt"]
            )
        write_json_atomic(out / "config.json", cfg)
        (out / "run.sh").write_text("fixture\n")

    monkeypatch.setattr(compute, "run_cli", simulate)
    return tmp_path, bank.run_id, calls


def resign(directory):
    record = load_json(directory / "run.json")
    record["payload_digest"] = payload_digest(directory)
    write_json_atomic(directory / "run.json", record)


def test_ancestor_metadata_amendment_during_stage_is_allowed(lab):
    root, bank_id, _ = lab
    bank = inputs.source(root, bank_id, "compute", experiment="exp022")
    compute_id = compute.compute(bank_id)
    source = inputs.source(root, compute_id, "compute")
    with inputs.execution(root, "analyse", sources={"compute": source}) as run:
        record = load_json(bank.directory / "run.json")
        record["execution"]["changed"] = True
        write_json_atomic(bank.directory / "run.json", record)
        write_json_atomic(run.export / "fixture.json", {})
    assert (root / ".pingstore/runs" / run.run_id).is_dir()


def test_v2_is_rejected_even_with_an_incomplete_payload(lab):
    root, bank_id, calls = lab
    path = root / ".pingstore/runs" / bank_id / "run.json"
    r = load_json(path)
    r["schema"] = "pingstore.run/v2"
    write_json_atomic(path, r)
    with pytest.raises(PingstoreError, match="requires v4"):
        compute.compute(bank_id)
    assert calls == []


def test_changed_authoritative_ancestor_metadata_is_allowed(lab):
    root, bank_id, _ = lab
    compute_id = compute.compute(bank_id)
    path = root / ".pingstore/runs" / bank_id / "run.json"
    r = load_json(path)
    r["execution"]["note"] = "manifest-only change"
    write_json_atomic(path, r)
    assert analyse.analyse(compute_id).endswith("-analyse")


def test_failure_stays_hidden_and_reservations_cannot_be_reused(lab, monkeypatch):
    root, bank_id, _ = lab
    identity = stages.reserve_stage(root / ".pingstore", recipe.SLUG, "compute")

    def fail(*a, **k):
        raise RuntimeError("fixture failure")

    monkeypatch.setattr(compute, "run_cli", fail)
    with pytest.raises(RuntimeError, match="fixture failure"):
        compute.compute(bank_id, run_id=identity)
    assert (root / ".pingstore/runs" / f".{identity}.tmp").is_dir()
    assert not (root / ".pingstore/runs" / identity).exists()
    with pytest.raises(PingstoreError, match="interrupted execution"):
        compute.compute(bank_id, run_id=identity)


@pytest.mark.parametrize(
    "damage", ["missing_rate", "zero_samples", "nan_rate", "missing_history"]
)
def test_missing_measurements_never_become_zeros(lab, damage):
    root, bank_id, calls = lab
    if damage == "missing_history":
        bank = inputs.source(root, bank_id, "compute", experiment="exp022")
        path = bank.export / recipe.cell_name(4.5, 42) / "metrics.json"
        r = load_json(path)
        r["epochs"].pop()
        write_json_atomic(path, r)
        resign(bank.directory)
        with pytest.raises(PingstoreError, match="incomplete training history"):
            compute.compute(bank_id)
        assert not calls
        return
    compute_id = compute.compute(bank_id)
    output = inputs.source(root, compute_id, "compute")
    path = output.file("infer", recipe.cell_name(4.5, 42), "metrics.json")
    r = load_json(path)
    if damage == "missing_rate":
        del r["rates_hz"]["hid"]
    elif damage == "zero_samples":
        r["n_total"] = 0
    else:
        r["rates_hz"]["hid"] = float("nan")
    write_json_atomic(path, r)
    resign(output.directory)
    with pytest.raises(PingstoreError):
        analyse.analyse(compute_id)
    assert len(calls) == 24


def test_collection_reserves_and_dispatches_explicit_stages(lab, monkeypatch):
    from experiments.collections.gamma_gated_sparsity.plan import build_plan

    root, bank_id, _ = lab
    plan = build_plan(root / "campaign", "fixture", smoke=True)
    plan["profile"] = "smoke"
    plan["exp022_manifest"] = str(root / "bank-manifest.json")
    write_json_atomic(Path(plan["exp022_manifest"]), {"pingstore_run_id": bank_id})
    row = next(
        r for s in plan["stages"] for r in s["experiments"] if r["slug"] == "exp041"
    )
    assert row["command"] == []
    assert row["execution"]["stages"] == ["compute", "analyse", "present"]
    ids = collection.reserve(root, row, origin="slurm-wilkes")
    for stage, identity in ids.items():
        assert identity.endswith("-" + stage)
        reservation = load_json(
            root
            / ".pingstore/runs"
            / f".{identity}.tmp"
            / ".reservation.json"
        )
        assert reservation["origin"] == "slurm-wilkes"
    assert collection.reserve(root, row) == ids
    with pytest.raises(PingstoreError, match="legacy exp041"):
        collection.require_staged({"execution": {"mode": "monolithic"}})
    commands = []

    def dispatch(command, **kwargs):
        commands.append(command)
        stage = command[2].rsplit(".", 1)[-1]
        module = {"compute": compute, "analyse": analyse, "present": present}[stage]
        getattr(module, stage)(
            command[command.index("--source") + 1],
            run_id=command[command.index("--run-id") + 1],
        )
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(collection.subprocess, "run", dispatch)
    refs = collection.execute(root, plan, row)
    assert len(commands) == 3
    assert collection.execute(root, plan, row) == refs
    assert len(commands) == 3
    assert (
        collection.completed(root, plan, row).record["run_id"]
        == refs["present"]["run_id"]
    )
    plan["profile"] = "production"
    with pytest.raises(PingstoreError, match="profile"):
        collection.execute(root, plan, row)


def test_independent_stages_retain_science_and_never_publish(lab, monkeypatch):
    from experiments.exp041 import measurements

    root, bank_id, calls = lab
    bank = inputs.source(root, bank_id, "compute", experiment="exp022")
    before = bank.reference
    compute_id = compute.compute(bank_id)
    assert len(calls) == 24
    assert sum("--sample-index" in args for args in calls) == 6
    output = inputs.source(root, compute_id, "compute")
    assert output.record["inputs"] == {"bank": before}
    assert len(list(output.export.glob("infer--*/pop_traces.npz"))) == 18
    assert not list(output.export.rglob("run.sh"))
    monkeypatch.setenv("PINGLAB_SMOKE", "0")
    monkeypatch.setattr(
        compute, "run_cli", lambda *a, **k: pytest.fail("downstream inference")
    )
    analysis_id = analyse.analyse(compute_id)
    analysis_run = inputs.source(root, analysis_id, "analyse")
    results = load_json(analysis_run.export / "results.json")
    assert results["config"]["evaluation_samples"] == 100
    assert len(results["results"]) == 18
    assert len(results["aggregate"]) == 6
    assert results["measurement"]["history_partition"] == "validation"
    raw = evidence.snapshot(
        output.file("snapshot", recipe.cell_name(4.5, 42), "recording.npz"),
        recipe.DT_TRAIN,
        results["config"]["training_contract"]["common"],
    )
    rng = np.random.default_rng(0)
    e_idx = np.sort(rng.choice(4, 2, replace=False))
    i_idx = np.sort(rng.choice(2, 1, replace=False))
    assert results["rasters"][0]["e_indices"] == e_idx.tolist()
    assert results["rasters"][0]["i_indices"] == i_idx.tolist()
    assert results["rasters"][0]["e_rate_hz"] == pytest.approx(500)
    with np.load(analysis_run.export / "rasters.npz") as data:
        np.testing.assert_array_equal(
            data[recipe.cell_name(4.5, 42) + "__e"], raw["spk_e"][:, e_idx]
        )
    for name in ("spectrum", "summarize", "fit_law"):
        monkeypatch.setattr(
            measurements, name, lambda *a: pytest.fail("presentation remeasurement")
        )
    monkeypatch.setattr(
        evidence,
        "histories",
        lambda *a: pytest.fail("presentation reads raw histories"),
    )
    present_id = present.present(analysis_id)
    presentation = inputs.source(root, present_id, "present")
    assert presentation.record["inputs"] == {"analysis": analysis_run.reference}
    assert all((presentation.export / name).is_file() for name in recipe.FIGURES)
    assert all(p.is_file() for p in presentation.export.iterdir())
    numbers = load_json(presentation.export / "numbers.json")
    assert numbers["results"] == results["results"] and numbers["fit"] == results["fit"]
    assert (
        "aggregate" not in numbers and "per_trial_peaks_hz" not in numbers["results"][0]
    )
    assert not (root / ".artifacts").exists()
    assert (
        inputs.source(root, bank_id, "compute", experiment="exp022").reference == before
    )
    assert present_id not in (presentation.export / "rate_vs_fgamma.svg").read_text()


def test_mean_psd_peak_not_median_of_trial_peaks():
    from experiments.exp041 import measurements

    t = np.arange(2000) * 0.0001
    traces = np.array(
        [
            1 + np.sin(2 * np.pi * 20 * t),
            1 + 0.05 * np.sin(2 * np.pi * 60 * t),
            1 + 0.05 * np.sin(2 * np.pi * 60 * t),
        ]
    )
    result = measurements.spectrum(traces, 0.1)
    assert result["f_gamma_hz"] == pytest.approx(20, abs=0.01)
    assert np.median(result["per_trial_peaks_hz"]) == pytest.approx(60, abs=0.01)
    assert np.diff(result["freqs_hz"])[0] == 5
    assert np.isnan(measurements._peak_with_parabolic(np.zeros(5), np.arange(5) * 5))


def test_affine_and_origin_fits_use_six_condition_means():
    from experiments.exp041 import measurements

    rows = []
    for tau in recipe.TAU_GABA_SWEEP:
        for seed in recipe.SEEDS:
            f = 50 - tau
            rows.append(
                {
                    "tau_gaba_ms": tau,
                    "f_gamma_hz": f,
                    "e_rate_hz": 1.15 + 0.18 * f + (seed - 43),
                    "acc": 90,
                    "freqs_hz": [0, 5, 10, 15],
                    "psd": [0, 0.5, 1, 0.2],
                    "per_trial_peaks_hz": [10],
                }
            )
    aggregate = measurements.summarize(rows)
    assert len(aggregate) == 6
    assert aggregate[0]["e_rate_hz"]["sem"] == pytest.approx(1 / np.sqrt(3))
    fit = measurements.fit_law(aggregate)
    assert fit["a_affine"] == pytest.approx(1.15)
    assert fit["p_affine"] == pytest.approx(0.18)
    assert fit["r2_affine"] == pytest.approx(1)
    assert fit["r2_origin"] < 1


@pytest.mark.parametrize(
    "damage",
    [
        "traces_shape",
        "traces_dt",
        "nan_trace",
        "flat_spectrum",
        "tau",
        "snapshot_shape",
    ],
)
def test_corrupt_scientific_evidence_fails_closed(lab, damage):
    root, bank_id, calls = lab
    identity = compute.compute(bank_id)
    run = inputs.source(root, identity, "compute")
    directory = run.unit("infer", recipe.cell_name(4.5, 42))
    if damage == "tau":
        path = directory / "metrics.json"
        data = load_json(path)
        data["config"]["tau_gaba_ms"] = 99
        write_json_atomic(path, data)
    elif damage == "snapshot_shape":
        path = run.file("snapshot", recipe.cell_name(4.5, 42), "recording.npz")
        np.savez_compressed(
            path, spk_e=np.zeros((1, 4)), spk_i=np.zeros((1, 2)), dt=0.1, label=3
        )
    else:
        path = directory / "pop_traces.npz"
        with np.load(path) as data:
            arrays = {k: np.array(data[k]) for k in data.files}
        if damage == "traces_shape":
            arrays["pop_e"] = arrays["pop_e"][:1]
        if damage == "traces_dt":
            arrays["dt"] = 1.0
        if damage == "nan_trace":
            arrays["pop_e"][0, 0] = np.nan
        if damage == "flat_spectrum":
            arrays["pop_e"][:] = 0
        np.savez_compressed(path, **arrays)
    with pytest.raises(PingstoreError, match="checksum"):
        analyse.analyse(identity)
    resign(run.directory)
    with pytest.raises(PingstoreError):
        analyse.analyse(identity)
    assert len(calls) == 24
    assert not list((root / ".pingstore/runs").glob("exp041-*-analyse"))


def test_strict_lineage_does_not_ignore_missing_historical_input(lab):
    root, bank_id, calls = lab
    path = root / ".pingstore/runs" / bank_id / "run.json"
    record = load_json(path)
    record["inputs"] = {
        "missing": {
            "run_id": "exp022-r999-compute",
            "payload_digest": "sha256:" + "0" * 64,
        }
    }
    write_json_atomic(path, record)
    with pytest.raises((PingstoreError, FileNotFoundError)):
        compute.compute(bank_id)
    assert not calls


def test_inference_caps_and_import_side_effects(tmp_path):
    assert recipe.configuration()["evaluation_samples"] == 1000
    assert recipe.configuration(smoke=True)["evaluation_samples"] == 100
    args = recipe.inference_args(
        Path("cell"),
        Path("weights_final.pth"),
        Path("out"),
        samples=100,
        tau_gaba_ms=6,
        sample_index=50,
    )
    assert (
        "--max-samples" not in args and args[args.index("--sample-index") + 1] == "50"
    )
    args = recipe.inference_args(
        Path("cell"), Path("weights_final.pth"), Path("out"), samples=100, tau_gaba_ms=6
    )
    assert args[args.index("--max-samples") + 1] == "100"
    assert args[args.index("--outputs") + 1] == "pop_traces"
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from experiments import exp041; assert not hasattr(exp041,'RUN_PATHS'); assert not hasattr(exp041,'cell_dir'); assert exp041.cell_name(4.5,42)=='ping__tg4p5__seed42'",
        ],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("slug", ["exp054"])
def test_migrated_consumers_require_explicit_exp041_stage_references(
    tmp_path, monkeypatch, slug
):
    from experiments.collections.gamma_gated_sparsity import execution
    from experiments.collections.gamma_gated_sparsity.plan import build_plan

    plan = build_plan(tmp_path / "campaign", "fixture", smoke=True)
    row = next(
        r for stage in plan["stages"] for r in stage["experiments"] if r["slug"] == slug
    )
    monkeypatch.setattr(
        execution.subprocess,
        "run",
        lambda *a, **k: pytest.fail("legacy downstream launch"),
    )
    with pytest.raises(PingstoreError, match="requires completed exp041 analysis"):
        execution._run_downstream(plan, row)


def test_article_renders_only_explicit_present_inputs(lab):
    import shutil

    from demolab_cli import _paths

    root, bank_id, _ = lab
    compute_id = compute.compute(bank_id)
    analysis_id = analyse.analyse(compute_id)
    present_id = present.present(analysis_id)
    output = inputs.source(root, present_id, "present")
    source_root = Path(__file__).resolve().parents[2]
    shutil.copytree(source_root / "writings", root / "writings")
    (root / ".demolab").mkdir()
    shutil.copy2(_paths.TYP / "lib.typ", root / ".demolab/lib.typ")
    write_json_atomic(
        root / "preview.json",
        {"exp041": {"exp041": "/" + str(output.export.relative_to(root))}},
    )
    document = root / "document.typ"
    document.write_text(
        '#set page(paper: "a4", margin: 18mm)\n#set text(size: 10pt)\n#block(fill: yellow.lighten(60%), inset: 8pt)[Synthetic test data — not scientific results.]\n#import "writings/exp041.typ": body\n#body\n'
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
    (output.export / "numbers.json").write_text("corrupt")
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode != 0
