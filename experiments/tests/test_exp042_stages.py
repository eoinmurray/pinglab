"""Synthetic fixtures only: no historical import, dataset download or scientific run."""

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from experiments.exp042 import (
    analyse,
    collection,
    compute,
    import_gold2,
    inputs,
    present,
    recipe,
    simulation,
    transforms,
)
from pingstore import stages
from pingstore.contracts import (
    PingstoreError,
    file_sha256,
    load_json,
    write_json_atomic,
)


@pytest.fixture
def lab(tmp_path, monkeypatch):
    for module in (compute, analyse, present, import_gold2):
        monkeypatch.setattr(module, "REPO", tmp_path)
    monkeypatch.setattr(
        stages, "memberships", lambda _: {"exp022": "demo", "exp042": "demo"}
    )

    def code(*args):
        return {"git_commit": "fixture", "dirty": False, "code_dirty": False}

    monkeypatch.setattr(stages, "_capture_code", code)
    monkeypatch.setattr(compute, "_capture_code", code)
    monkeypatch.setattr(recipe, "RASTER_N_E_PLOT", 2)
    monkeypatch.setattr(recipe, "RASTER_N_I_PLOT", 1)
    monkeypatch.setenv("PINGLAB_SMOKE", "1")
    with stages.stage_run(
        tmp_path, "exp022", "compute", export_root="export/cells"
    ) as bank:
        for seed in recipe.SEEDS:
            cell = bank.export / "cells" / recipe.cell_name(seed)
            cell.mkdir(parents=True)
            config = {
                "training_run_id": "TR-02",
                "training_cell_name": cell.name,
                "seed": seed,
                "dataset": "mnist",
                "ei_strength": 1.0,
                "fr_reg_upper_strength": 0.0,
                "dt": 0.1,
                "t_ms": 2.0,
                "n_hidden": 4,
                "n_inh": 2,
                "epochs": 50,
                "tau_gaba_ms": 6.0,
                "ei_ratio": 2.0,
                "w_in": [0.9, 0.09],
            }
            write_json_atomic(cell / "config.json", config)
            (cell / "weights_final.pth").write_bytes(b"final epoch")
            (cell / "weights.pth").write_bytes(b"not the selected checkpoint")
            write_json_atomic(
                cell / "metrics.json",
                {
                    "config": config,
                    "checkpoints": {
                        "final_epoch": {
                            "filename": "weights_final.pth",
                            "epoch": 50,
                            "sha256": file_sha256(cell / "weights_final.pth"),
                        }
                    },
                },
            )
    calls = []

    def simulate(args):
        calls.append(args)

        def get(key):
            return args[args.index(key) + 1]

        cfg = load_json(Path(get("--load-config")))
        assert Path(get("--load-weights")).name == "weights_final.pth"
        out = Path(get("--out-dir"))
        out.mkdir(parents=True, exist_ok=True)
        if "--sample-index" in args:
            assert "--max-samples" not in args
            e, i = np.zeros((20, 4), dtype=bool), np.zeros((20, 2), dtype=bool)
            e[::4] = True
            i[::3] = True
            mode = get("--recording-mode")
            assert mode == ("spikes" if "--i-override-file" in args else "inhibitory")
            arrays = {"spk_i": i, "label": np.int64(0)}
            if mode == "spikes":
                arrays["spk_e"] = e
            np.savez(out / "snapshot.npz", **arrays)
        else:
            samples = int(get("--max-samples"))
            write_json_atomic(
                out / "metrics.json",
                {
                    "best_acc": 90 + cfg["seed"] - 42,
                    "n_total": samples,
                    "rates_hz": {"hid": 10 + cfg["seed"] - 42, "inh": 20},
                    "config": {
                        "evaluation_samples": samples,
                        "evaluation_partition": "official_mnist_test",
                    },
                },
            )
        if "--outputs" in args and get("--outputs") == "rasters":
            assert get("--recording-mode") == "inhibitory"
            np.savez(
                out / "rasters.npz",
                T=np.int32(20),
                n_i=np.int32(2),
                n_trials=np.int32(1),
                i_trial=np.array([0, 0]),
                i_t=np.array([2, 12]),
                i_cell=np.array([0, 1]),
            )

    monkeypatch.setattr(simulation, "run_cli", simulate)
    return tmp_path, bank.run_id, calls


def test_stages_preserve_small_evidence_and_never_run_upstream(lab, monkeypatch):
    root, bank_id, calls = lab
    bank = inputs.source(root, bank_id, "compute", experiment="exp022")
    before = bank.reference
    old = root / ".artifacts/exp042/xtau_raw_sweeps.svg"
    old.parent.mkdir(parents=True)
    old.write_text("unrelated historical view")
    identity = compute.compute(bank_id)
    raw = inputs.source(root, identity, "compute")
    assert len(list(raw.export.glob("jobs/*.json"))) == 39
    assert len(calls) == 39  # 36 distinct evaluations plus three illustrative launches
    for job in recipe.jobs(recipe.configuration(smoke=True)):
        if job["condition"] == "cell_jitter_sigma_0":
            row = load_json(raw.export / "jobs" / (job["id"] + ".json"))
            source = recipe.replay_job(job)["id"]
            assert row["replay_of"] == source
            assert (
                row["metrics"]
                == load_json(raw.export / "jobs" / (source + ".json"))["metrics"]
            )
    assert raw.record["inputs"] == {"bank": before}
    assert not list(raw.directory.glob(".scratch-*"))
    assert not list(raw.export.rglob("rasters.npz"))
    for filename in ("cell.npz", "cycle.npz"):
        with np.load(raw.export / filename) as data:
            assert set(data.files) == {"spk_e", "spk_i", "label"}
    assert {p.name for p in raw.export.iterdir()} == {
        "jobs",
        "cell.npz",
        "cycle.npz",
        "evidence.json",
    }
    monkeypatch.setattr(
        simulation, "run_cli", lambda *a: pytest.fail("downstream simulation")
    )
    monkeypatch.setenv("PINGLAB_SMOKE", "0")
    analysis_id = analyse.analyse(identity)
    analysis = inputs.source(root, analysis_id, "analyse")
    results = load_json(analysis.export / "results.json")
    assert results["recipe"]["profile"] == "smoke"
    assert results["aggregate"]["jitter_sweep"][0]["e_rate_hz"]["mean"] == 11
    assert results["aggregate"]["jitter_sweep"][0]["e_rate_hz"]["sem"] == pytest.approx(
        1 / np.sqrt(3)
    )
    assert results["rasters"]["cycle"]["e_rate_hz"] == 2500
    presentation_id = present.present(analysis_id)
    presentation = inputs.source(root, presentation_id, "present")
    assert presentation.record["inputs"] == {"analysis": analysis.reference}
    assert all(path.is_file() for path in presentation.export.iterdir())
    assert {p.name for p in presentation.export.iterdir()} == {
        *recipe.FIGURES,
        "numbers.json",
        "_manifest.json",
    }
    assert (
        inputs.source(root, bank_id, "compute", experiment="exp022").reference == before
    )
    assert old.read_text() == "unrelated historical view"


def test_missing_metric_fails_without_simulation(lab, monkeypatch):
    root, bank_id, _ = lab
    identity = compute.compute(bank_id)
    raw = inputs.source(root, identity, "compute")
    next(raw.export.glob("jobs/*.json")).unlink()
    monkeypatch.setattr(
        simulation, "run_cli", lambda *a: pytest.fail("missing-data fallback")
    )
    with pytest.raises(PingstoreError, match="checksum"):
        analyse.analyse(identity)


@pytest.mark.parametrize("stage", ["analyse", "present"])
def test_wrong_source_stage_rejected_before_reservation(lab, stage):
    root, bank_id, calls = lab
    before = sorted(p.name for p in (root / ".pingstore/runs").iterdir())
    with pytest.raises(PingstoreError):
        getattr({"analyse": analyse, "present": present}[stage], stage)(bank_id)
    assert before == sorted(p.name for p in (root / ".pingstore/runs").iterdir())
    assert not calls


def test_v2_and_missing_ancestor_rejected_before_reservation(lab):
    root, bank_id, calls = lab
    path = root / ".pingstore/runs" / bank_id / "run.json"
    record = load_json(path)
    write_json_atomic(path, {**record, "schema": "pingstore.run/v2"})
    with pytest.raises(PingstoreError, match="v3"):
        compute.compute(bank_id)
    record["inputs"] = {
        "missing": {
            "run_id": "exp022-r999-compute-local",
            "payload_digest": "sha256:" + "a" * 64,
            "run_json_sha256": "a" * 64,
        }
    }
    write_json_atomic(path, record)
    with pytest.raises(PingstoreError):
        compute.compute(bank_id)
    assert len(list((root / ".pingstore/runs").iterdir())) == 1
    assert not calls


def test_failure_is_hidden_and_source_immutable(lab, monkeypatch):
    root, bank_id, _ = lab
    source = inputs.source(root, bank_id, "compute", experiment="exp022")
    monkeypatch.setattr(
        simulation,
        "run_cli",
        lambda *a: (_ for _ in ()).throw(RuntimeError("fixture failure")),
    )
    with pytest.raises(RuntimeError, match="fixture failure"):
        compute.compute(bank_id)
    assert not list((root / ".pingstore/runs").glob("exp042-*"))
    failed = next((root / ".pingstore/runs").glob(".exp042-*"))
    assert not list(failed.glob(".scratch-*"))
    source.check_unchanged()


def test_shards_are_pinned_resumable_and_collect_without_repeating_sweeps(lab):
    root, bank_id, calls = lab
    identity = stages.reserve_stage(
        root / ".pingstore", "exp042", "compute", origin="slurm-wilkes"
    )
    with pytest.raises((PingstoreError, OSError)):
        compute.compute(bank_id, run_id=identity, collect=True)
    for index in range(8):
        compute.shard(bank_id, run_id=identity, index=index)
    before = len(calls)
    assert before == 36  # shared baselines and zero replays run once, not per shard
    compute.shard(bank_id, run_id=identity, index=0)
    assert len(calls) == before
    compute.compute(bank_id, run_id=identity, collect=True)
    assert (
        len(calls) == before + 3
    )  # only two illustrative arms and their shared baseline
    output = inputs.source(root, identity, "compute")
    assert len(list(output.export.glob("jobs/*.json"))) == 39
    assert output.record["origin"] == "slurm-wilkes"
    assert not (output.directory / ".baseline-scratch").exists()
    with pytest.raises((PingstoreError, OSError)):
        compute.shard(bank_id, run_id=identity, index=0)


def test_production_retains_all_rows_with_66_launches(lab, monkeypatch):
    root, bank_id, calls = lab
    monkeypatch.setenv("PINGLAB_SMOKE", "0")
    identity = compute.compute(bank_id)
    raw = inputs.source(root, identity, "compute")
    assert len(list(raw.export.glob("jobs/*.json"))) == 66
    assert len(calls) == 66
    assert sum("--sample-index" in args for args in calls) == 3


def test_zero_replay_is_shared_between_concurrent_workers(lab):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    root, bank_id, calls = lab
    bank = inputs.source(root, bank_id, "compute", experiment="exp022")
    cfg = recipe.configuration(smoke=True)
    jobs = [
        j
        for j in recipe.jobs(cfg)
        if j["seed"] == 42
        and j["condition"] in ("jitter_sigma_0", "cell_jitter_sigma_0")
    ]
    barrier = Barrier(2)

    def worker(index):
        scratch = root / f"worker-{index}"
        scratch.mkdir()
        simulator = simulation.Simulator(
            scratch, scratch / "commands", cfg, baseline_root=root / "shared"
        )
        barrier.wait(timeout=5)
        job = jobs[index]
        return simulator.evaluate(bank.export / job["cell"], job)

    with ThreadPoolExecutor(max_workers=2) as pool:
        a, b = list(pool.map(worker, range(2)))
    assert a == b
    assert len(calls) == 2  # one baseline and one zero replay
    assert sum("--i-override-file" in args for args in calls) == 1


def test_zero_replay_cache_rejects_recipe_drift(lab):
    root, bank_id, _ = lab
    bank = inputs.source(root, bank_id, "compute", experiment="exp022")
    cfg = recipe.configuration(smoke=True)
    job = next(j for j in recipe.jobs(cfg) if j["condition"] == "jitter_sigma_0")
    scratch = root / "scratch"
    scratch.mkdir()
    simulator = simulation.Simulator(scratch, scratch / "commands", cfg)
    simulator.evaluate(bank.export / job["cell"], job)
    path = next(scratch.glob("zero-replay/*/*/result.json"))
    record = load_json(path)
    record["recipe"]["evaluation_samples"] += 1
    write_json_atomic(path, record)
    with pytest.raises(ValueError, match="zero-replay scratch"):
        simulator.evaluate(bank.export / job["cell"], job)


def test_shard_does_not_reuse_tampered_metrics(lab):
    root, bank_id, _ = lab
    identity = stages.reserve_stage(root / ".pingstore", "exp042", "compute")
    record = compute.shard(bank_id, run_id=identity, index=0)
    directory = root / ".pingstore/runs" / f".{identity}.tmp"
    (directory / "export/jobs" / (record["jobs"][0] + ".json")).write_text("{}")
    with pytest.raises(PingstoreError, match="changed"):
        compute.shard(bank_id, run_id=identity, index=0)


def test_collection_dispatches_explicit_sources_and_reservations(lab, monkeypatch):
    root, bank_id, _ = lab
    manifest = root / "bank.json"
    write_json_atomic(manifest, {"pingstore_run_id": bank_id})
    row = {
        "slug": "exp042",
        "execution": {"mode": "exp042-staged"},
        "paths": {"state": str(root / "campaign")},
        "required_outputs": [str(root / "campaign/stage-refs.json")],
    }
    plan = {"exp022_manifest": str(manifest), "profile": "smoke"}
    commands = []

    def dispatch(args, **kwargs):
        commands.append(args)
        name = args[2].rsplit(".", 1)[1]
        module = {"compute": compute, "analyse": analyse, "present": present}[name]
        source = args[args.index("--source") + 1]
        identity = args[args.index("--run-id") + 1]
        getattr(module, name)(source, run_id=identity)
        return subprocess.CompletedProcess(args, 0, stdout=identity + "\n")

    monkeypatch.setattr(collection.subprocess, "run", dispatch)
    refs = collection.execute(root, plan, row)
    assert len(commands) == 3
    assert refs["bank"]["run_id"] == bank_id
    assert (
        collection.completed(root, plan, row).record["run_id"]
        == refs["present"]["run_id"]
    )
    collection.execute(root, plan, row)
    assert len(commands) == 3


def test_transforms_match_retained_algorithm():
    # Compare against deterministic expected properties, not archived scientific data.
    import torch

    baseline = torch.zeros(30, 2, 3)
    baseline[::5] = 1
    for name in ("jitter_sigma_0", "cell_jitter_sigma_0"):
        actual = transforms._build_override(
            baseline, name, torch.Generator().manual_seed(42)
        )
        assert torch.equal(actual, baseline)
    phase = transforms._build_override(
        baseline, "phase_shuffled_i", torch.Generator().manual_seed(42)
    )
    assert torch.equal(phase.sum(0), baseline.sum(0))
    for name in ("jitter_sigma_14", "cell_jitter_sigma_14", "poisson_matched_i"):
        a = transforms._build_override(
            baseline, name, torch.Generator().manual_seed(42)
        )
        b = transforms._build_override(
            baseline, name, torch.Generator().manual_seed(42)
        )
        assert torch.equal(a, b)
        assert set(a.unique().tolist()) <= {0.0, 1.0}


def test_article_renders_fixture_and_unavailable_data_states(lab):
    from demolab_cli import _paths

    root, bank_id, _ = lab
    identity = present.present(analyse.analyse(compute.compute(bank_id)))
    output = inputs.source(root, identity, "present")
    source_root = Path(__file__).resolve().parents[2]
    (root / "writings").mkdir()
    for name in ("exp042.typ", "run-inputs.typ"):
        shutil.copy2(source_root / "writings" / name, root / "writings" / name)
    (root / ".demolab").mkdir()
    shutil.copy2(_paths.TYP / "lib.typ", root / ".demolab/lib.typ")
    mapping = {"exp042": {"exp042": "/" + str(output.export.relative_to(root))}}
    write_json_atomic(root / "preview.json", mapping)
    document = root / "document.typ"
    document.write_text(
        '#set page(paper: "a4", margin: 18mm)\n#set text(size: 10pt)\n'
        "#set page(header: [SYNTHETIC TEST FIXTURE — NOT SCIENTIFIC RESULTS])\n"
        '#import "writings/exp042.typ": body\n#body\n'
    )
    command = [
        str(_paths.find_typst(source_root)),
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
    write_json_atomic(root / "preview.json", {"exp042": {"exp042": None}})
    command[-1] = str(root / "pending-{p}.png")
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    write_json_atomic(root / "preview.json", mapping)
    (output.export / "numbers.json").write_text("corrupt")
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode != 0


def test_active_worker_blocks_collection(lab):
    root, bank_id, _ = lab
    identity = stages.reserve_stage(root / ".pingstore", "exp042", "compute")
    directory = root / ".pingstore/runs" / f".{identity}.tmp"
    with compute._compute_lock(directory, exclusive=False):
        with pytest.raises(PingstoreError, match="busy"):
            compute.compute(bank_id, run_id=identity, collect=True)
    assert not (directory / "run.json").exists()


def test_ancestor_manifest_drift_prevents_downstream_completion(lab):
    root, bank_id, _ = lab
    compute_id = compute.compute(bank_id)
    bank = inputs.source(root, bank_id, "compute", experiment="exp022")
    record = dict(bank.record)
    record["execution"] = {**record["execution"], "note": "fixture manifest change"}
    write_json_atomic(bank.directory / "run.json", record)
    with pytest.raises(PingstoreError, match="checksum changed"):
        analyse.analyse(compute_id)


def test_import_has_no_implicit_paths_or_execution(tmp_path, monkeypatch):
    monkeypatch.setenv("PINGLAB_TRAINING_ROOT", str(tmp_path / "missing-bank"))
    monkeypatch.setenv("PINGLAB_RUN_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("PINGLAB_RUN_DERIVED_DIR", str(tmp_path / "derived"))
    monkeypatch.setenv("PINGLAB_RUN_LOG_DIR", str(tmp_path / "logs"))
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from experiments import exp042; "
            "assert not hasattr(exp042, 'RUN_PATHS'); "
            "assert not hasattr(exp042, 'TRAINING_ROOT'); "
            "assert len(exp042.jobs(exp042.configuration())) == 66",
        ],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert not list(tmp_path.iterdir())


def test_combined_runner_is_retired():
    result = subprocess.run(
        [sys.executable, "-m", "experiments.exp042", "--skip-training"],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "independent stages" in result.stderr


@pytest.fixture
def historical_subset(lab):
    """Tiny invented archive matching the real import layout, never R2 data."""
    root, bank_id, calls = lab
    archive = root / "archive"
    archive.mkdir()
    bank = inputs.source(root, bank_id, "compute", experiment="exp022")
    evidence = inputs.bank_evidence(bank)
    cfg = recipe.configuration()
    groups = {key: [] for key in ("results", "jitter_sweep", "cell_jitter_sweep")}
    tag = "final_epoch__" + evidence["checkpoints"][0]["sha256"][:12]

    def family(directory, training, *, condition=None, snapshot=False):
        directory.mkdir(parents=True, exist_ok=True)
        config = {
            **training,
            "n_hidden": [training["n_hidden"]],
            "tau_gaba": training["tau_gaba_ms"],
            "infer": True,
            "model": "ping",
            "load_weights": f"/original/{training['training_cell_name']}/weights_final.pth",
            "max_samples": cfg["evaluation_samples"],
            "sample_index": 0,
            "i_override_file": f"/original/{condition}.npz",
        }
        write_json_atomic(directory / "config.json", config)
        for name in ("run.sh", "run.jsonl", "output.log"):
            (directory / name).write_text("synthetic execution record\n")
        if snapshot:
            np.savez(
                directory / "snapshot.npz",
                spk_e=np.zeros((20, 4), dtype=np.float32),
                spk_i=np.ones((20, 2), dtype=np.float32),
                label=np.int32(7),
                unused_voltage=np.ones((20, 4)),
            )

    for job in recipe.jobs(cfg):
        cell, condition = job["cell"], job["condition"]
        training = evidence["configurations"][cell]
        stem = f"{cell}_{condition}_{job['seed_offset']}"
        directory = (
            archive
            / import_gold2.STATE
            / (
                f"baseline/{cell}/{tag}"
                if condition == "baseline"
                else f"ovrun/{cell}__{stem}/{tag}"
            )
        )
        directory.mkdir(parents=True, exist_ok=True)
        if condition != "baseline":
            family(directory, training, condition=stem)
        metrics = {
            "best_acc": 90.0,
            "n_total": cfg["evaluation_samples"],
            "rates_hz": {"hid": 10.0, "inh": 20.0},
            "config": {
                **training,
                "evaluation_partition": cfg["evaluation_partition"],
                "load_weights": f"/original/{cell}/weights_final.pth",
            },
        }
        write_json_atomic(directory / "metrics.json", metrics)
        groups[job["group"]].append(analyse.measurement(metrics, job, cfg))
    cell = recipe.cell_name(42)
    for condition in ("jitter_sigma_14", "cell_jitter_sigma_14"):
        family(
            archive / import_gold2.STATE / f"condraster/{cell}_{condition}_s0/{tag}",
            evidence["configurations"][cell],
            snapshot=True,
            condition=f"{cell}_{condition}_s0_ov",
        )
    derived = archive / import_gold2.DERIVED
    derived.mkdir(parents=True)
    for name in (*recipe.FIGURES, "_manifest.json", "run.sh"):
        (derived / name).write_text("synthetic original comparison output")
    write_json_atomic(
        derived / "numbers.json",
        {
            **groups,
            "checkpoint_provenance": evidence["checkpoints"],
            "config": {
                "seeds": cfg["seeds"],
                "conditions": cfg["conditions"],
                "jitter_sigmas_ms": cfg["jitter_sigmas_ms"],
                "evaluation_samples_per_condition": cfg["evaluation_samples"],
                "raster_sample_idx": 0,
                "f_gamma_reference_hz": cfg["f_gamma_reference_hz"],
            },
        },
    )
    base = archive / "provenance/source-records/base"
    for name in (
        "run.json",
        "inventory.json",
        "collection-plan.json",
        "collection-status/exp042.json",
        "submissions/collection-submission.json",
    ):
        write_json_atomic(base / name, {"fixture": True})
    rows = [
        {
            "path": str(p.relative_to(archive)),
            "size_bytes": p.stat().st_size,
            "sha256": file_sha256(p),
            "role": "fixture",
        }
        for p in sorted(archive.rglob("*"))
        if p.is_file()
    ]
    write_json_atomic(
        archive / "inventory.json",
        {
            "run_id": "gold-2",
            "contract_version": "runstore/v1",
            "files": rows,
            "file_count": len(rows),
            "total_size_bytes": sum(r["size_bytes"] for r in rows),
        },
    )
    write_json_atomic(
        archive / "run.json",
        {
            "run_id": "gold-2",
            "contract_version": "runstore/v1",
            "archive": {"uri": import_gold2.URI},
        },
    )
    return archive, import_gold2.make_plan(archive, bank_id)


def test_gold2_import_preserves_originals_and_supports_independent_stages(
    lab, historical_subset
):
    root, bank_id, calls = lab
    archive, plan = historical_subset
    bank_before = inputs.source(root, bank_id, "compute", experiment="exp022").reference
    identity = import_gold2.import_subset(archive, plan)
    raw = inputs.source(root, identity, "compute")
    assert raw.record["execution"]["operation"] == "historical-import"
    assert raw.record["historical_import"]["simulation_executed"] is False
    import_gold2.verify_files(raw.directory / "provenance/gold-2", plan)
    import_gold2.verify_files(archive, plan)
    assert len(list(raw.export.glob("jobs/*.json"))) == 66
    with np.load(raw.export / "cell.npz") as data:
        assert set(data.files) == {"spk_e", "spk_i", "label"}
        assert data["spk_e"].dtype == np.float32
    output = inputs.source(root, present.present(analyse.analyse(identity)), "present")
    numbers = load_json(output.export / "numbers.json")
    original = load_json(archive / import_gold2.DERIVED / "numbers.json")
    for group in ("results", "jitter_sweep", "cell_jitter_sweep"):
        assert numbers[group] == original[group]
    assert (
        inputs.source(root, bank_id, "compute", experiment="exp022").reference
        == bank_before
    )
    assert not calls
    assert not (root / ".artifacts").exists()


@pytest.mark.parametrize(
    "failure",
    ["checksum", "missing", "plan", "numbers", "configuration", "snapshot", "symlink"],
)
def test_gold2_import_rejects_bad_evidence_before_reservation(
    lab, historical_subset, failure
):
    root, _, calls = lab
    archive, plan = historical_subset
    target = archive / plan["jobs"][3]["source"]
    if failure == "checksum":
        target.write_text("corrupted")
    elif failure == "missing":
        target.unlink()
    elif failure == "plan":
        plan["recipe"]["evaluation_samples"] = 10
    elif failure == "symlink":
        target.rename(target.with_suffix(".original"))
        target.symlink_to(target.with_suffix(".original"))
    else:
        if failure == "numbers":
            target = archive / import_gold2.DERIVED / "numbers.json"
            data = load_json(target)
            data["results"][0]["acc"] = 1
            write_json_atomic(target, data)
        elif failure == "configuration":
            target = target.with_name("config.json")
            data = load_json(target)
            data["tau_gaba"] = 12
            write_json_atomic(target, data)
        else:
            target = archive / plan["recordings"]["cell"]
            np.savez(
                target, spk_e=np.full((20, 4), 0.5), spk_i=np.ones((20, 2)), label=7
            )
        # Valid checksums do not excuse scientifically inconsistent source data.
        inventory = load_json(archive / "inventory.json")
        for row in inventory["files"]:
            if row["path"] == str(target.relative_to(archive)):
                row.update(size_bytes=target.stat().st_size, sha256=file_sha256(target))
        inventory["total_size_bytes"] = sum(r["size_bytes"] for r in inventory["files"])
        write_json_atomic(archive / "inventory.json", inventory)
        plan = import_gold2.make_plan(archive, plan["bank"]["run_id"])
    before = sorted(p.name for p in (root / ".pingstore/runs").iterdir())
    with pytest.raises((PingstoreError, OSError)):
        import_gold2.import_subset(archive, plan)
    assert sorted(p.name for p in (root / ".pingstore/runs").iterdir()) == before
    assert not calls
