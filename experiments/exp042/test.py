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
    for module in (compute, analyse, present):
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
    with stages.stage_run(tmp_path, "exp022", "compute") as bank:
        for seed in recipe.SEEDS:
            cell = bank.export / recipe.cell_name(seed)
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
            np.savez(out / "recording.npz", **arrays)
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
                n_trials=np.int32(samples),
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
    assert len(list(raw.export.glob("jobs/*.json"))) == 30
    assert len(calls) == 33  # 30 sweep evaluations plus three illustrative launches
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
    assert "results" not in results
    assert "conditions" not in results["config"]
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
    with pytest.raises(PingstoreError, match="v4"):
        compute.compute(bank_id)
    record["inputs"] = {
        "missing": {
            "run_id": "exp022-r999-compute",
            "payload_digest": "sha256:" + "a" * 64,
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
    assert before == 30  # shared baselines and zero replays run once, not per shard
    compute.shard(bank_id, run_id=identity, index=0)
    assert len(calls) == before
    compute.compute(bank_id, run_id=identity, collect=True)
    assert (
        len(calls) == before + 3
    )  # only two illustrative arms and their shared baseline
    output = inputs.source(root, identity, "compute")
    assert len(list(output.export.glob("jobs/*.json"))) == 30
    assert output.record["origin"] == "slurm-wilkes"
    assert not (output.directory / ".baseline-scratch").exists()
    with pytest.raises((PingstoreError, OSError)):
        compute.shard(bank_id, run_id=identity, index=0)


def test_production_retains_all_figure_rows(lab, monkeypatch):
    root, bank_id, calls = lab
    monkeypatch.setenv("PINGLAB_SMOKE", "0")
    identity = compute.compute(bank_id)
    raw = inputs.source(root, identity, "compute")
    assert len(list(raw.export.glob("jobs/*.json"))) == 57
    assert len(calls) == 60
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


def test_transforms_are_deterministic_binary_and_count_preserving_over_full_grid():
    import torch

    generator = torch.Generator().manual_seed(7)
    baseline = (torch.rand(61, 3, 5, generator=generator) < 0.16).float()
    baseline[0, :, :] = 1
    baseline[-1, :, :] = 1
    for name in ("jitter_sigma_0", "cell_jitter_sigma_0"):
        actual, diagnostics = transforms._build_override(
            baseline,
            name,
            torch.Generator().manual_seed(42),
            return_diagnostics=True,
        )
        assert torch.equal(actual, baseline)
        assert diagnostics["input_spikes"] == diagnostics["output_spikes"]
        assert diagnostics["per_trial_cell_count_invariant"] is True
    conditions = [f"jitter_sigma_{sigma:g}" for sigma in recipe.JITTER_SIGMAS_MS]
    conditions += [
        f"cell_jitter_sigma_{sigma:g}" for sigma in recipe.CELL_JITTER_SIGMAS_MS
    ]
    for name in conditions:
        a, diagnostics = transforms._build_override(
            baseline,
            name,
            torch.Generator().manual_seed(42),
            return_diagnostics=True,
        )
        b = transforms._build_override(
            baseline, name, torch.Generator().manual_seed(42)
        )
        assert torch.equal(a, b)
        assert set(a.unique().tolist()) <= {0.0, 1.0}
        assert torch.equal(a.sum(dim=0), baseline.sum(dim=0))
        assert diagnostics["input_spikes"] == diagnostics["output_spikes"]
        assert diagnostics["per_trial_cell_count_invariant"] is True


@pytest.mark.parametrize(
    ("condition", "seed"),
    [("jitter_sigma_100", 0), ("cell_jitter_sigma_100", 1)],
)
def test_temporal_boundaries_reflect_without_losing_spikes(condition, seed):
    import torch

    baseline = torch.zeros(20, 1, 1)
    baseline[0:2, 0, 0] = 1
    actual, diagnostics = transforms._build_override(
        baseline,
        condition,
        torch.Generator().manual_seed(seed),
        dt_ms=1.0,
        return_diagnostics=True,
    )
    assert int(actual.sum()) == 2
    assert torch.equal(actual.sum(dim=0), baseline.sum(dim=0))
    assert diagnostics["boundary_reflected_spikes"] > 0
    assert diagnostics["input_spikes"] == diagnostics["output_spikes"] == 2


def test_collision_resolution_is_binary_nearest_free_and_count_preserving():
    import torch

    candidates = torch.zeros(7, dtype=torch.long)
    batch = torch.zeros(7, dtype=torch.long)
    cells = torch.zeros(7, dtype=torch.long)
    resolved, moved, max_steps = transforms._resolve_collisions(
        candidates, batch, cells, T=7, N_I=1
    )
    assert resolved.tolist() == [0, 1, 2, 3, 4, 5, 6]
    assert int(moved.sum()) == 6
    assert max_steps == 6


def test_integer_reflection_handles_both_edges_and_multiple_bounces():
    import torch

    proposed = torch.tensor([-9, -1, 0, 4, 5, 9, 13])
    reflected = transforms._reflect_into_interval(
        proposed, torch.tensor(0), torch.tensor(4)
    )
    assert reflected.tolist() == [1, 1, 0, 4, 3, 1, 3]


def test_collision_resolution_near_upper_edge_does_not_wrap():
    import torch

    candidates = torch.full((7,), 6, dtype=torch.long)
    batch = torch.zeros(7, dtype=torch.long)
    cells = torch.zeros(7, dtype=torch.long)
    resolved, _moved, _max_steps = transforms._resolve_collisions(
        candidates, batch, cells, T=7, N_I=1
    )
    assert resolved.tolist() == [6, 5, 4, 3, 2, 1, 0]


def test_fixed_window_arm_moves_same_window_population_events_together():
    import torch

    baseline = torch.zeros(600, 1, 2)
    baseline[10:12, 0, :] = 1
    actual = transforms._build_override(
        baseline, "jitter_sigma_1", torch.Generator().manual_seed(3)
    )
    assert torch.equal(actual[:, 0, 0], actual[:, 0, 1])
    assert int(actual[:, 0, 0].sum()) == 2


def test_fixed_window_boundary_reflects_one_shared_displacement():
    import torch

    baseline = torch.zeros(20, 1, 2)
    baseline[0, 0, 0] = 1
    baseline[2, 0, 1] = 1
    actual, diagnostics = transforms._build_override(
        baseline,
        "jitter_sigma_100",
        torch.Generator().manual_seed(0),
        dt_ms=1.0,
        return_diagnostics=True,
    )
    new_t_0 = int(actual[:, 0, 0].nonzero()[0])
    new_t_1 = int(actual[:, 0, 1].nonzero()[0])
    assert new_t_0 - 0 == new_t_1 - 2
    assert diagnostics["boundary_reflected_spikes"] == 2


@pytest.mark.parametrize("condition", ["jitter_sigma_100", "cell_jitter_sigma_50"])
def test_sparse_override_serialization_retains_every_trial_cell_count(
    tmp_path, condition
):
    import torch

    R = {
        "T": np.int32(20),
        "n_i": np.int32(2),
        "n_trials": np.int32(2),
        "i_trial": np.array([0, 0, 0, 1, 1, 1], dtype="int32"),
        "i_t": np.array([0, 1, 19, 0, 18, 19], dtype="int32"),
        "i_cell": np.array([0, 0, 1, 1, 1, 0], dtype="int32"),
    }
    cfg = recipe.configuration()
    simulator = simulation.Simulator(tmp_path, tmp_path / "commands", cfg)
    path = tmp_path / "override.npz"
    diagnostics = simulator._build_override_file(
        R,
        condition,
        torch.Generator().manual_seed(0),
        1.0,
        path,
    )
    source_counts = np.zeros((2, 2), dtype=int)
    np.add.at(source_counts, (R["i_trial"], R["i_cell"]), 1)
    with np.load(path) as data:
        output_counts = np.zeros((2, 2), dtype=int)
        np.add.at(output_counts, (data["i_trial"], data["i_cell"]), 1)
    assert np.array_equal(output_counts, source_counts)
    assert diagnostics["input_spikes"] == diagnostics["output_spikes"] == 6
    assert diagnostics["trials_checked"] == 2
    assert diagnostics["cells_checked_per_trial"] == 2
    assert diagnostics["per_trial_cell_count_invariant"] is True


def test_sparse_override_rejects_duplicate_baseline_events(tmp_path):
    import torch

    R = {
        "T": np.int32(20),
        "n_i": np.int32(2),
        "n_trials": np.int32(1),
        "i_trial": np.array([0, 0], dtype="int32"),
        "i_t": np.array([3, 3], dtype="int32"),
        "i_cell": np.array([1, 1], dtype="int32"),
    }
    simulator = simulation.Simulator(
        tmp_path, tmp_path / "commands", recipe.configuration()
    )
    with pytest.raises(ValueError, match="duplicate events"):
        simulator._build_override_file(
            R,
            "cell_jitter_sigma_50",
            torch.Generator().manual_seed(0),
            1.0,
            tmp_path / "override.npz",
        )


def test_analysis_rejects_missing_or_false_count_invariant():
    cfg = recipe.configuration(smoke=True)
    job = recipe.jobs(cfg)[0]
    metrics = {
        "best_acc": 90.0,
        "n_total": cfg["evaluation_samples"],
        "rates_hz": {"hid": 10.0, "inh": 20.0},
    }
    with pytest.raises(PingstoreError, match="spike-count invariant"):
        analyse.measurement(metrics, job, cfg)
    metrics["override_transform"] = {
        "schema": "exp042.override/v2",
        "boundary_policy": cfg["jitter_policy"]["boundary"],
        "collision_policy": cfg["jitter_policy"]["collision"],
        "input_spikes": 2,
        "output_spikes": 1,
        "boundary_reflected_spikes": 1,
        "collision_resolved_spikes": 0,
        "max_collision_resolution_steps": 0,
        "trials_checked": cfg["evaluation_samples"],
        "cells_checked_per_trial": 2,
        "per_trial_cell_count_invariant": False,
    }
    with pytest.raises(PingstoreError, match="spike-count invariant"):
        analyse.measurement(metrics, job, cfg)


def test_article_renders_fixture_and_unavailable_data_states(lab):
    from demolab_cli import _paths

    root, bank_id, _ = lab
    identity = present.present(analyse.analyse(compute.compute(bank_id)))
    output = inputs.source(root, identity, "present")
    source_root = Path(__file__).resolve().parents[2]
    (root / "writings").mkdir()
    for name in (
        "exp042.typ", "templates/dataset.typ", "templates/abstract.typ",
        "templates/methods.typ", "templates/article-layout.typ",
        "templates/result-card.typ", "templates/contents.typ",
        "templates/equations.typ", "templates/status.typ",
    ):
        target = root / "writings" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_root / "writings" / name, target)
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


def test_ancestor_metadata_amendment_does_not_change_payload_identity(lab):
    root, bank_id, _ = lab
    compute_id = compute.compute(bank_id)
    bank = inputs.source(root, bank_id, "compute", experiment="exp022")
    record = dict(bank.record)
    record["execution"] = {**record["execution"], "note": "fixture manifest change"}
    write_json_atomic(bank.directory / "run.json", record)
    assert analyse.analyse(compute_id).endswith("-analyse")


def test_combined_runner_is_retired():
    result = subprocess.run(
        [sys.executable, "-m", "experiments.exp042", "--skip-training"],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "independent stages" in result.stderr
