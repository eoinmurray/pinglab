from __future__ import annotations

from pathlib import Path

import numpy as np
from experiments import exp022, exp082
from experiments.collections.gamma_gated_sparsity.plan import build_plan
from experiments.exp022 import compute as exp022_compute

"""Synthetic contract tests; never use the workspace store or real inference."""

from types import SimpleNamespace

import pytest
import torch
from experiments.collections.gamma_gated_sparsity import execution, plan, workloads
from experiments.exp082 import (
    analyse,
    collection,
    compute,
    evidence,
    inference,
    inputs,
    measurements,
    present,
    recipe,
)
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
        stages, "memberships", lambda _: {"exp022": "demo", "exp082": "demo"}
    )

    def code(*args):
        return {"git_commit": "fixture", "dirty": False, "code_dirty": False}

    monkeypatch.setattr(stages, "_capture_code", code)
    monkeypatch.setattr(compute, "_capture_code", code)
    monkeypatch.setenv("PINGLAB_SMOKE", "1")
    for key in ("STREAMS_PER_CELL", "DIGITS_PER_STREAM", "STREAM_BATCH_SIZE"):
        monkeypatch.delenv("PINGLAB_EXP082_" + key, raising=False)
    with stages.stage_run(tmp_path, "exp022", "compute") as run:
        for seed in recipe.SEEDS:
            name = recipe.training_cell_name(seed)
            folder = run.export / name
            folder.mkdir(parents=True)
            cfg = {
                "model": "ping",
                "dataset": "mnist",
                "dt": 0.1,
                "t_ms": 200.0,
                "n_in": 784,
                "n_hidden": 1024,
                "n_inh": 256,
                "n_out": 10,
                "epochs": 50,
                "max_samples": 7000,
                "seed": seed,
                "readout_mode": "spike-count",
                "input_rates": list(recipe.TRAINING_RATES_HZ),
                "dataset_split": {
                    "checkpoint_selection_partition": "validation",
                    "official_test_used_during_training": False,
                },
            }
            write_json_atomic(folder / "config.json", cfg)
            cps = {}
            for role, filename, epoch in (
                ("best_validation", "weights.pth", 43),
                ("final_epoch", "weights_final.pth", 50),
            ):
                (folder / filename).write_bytes((name + role).encode())
                cps[role] = {
                    "filename": filename,
                    "epoch": epoch,
                    "sha256": file_sha256(folder / filename),
                }
            write_json_atomic(
                folder / "metrics.json",
                {
                    "config": cfg,
                    "training_cell_name": name,
                    "best_epoch": 43,
                    "checkpoints": cps,
                    "epochs": [{"ep": i} for i in range(1, 51)],
                },
            )
    calls = []

    class Worker:
        dataset = {"fixture": True}

        def __init__(self, bank, directory, cfg):
            self.directory, self.cfg = directory, cfg

        def condition(self, job):
            calls.append(job["id"])
            folder = self.directory / "export" / job["path"]
            folder.mkdir(parents=True)
            shape = (self.cfg["streams_per_cell"], self.cfg["digits_per_stream"])
            out = np.zeros((*shape, 10), dtype=np.int64)
            out[..., 0] = 1
            np.savez_compressed(
                folder / "counts.npz",
                out_counts=out,
                e_counts=np.ones(shape, dtype=np.int64),
                i_counts=np.ones(shape, dtype=np.int64),
                labels=np.zeros(shape, dtype=np.int64),
            )
            attachment = self.directory / ".scratch/simulations" / job["path"]
            attachment.mkdir(parents=True)
            write_json_atomic(attachment / "command.json", {"fixture": True})

        def stream(self, name):
            calls.append(name)
            conditions = (
                [[200.0, 5.0]] * 5
                if name == "matched"
                else [list(c) for c in recipe.VARIABLE_STREAM]
            )
            bounds = np.cumsum([0, *[int(d / 0.1) for d, _ in conditions]]).tolist()
            folder = self.directory / "export/streams" / name
            folder.mkdir(parents=True)
            out = np.zeros((bounds[-1], 10), dtype=np.int8)
            out[bounds[:-1], 0] = 1
            np.savez_compressed(
                folder / "recording.npz",
                pixels=np.zeros((5, 784), dtype=np.float32),
                spikes_e=np.zeros((bounds[-1], 1024), dtype=np.int8),
                spikes_i=np.zeros((bounds[-1], 256), dtype=np.int8),
                spikes_out=out,
            )
            write_json_atomic(
                folder / "stream.json",
                {"labels": [0] * 5, "boundaries": bounds, "conditions": conditions},
            )

    monkeypatch.setattr(compute, "Inference", Worker)
    return tmp_path, run.run_id, calls


def resign(folder):
    record = load_json(folder / "run.json")
    record["payload_digest"] = payload_digest(folder)
    write_json_atomic(folder / "run.json", record)


def test_retained_aggregate_rows_are_still_validated(tmp_path):
    cfg = recipe.configuration()
    job = recipe.jobs(cfg)[0]
    row = {k: job[k] for k in ("seed", "duration_ms", "rate_hz")}
    row.update(
        stream_batch_size=cfg["stream_batch_size"],
        n_correct=cfg["digits_per_seed_cell"],
        n_total=cfg["digits_per_seed_cell"],
        accuracy=1.0,
        output_spikes_per_presentation=1.0,
        silent_fraction=0.0,
        class_spike_totals=[cfg["digits_per_seed_cell"], *([0] * 9)],
        rate_e_hz=1.0,
        rate_i_hz=1.0,
    )
    path = tmp_path / "condition.json"
    write_json_atomic(path, row)
    assert evidence.aggregate(path, job, cfg) == row
    row["accuracy"] = 0.5
    write_json_atomic(path, row)
    with pytest.raises(PingstoreError, match="condition totals"):
        evidence.aggregate(path, job, cfg)


def test_compact_rate_ticks_do_not_overlap(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    cfg = recipe.configuration()
    rows = [{**j, "accuracy": 0.5} for j in recipe.jobs(cfg)]
    saved = []
    monkeypatch.setattr(plt, "close", lambda fig: saved.append(fig))
    present.plots.plot_duration_rate_summary(
        measurements.plot_data(rows, cfg), tmp_path / "grid.png", "review"
    )
    fig = saved[-1]
    fig.canvas.draw()
    curve = next(a for a in fig.axes if a.get_ylabel() == "accuracy at 200 ms")
    labels = curve.get_xticklabels()
    assert [t.get_text() for t in labels] == [
        f"{r:g}" for r in cfg["psychometric_rates_hz"]
    ]
    boxes = [t.get_window_extent() for t in labels]
    assert all(a.x1 < b.x0 for a, b in zip(boxes, boxes[1:]))


def fake_plots(monkeypatch):
    for name in (
        "plot_single_trial",
        "plot_single_trial_transition",
        "plot_stream",
        "plot_variable_headline",
        "plot_psychometric",
        "plot_duration_rate_summary",
    ):
        monkeypatch.setattr(
            present.plots, name, lambda data, path, rid: path.write_text("figure")
        )
    monkeypatch.setattr(
        present.plots, "plot_design", lambda path: path.write_text("diagram")
    )


def test_independent_stages_and_flat_export(lab, monkeypatch):
    repo, bank, calls = lab
    cid = compute.compute(bank)
    assert len(calls) == 20
    assert not (repo / ".artifacts").exists()
    monkeypatch.setattr(
        compute, "Inference", lambda *a: pytest.fail("downstream launched inference")
    )
    aid = analyse.analyse(cid)
    fake_plots(monkeypatch)
    pid = present.present(aid)
    source = inputs.source(repo, pid, "present")
    assert all((source.export / f).is_file() for f in recipe.FIGURES)
    assert all(p.is_file() for p in source.export.iterdir())
    assert set(inputs.lineage(repo, pid)) == {bank, cid, aid, pid}
    assert (
        load_json(source.export / "numbers.json")["grid_per_seed"][0]["accuracy"] == 1.0
    )
    assert all(
        name.endswith(stage)
        for name, stage in ((cid, "compute"), (aid, "analyse"), (pid, "present"))
    )


@pytest.mark.parametrize("stage", ["compute", "analyse", "present"])
def test_missing_source_never_allocates(lab, stage):
    repo, _, _ = lab
    before = set((repo / ".pingstore/runs").iterdir())
    with pytest.raises((PingstoreError, OSError)):
        getattr(
            {"compute": compute, "analyse": analyse, "present": present}[stage], stage
        )("exp082-r999-compute")
    assert set((repo / ".pingstore/runs").iterdir()) == before


@pytest.mark.parametrize("target", ["payload", "v2", "root", "symlink"])
def test_source_corruption_rejected(lab, target):
    repo, bank, _ = lab
    cid = compute.compute(bank)
    folder = repo / ".pingstore/runs" / bank
    if target == "payload":
        (folder / "export/changed").write_text("corruption")
    elif target == "manifest":
        record = load_json(folder / "run.json")
        record["extra"] = 1
        write_json_atomic(folder / "run.json", record)
    elif target == "v2":
        record = load_json(folder / "run.json")
        record["schema"] = "pingstore.run/v2"
        write_json_atomic(folder / "run.json", record)
    elif target == "root":
        (folder / "unexpected").write_text("invalid root")
        resign(folder)
    else:
        (folder / "export/link").symlink_to(folder / "export/cells")
    with pytest.raises(PingstoreError):
        analyse.analyse(cid)


def test_compute_failure_stays_hidden(lab, monkeypatch):
    repo, bank, _ = lab
    monkeypatch.setattr(
        compute.Inference,
        "condition",
        lambda *a: (_ for _ in ()).throw(RuntimeError("failed simulator")),
    )
    with pytest.raises(RuntimeError, match="failed simulator"):
        compute.compute(bank)
    assert not list((repo / ".pingstore/runs").glob("exp082-*"))
    assert len(list((repo / ".pingstore/runs").glob(".exp082-*.tmp"))) == 1


def test_input_change_during_compute_stays_hidden(lab, monkeypatch):
    repo, bank, _ = lab
    original = compute.Inference.stream

    def mutate(self, name):
        original(self, name)
        (repo / ".pingstore/runs" / bank / "export/changed").write_text("changed")

    monkeypatch.setattr(compute.Inference, "stream", mutate)
    with pytest.raises(PingstoreError):
        compute.compute(bank)
    assert not list((repo / ".pingstore/runs").glob("exp082-*"))


def test_six_shards_reuse_collect_and_no_reexecution(lab):
    repo, bank, calls = lab
    identity = stages.reserve_stage(repo / ".pingstore", "exp082", "compute")
    with pytest.raises((OSError, PingstoreError)):
        compute.compute(bank, run_id=identity, collect=True)
    for index in range(6):
        compute.shard(bank, run_id=identity, index=index)
    assert len(calls) == 18
    compute.shard(bank, run_id=identity, index=0)
    assert len(calls) == 18
    assert compute.compute(bank, run_id=identity, collect=True) == identity
    assert calls[-2:] == ["matched", "variable"]


def test_shard_rejects_changed_payload(lab):
    repo, bank, _ = lab
    identity = stages.reserve_stage(repo / ".pingstore", "exp082", "compute")
    compute.shard(bank, run_id=identity, index=0)
    folder = repo / ".pingstore/runs" / f".{identity}.tmp"
    next((folder / "export/jobs").glob("*/counts.npz")).write_bytes(b"broken")
    with pytest.raises(PingstoreError, match="payload"):
        compute.shard(bank, run_id=identity, index=0)


def test_collection_adapter_and_workload():
    cfg = recipe.configuration()
    assert len(recipe.jobs(cfg)) == 132
    row = next(
        r
        for s in plan.build_plan(Path("/tmp/exp082-campaign"), "fixture")["stages"]
        for r in s["experiments"]
        if r["slug"] == "exp082"
    )
    assert row["execution"]["mode"] == "exp082-staged"
    assert execution._stage_adapter("exp082") is collection
    assert row["execution"]["workload_contract"]["classified_presentations"] == 26400
    with pytest.raises(ValueError, match="explicit v3 bank"):
        workloads.execute_shard("exp082", 0, 6, smoke=False)


def test_batches_preserve_time_axis_and_partial_batch(tmp_path, monkeypatch):
    monkeypatch.setattr(
        inference,
        "load_mnist_split",
        lambda **k: (None, np.ones((20, 784)), None, np.arange(20) % 10),
    )
    monkeypatch.setattr(inference, "encode_stream", lambda *a: torch.ones((4, 1, 784)))
    cfg = recipe.configuration(streams=5, digits=2, batch=3)
    worker = inference.Inference(SimpleNamespace(export=tmp_path), tmp_path, cfg)
    seen = []

    def simulate(train, spikes, resets, attachments, kind):
        seen.append(tuple(spikes.shape))
        assert resets == (0, 20)
        shape = (spikes.shape[1], 2)
        return {
            "out_counts": np.ones((*shape, 10), dtype=np.int64),
            "e_counts": np.ones(shape, dtype=np.int64),
            "i_counts": np.ones(shape, dtype=np.int64),
        }

    monkeypatch.setattr(worker, "simulate", simulate)
    job = {
        "id": "fixture",
        "path": "jobs/fixture",
        "seed": 42,
        "duration_ms": 2.0,
        "rate_hz": 5.0,
        "cell_name": recipe.training_cell_name(42),
    }
    worker.condition(job)
    assert seen == [(4, 3, 784), (4, 2, 784)]
    counts = evidence.counts(tmp_path / "export/jobs/fixture/counts.npz", cfg)
    assert measurements.condition_row(job, counts, cfg)["n_total"] == 10


def test_analysis_sem_preserves_three_seed_estimator():
    cfg = recipe.configuration()
    rows = [{**j, "accuracy": (j["seed"] - 42) / 2} for j in recipe.jobs(cfg)]
    data = measurements.plot_data(rows, cfg)
    np.testing.assert_allclose(data["means"], 0.5)
    np.testing.assert_allclose(data["sems"], 0.5 / np.sqrt(3))


def test_missing_pixels_is_an_error_not_dataset_fallback(lab):
    repo, bank, _ = lab
    cid = compute.compute(bank)
    root = repo / ".pingstore/runs" / cid
    path = root / "export/streams--matched/recording.npz"
    data = evidence.arrays(path)
    del data["pixels"]
    np.savez_compressed(path, **data)
    resign(root)
    with pytest.raises(PingstoreError, match="explicit pixels"):
        analyse.analyse(cid)


def test_plot_failure_preserves_sources_and_stays_hidden(lab, monkeypatch):
    repo, bank, _ = lab
    cid = compute.compute(bank)
    aid = analyse.analyse(cid)
    before = {
        p.name: file_sha256(p / "run.json")
        for p in (repo / ".pingstore/runs").iterdir()
    }
    monkeypatch.setattr(
        present.plots,
        "plot_single_trial",
        lambda *a: (_ for _ in ()).throw(RuntimeError("plot failure")),
    )
    with pytest.raises(RuntimeError, match="plot failure"):
        present.present(aid)
    assert {
        p.name: file_sha256(p / "run.json")
        for p in (repo / ".pingstore/runs").iterdir()
        if not p.name.startswith(".")
    } == before
    assert len(list((repo / ".pingstore/runs").glob(".exp082-*-present.tmp"))) == 1


def test_all_saved_figures_render_without_inference(lab, monkeypatch):
    repo, bank, _ = lab
    cid = compute.compute(bank)
    aid = analyse.analyse(cid)
    monkeypatch.setattr(
        inference,
        "load_mnist_split",
        lambda **k: pytest.fail("presentation fetched dataset"),
    )
    pid = present.present(aid)
    output = inputs.source(repo, pid, "present").export
    assert all((output / name).stat().st_size > 100 for name in recipe.FIGURES)


def test_compute_lock_excludes_collector(lab):
    repo, bank, _ = lab
    identity = stages.reserve_stage(repo / ".pingstore", "exp082", "compute")
    directory = repo / ".pingstore/runs" / f".{identity}.tmp"
    with compute._compute_lock(directory, exclusive=False):
        with pytest.raises(PingstoreError, match="busy"):
            compute.compute(bank, run_id=identity, collect=True)


def test_dirty_shards_fail_without_scientific_work(lab, monkeypatch):
    repo, bank, calls = lab
    identity = stages.reserve_stage(repo / ".pingstore", "exp082", "compute")
    monkeypatch.setattr(compute, "_capture_code", lambda *a: {"code_dirty": True})
    with pytest.raises(PingstoreError, match="committed"):
        compute.shard(bank, run_id=identity, index=0)
    assert calls == []


def test_collection_dispatches_explicit_stage_sources(lab, monkeypatch):
    repo, bank, calls = lab
    fake_plots(monkeypatch)
    campaign = plan.build_plan(repo / "campaign", "fixture", smoke=True)
    campaign["profile"] = "smoke"
    campaign["exp022_manifest"] = str(repo / "campaign/exp022/campaign.json")
    write_json_atomic(Path(campaign["exp022_manifest"]), {"pingstore_run_id": bank})
    row = next(
        r for s in campaign["stages"] for r in s["experiments"] if r["slug"] == "exp082"
    )
    commands = []

    def dispatch(command, **kwargs):
        commands.append(command)
        stage = command[2].rsplit(".", 1)[1]
        source = command[command.index("--source") + 1]
        identity = command[command.index("--run-id") + 1]
        getattr(
            {"compute": compute, "analyse": analyse, "present": present}[stage], stage
        )(source, run_id=identity)
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(collection.subprocess, "run", dispatch)
    references = collection.execute(repo, campaign, row)
    assert [c[2] for c in commands] == [
        "experiments.exp082." + s for s in collection.STAGES
    ]
    assert references["bank"]["run_id"] == bank
    assert set(references) == {"bank", "compute", "analyse", "present"}
    collection.execute(repo, campaign, row)
    assert len(commands) == 3
    assert len(calls) == 20


def test_simulator_serializes_batched_input_and_resets(tmp_path, monkeypatch):
    monkeypatch.setattr(
        inference,
        "load_mnist_split",
        lambda **k: (None, np.zeros((10, 784)), None, np.arange(10)),
    )
    worker = inference.Inference(
        SimpleNamespace(export=tmp_path), tmp_path, recipe.configuration()
    )

    def simulate(command, **kwargs):
        assert Path(command[1]).is_file()
        assert Path(command[1]).parts[-2:] == ("snnsim", "tool.py")
        assert Path(command[command.index("--load-weights") + 1]).name == "weights.pth"
        assert command[command.index("--device") + 1] == "auto"
        raw = evidence.arrays(Path(command[command.index("--input-file") + 1]))
        assert raw["input_spikes"].shape == (4, 3, 784)
        assert raw["readout_reset"].tolist() == [True, False, True, False]
        out = Path(command[command.index("--out-dir") + 1])
        out.mkdir()
        np.savez_compressed(
            out / "spike_summary.npz",
            dt=np.float32(0.1),
            T=4,
            n_trials=3,
            segment_starts=[0, 2],
            segment_stops=[2, 4],
            out_counts=np.zeros((3, 2, 10), dtype=np.int64),
            e_counts=np.zeros((3, 2), dtype=np.int64),
            i_counts=np.zeros((3, 2), dtype=np.int64),
        )

    monkeypatch.setattr(inference.subprocess, "run", simulate)
    raw = worker.simulate(
        tmp_path,
        torch.ones((4, 3, 784)),
        (0, 2),
        tmp_path / "attachments",
        "spike_summary",
    )
    assert raw["out_counts"].shape == (3, 2, 10)


def test_exp022_planned_bank_targets_exp082() -> None:
    cells = exp022.PLANNED_VARIABLE_RATE_CELLS
    assert [cell["name"] for cell in cells] == [
        "ping__variable_rate__seed42",
        "ping__variable_rate__seed43",
        "ping__variable_rate__seed44",
    ]
    assert all(cell["consumer"] == "exp082" for cell in cells)
    assert all(cell["readout"] == "spike-count" for cell in cells)
    assert all(cell["status"] == "ready_to_train" for cell in cells)
    assert all(cell["training_run_id"] == "TR-06" for cell in cells)
    assert all(
        tuple(cell["input_rates_hz"]) == exp082.TRAINING_RATES_HZ for cell in cells
    )
    assert all(cell in exp022.CANONICAL_CELLS for cell in cells)


def test_exp022_variable_rate_args() -> None:
    cell = exp022.PLANNED_VARIABLE_RATE_CELLS[0]
    args = exp022.build_train_args(cell, exp082.training_dir(42), 7000, 50)
    assert args[args.index("--readout") + 1] == "spike-count"
    assert args[args.index("--readout-w-init-mean") + 1] == "0.05"
    assert args[args.index("--readout-w-init-std") + 1] == "0.04"
    start = args.index("--input-rates") + 1
    stop = start + len(exp082.TRAINING_RATES_HZ)
    assert tuple(map(float, args[start:stop])) == exp082.TRAINING_RATES_HZ


def test_exp022_wilkes_resource_tiers_partition_registry() -> None:
    expected = {
        "standard": 90,
        "fine_dt": 3,
        "canonical_coba": 3,
        "canonical_ping": 3,
        "variable_rate": 3,
    }
    tiered_names = []
    for tier, count in expected.items():
        cells = exp022.cells_in_resource_tier(tier)
        assert len(cells) == count
        assert all(exp022.cell_resource_tier(cell) == tier for cell in cells)
        tiered_names.extend(cell["name"] for cell in cells)

    assert sorted(tiered_names) == sorted(
        cell["name"] for cell in exp022.CANONICAL_CELLS
    )


def test_training_run_ids_match_documented_families() -> None:
    by_family = {
        family: {
            cell["training_run_id"]
            for cell in exp022.CANONICAL_CELLS
            if cell["family"] == family
        }
        for family in exp022.TRAINING_RUN_IDS
    }
    assert by_family == {
        family: {run_id} for family, run_id in exp022.TRAINING_RUN_IDS.items()
    }


def test_completed_cell_artifacts_are_stamped_with_training_run_id(
    tmp_path,
    monkeypatch,
) -> None:
    cell = exp022.PLANNED_VARIABLE_RATE_CELLS[0]
    directory = tmp_path / cell["name"]
    directory.mkdir()
    (directory / "config.json").write_text('{"mode": "train"}\n')
    (directory / "metrics.json").write_text('{"config": {"epochs": 50}}\n')
    monkeypatch.setitem(
        exp022_compute._stamp_training_run_identity.__globals__,
        "cell_dir",
        lambda _name: directory,
    )

    exp022_compute._stamp_training_run_identity(cell)

    config = exp022_compute.json.loads((directory / "config.json").read_text())
    metrics = exp022_compute.json.loads((directory / "metrics.json").read_text())
    assert config["training_run_id"] == "TR-06"
    assert config["training_cell_name"] == cell["name"]
    assert metrics["training_run_id"] == "TR-06"
    assert metrics["config"]["training_run_id"] == "TR-06"


def test_spike_count_logits_use_only_matched_window() -> None:
    spikes = np.asarray(
        [
            [1, 0],
            [0, 1],
            [1, 1],
            [9, 9],
        ],
        dtype=np.float32,
    )
    logits = exp082.spike_count_logits(spikes, start=1, stop=3)
    np.testing.assert_array_equal(logits, np.asarray([1, 2], dtype=np.float32))


def test_psychometric_is_fixed_at_training_duration() -> None:
    assert exp082.MATCHED_DURATION_MS == 200.0
    assert exp082.PSYCHOMETRIC_RATES_HZ == exp082.TRAINING_RATES_HZ


def test_exp082_evaluation_scale_is_recorded() -> None:
    assert exp082.STREAMS_PER_CELL >= 1
    assert exp082.DIGITS_PER_STREAM >= 1
    assert exp082.STREAM_BATCH_SIZE >= 1
    assert exp082.EVALUATION_PROFILE in {"smoke", "pilot", "production"}
    if exp082.EVALUATION_PROFILE == "production":
        assert exp082.DIGITS_PER_STREAM == 5
        assert exp082.STREAMS_PER_CELL == 40
        assert exp082.STREAMS_PER_CELL * exp082.DIGITS_PER_STREAM == 200


def test_exp082_condition_jobs_cover_the_registered_grid() -> None:
    jobs = exp082.infer_jobs()
    assert len(jobs) == (
        len(exp082.SEEDS) * len(exp082.DURATIONS_MS) * len(exp082.PSYCHOMETRIC_RATES_HZ)
    )
    assert len(jobs) == len(set(jobs))
    assert {exp082.parse_condition_job_id(job_id) for job_id in jobs} == {
        (seed, duration, rate)
        for duration in exp082.DURATIONS_MS
        for rate in exp082.PSYCHOMETRIC_RATES_HZ
        for seed in exp082.SEEDS
    }


def test_output_activity_summary_uses_presentation_boundaries() -> None:
    spikes = np.zeros((5, 3), dtype=np.int8)
    spikes[0, 0] = 1
    spikes[3, 2] = 1
    summary = exp082.output_activity_summary(spikes, [0, 2, 5])
    assert summary == {
        "n_presentations": 2,
        "total_output_spikes": 2,
        "spikes_per_presentation": [1, 1],
        "silent_presentations": 0,
        "silent_fraction": 0.0,
        "class_spike_totals": [1, 0, 1],
    }


def test_single_trial_is_first_fully_reset_stream_presentation() -> None:
    stream = {
        "boundaries": [0, 2, 5],
        "conditions": [[0.2, 5.0], [0.3, 10.0]],
        "pixels": np.arange(2 * 784, dtype=np.float32).reshape(2, 784),
        "labels": [4, 7],
        "predictions": [4, 6],
        "correct": [1, 0],
        "spikes_e": np.arange(20).reshape(5, 4),
        "spikes_i": np.arange(10).reshape(5, 2),
        "spikes_out": np.eye(10, dtype=np.int8)[[4, 4, 6, 6, 7]],
        "probabilities": np.full((5, 10), 0.1, dtype=np.float32),
    }
    trial = exp082.single_trial_from_stream(stream)
    assert trial["boundaries"] == [0, 2]
    assert trial["conditions"] == [[0.2, 5.0]]
    assert trial["labels"] == [4]
    assert trial["predictions"] == [4]
    np.testing.assert_array_equal(trial["pixels"], stream["pixels"][:1])
    np.testing.assert_array_equal(trial["spikes_out"], stream["spikes_out"][:2])
    assert trial["output_activity"]["class_spike_totals"][4] == 2


def test_explanatory_trial_is_first_correct_presentation() -> None:
    stream = {
        "boundaries": [0, 1, 2],
        "conditions": [[0.1, 5.0], [0.1, 5.0]],
        "pixels": np.zeros((2, 784), dtype=np.float32),
        "labels": [8, 4],
        "predictions": [5, 4],
        "correct": [0, 1],
        "spikes_e": np.zeros((2, 4), dtype=np.int8),
        "spikes_i": np.zeros((2, 2), dtype=np.int8),
        "spikes_out": np.eye(10, dtype=np.int8)[[5, 4]],
        "probabilities": np.full((2, 10), 0.1, dtype=np.float32),
    }

    trial = exp082.first_correct_trial_from_stream(stream)

    assert trial["labels"] == [4]
    assert trial["predictions"] == [4]
    assert trial["correct"] == [1]


def test_grid_preflight_rejects_wholly_silent_readout() -> None:
    rows = [
        {"n_total": 5, "silent_fraction": 1.0, "output_spikes_per_presentation": 0.0},
    ]
    with np.testing.assert_raises_regex(RuntimeError, "output readout is silent"):
        exp082.grid_output_preflight(rows)


def test_collection_requires_exp082_measurements_and_figures(tmp_path) -> None:
    plan = build_plan(tmp_path / "campaign", "exp082-contract")
    row = next(
        row
        for stage in plan["stages"]
        for row in stage["experiments"]
        if row["slug"] == "exp082"
    )
    assert row["execution"]["mode"] == "exp082-staged"
    assert [Path(path).name for path in row["required_outputs"]] == ["stage-refs.json"]
