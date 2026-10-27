"""Synthetic staged probes; no production simulations or historical imports.

Run with `uv run pytest experiments/exp037/test.py`. Fixtures use temporary stores;
bounded native checks validate numerical behavior without production runs.
"""

import subprocess
from pathlib import Path

import numpy as np
import pytest
from experiments.exp037 import analyse, compute, present, recipe
from experiments.exp037 import analyse as measurements
from experiments.exp037 import recipe as exp022
from experiments.exp037.custom import inputs
from experiments.exp037.recipe import refractory_configuration


def _common_config() -> dict:
    return {
        **refractory_configuration(),
        "model": "ping",
        "dataset": "mnist",
        "max_samples": 7000,
        "epochs": 50,
        "t_ms": 200.0,
        "tau_ampa_ms": 2.0,
        "tau_gaba_ms": 6.0,
        "input_rate": 25.0,
        "input_rate_sampling": "fixed",
        "hidden_sizes": [1024],
        "n_in": 784,
        "n_hidden": 1024,
        "n_inh": 256,
        "n_out": 10,
        "ei_strength": 1.0,
        "ei_ratio": 2.0,
        "w_in": [0.9, 0.09],
        "w_in_initial_zero_fraction": 0.95,
        "recurrent_initial_zero_fraction": 0.0,
        "readout_w_init_mean": 1.12060546875,
        "readout_w_init_std": 0.8349609375,
        "weight_decay": 0.0,
        "grad_clip": 1.0,
        "trainable_w_ei": False,
        "trainable_w_ie": False,
        "trainable_w_ee": False,
        "trainable_w_ii": False,
        "fr_reg_upper_strength": 0.0,
        "fr_reg_upper_target_hz": 0.0,
        "readout_mode": "mem-mean",
        "surrogate_slope": 1.0,
        "lr": 0.0004,
        "batch_size": 256,
        "v_grad_dampen": 1000.0,
        "dales_law": True,
        "dataset_split": {
            "optimizer_train_samples": 6300,
            "validation_samples": 700,
            "official_test_samples": 10000,
            "checkpoint_selection_partition": "validation",
            "official_test_used_during_training": False,
            "source_train_partition": "official_mnist_train",
            "source_test_partition": "official_mnist_test",
            "split_seed": 42,
            "validation_fraction": 0.1,
        },
        "validation_encoder_draws": {
            "count": 3,
            "encoder_seeds": [1, 2, 3],
            "input_rate_seeds": [4, 5, 6],
        },
        "adaptive_threshold": False,
        "train_leak": False,
        "signed_readout": False,
        "readout_bias": False,
        "state_clamp": False,
    }


from pingstore import core
from pingstore.library import (
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
        core, "memberships", lambda _: {"exp022": "demo", "exp037": "demo"}
    )
    monkeypatch.setattr(
        core, "capture_code", lambda *a: {"git_commit": "fixture", "dirty": False}
    )
    monkeypatch.setenv("PINGLAB_SMOKE", "1")
    with core.stage_run(tmp_path, "exp022", "compute") as run:
        for cell in recipe.bank_cells():
            directory = run.export / cell["cell_name"]
            directory.mkdir(parents=True)
            cfg = {
                **_common_config(),
                "dt": 0.1,
                "seed": cell["seed"],
                "n_hidden": 200,
                "n_inh": 64,
                "hidden_sizes": [200],
                "ei_strength": float(cell["model"] == "ping"),
                "ei_ratio": 2.0,
                "w_in": [cell["w_in"], cell["w_in"] * 0.1],
                "v_grad_dampen": 1000.0,
                "fr_reg_upper_strength": 0.0
                if cell["rate_target_hz"] is None
                else 0.041,
                "fr_reg_upper_target_hz": cell["rate_target_hz"] or 0.0,
            }
            write_json_atomic(directory / "config.json", cfg)
            cps = {}
            for role, filename, epoch in (
                ("best_validation", "weights.pth", 43),
                ("final_epoch", "weights_final.pth", 50),
            ):
                target = directory / filename
                target.write_bytes((cell["cell_name"] + role).encode())
                cps[role] = {
                    "filename": filename,
                    "epoch": epoch,
                    "sha256": file_sha256(target),
                }
            write_json_atomic(
                directory / "metrics.json",
                {
                    "config": cfg,
                    "training_cell_name": cell["cell_name"],
                    "best_epoch": 43,
                    "best_acc": 91.0,
                    "checkpoints": cps,
                    "epochs": [
                        {
                            "ep": ep,
                            "acc": 80 + ep / 10,
                            "rate_e": ep / 2,
                            "test_rate_e": ep / 3,
                            "test_rate_i": ep / 4,
                        }
                        for ep in range(1, 51)
                    ],
                },
            )
    monkeypatch.setattr(
        compute,
        "shard_identity",
        lambda cfg, bank, module: {"recipe": cfg, "bank": bank.reference},
    )
    calls = []

    def simulate(*, train, output, attachments, training, request, author, **kwargs):
        train_dir, out = train, output
        calls.append(request)
        out.mkdir(parents=True)
        attachments.mkdir(parents=True, exist_ok=True)
        write_json_atomic(
            attachments / "request.json",
            {**request, "device": compute.resolve_device("auto")},
        )

        def save(path, **arrays):
            np.savez_compressed(path, **arrays)

        train = training
        uniform = request["input"] == "synthetic"
        strength = request.get("ei_strength", training["ei_strength"])
        rate = request["input_rate_hz"]
        cfg = {
            "max_samples": request.get("samples"),
            "perturb_mode": request.get("perturb_mode"),
            "perturb_level": [request.get("perturb_level", 0)],
        }
        if request["input"] == "snapshot":
            cfg["sample_index"] = request["sample_index"]
            e, i = np.zeros((2000, 200), dtype=bool), np.zeros((2000, 64), dtype=bool)
            e[::20, ::2] = True
            i[::30] = True
            save(
                out / "recording.npz",
                dt=0.1,
                n_e=200,
                n_i=64,
                label=7,
                spk_e=e,
                spk_i=i,
            )
        elif uniform:
            cfg["n_batch"] = request["trials"]
            write_json_atomic(
                out / "metrics.json",
                {
                    "config": {
                        **train,
                        "ei_strength": strength,
                        "input_rate_hz": rate,
                        "n_batch": cfg["n_batch"],
                    },
                    "rate_e_hz": rate / 2,
                    "rate_i_hz": rate,
                },
            )
        else:
            n = int(cfg["max_samples"] or 0)
            acc = (
                90 - int(cfg["perturb_level"][0])
                if cfg["perturb_mode"] == "add"
                else 90
            )
            write_json_atomic(
                out / "metrics.json",
                {
                    "config": {
                        **train,
                        "ei_strength": strength,
                        "load_weights": str(train_dir / "weights.pth"),
                        "evaluation_partition": "official_mnist_test",
                        "evaluation_samples": n,
                    },
                    "best_acc": acc,
                    "n_correct": n * acc // 100,
                    "n_total": n,
                    "rates_hz": {"hid": 20.0, "inh": 10.0},
                    "perturbation": {
                        "mode": cfg["perturb_mode"],
                        "level": cfg["perturb_level"][0],
                        "dt_ms": train["dt"],
                        "populations": {
                            key: {
                                "raw_spikes": int(
                                    rate * n * size * train["t_ms"] / 1000
                                ),
                                "transmitted_spikes": int(
                                    rate * n * size * train["t_ms"] / 1000
                                ),
                                "inserted_spikes": 0,
                                "deleted_spikes": 0,
                                "slots": round(train["t_ms"] / train["dt"]) * n * size,
                            }
                            for key, size, rate in (
                                ("e1", train["n_hidden"], 20),
                                ("i1", train["n_inh"], 10),
                            )
                        },
                    },
                },
            )

    monkeypatch.setattr(compute, "run_inference", simulate)
    return tmp_path, run.run_id, calls


def resign(directory):
    path = directory / "run.json"
    record = load_json(path)
    record["payload_digest"] = payload_digest(directory)
    write_json_atomic(path, record)


@pytest.mark.parametrize("mutation", ["sample_count", "snapshot", "missing"])
def test_analyse_rejects_corrupt_even_resigned_payload(lab, mutation):
    root, bank, _ = lab
    cid = compute.compute(bank)
    c = inputs.source(root, cid, "compute")
    cfg = recipe.configuration(smoke=True)
    jobs = recipe.jobs(cfg)
    if mutation == "snapshot":
        p = c.file(
            next(j["path"] for j in jobs if j["kind"] == "raster"),
            "recording.npz",
        )
        with np.load(p) as raw:
            data = {k: raw[k] for k in raw.files}
        data["spk_e"] = np.full_like(data["spk_e"], 2, dtype=np.int8)
        np.savez_compressed(p, **data)
    else:
        job = next(j for j in jobs if j["kind"] == "sweep")
        p = c.file(job["path"], "metrics.json")
        if mutation == "missing":
            p.unlink()
        else:
            d = load_json(p)
            d["n_total"] -= 1
            write_json_atomic(p, d)
    resign(c.directory)
    with pytest.raises((PingstoreError, OSError, ValueError)):
        analyse.analyse(cid)
    assert not list((root / ".pingstore/runs").glob("*-analyse"))


def test_inputs_allow_upstream_metadata_amendment(lab):
    root, bank, _ = lab
    cid = compute.compute(bank)
    p = root / ".pingstore/runs" / bank / "run.json"
    d = load_json(p)
    d["execution"]["extra"] = "changed"
    write_json_atomic(p, d)
    assert analyse.analyse(cid).endswith("-analyse")


def test_failed_simulation_never_completes(lab, monkeypatch):
    root, bank, _ = lab

    def fail(*a, **k):
        raise RuntimeError("fixture failure")

    monkeypatch.setattr(compute, "run_inference", fail)
    with pytest.raises(RuntimeError, match="fixture failure"):
        compute.compute(bank)
    assert not list((root / ".pingstore/runs").glob("exp037-*-compute"))
    assert list((root / ".pingstore/runs").glob(".exp037-*-compute.tmp"))


def test_v2_and_wrong_stage_inputs_are_rejected(lab):
    root, bank_id, _ = lab
    cid = compute.compute(bank_id)
    with pytest.raises(PingstoreError, match="not a"):
        present.present(cid)
    path = root / ".pingstore/runs" / cid / "run.json"
    record = load_json(path)
    record["schema"] = "pingstore.run/v2"
    write_json_atomic(path, record)
    with pytest.raises(PingstoreError):
        analyse.analyse(cid)


def test_source_readme_change_during_compute_is_allowed(lab, monkeypatch):
    root, bank_id, _ = lab
    original = compute.run_inference

    def simulate(*args, **kwargs):
        original(*args, **kwargs)
        path = root / ".pingstore/runs" / bank_id / "README.md"
        path.write_text("changed while running\n")

    monkeypatch.setattr(compute, "run_inference", simulate)
    identity = compute.compute(bank_id)
    assert (root / ".pingstore/runs" / identity).is_dir()


def test_present_rejects_resigned_incomplete_analysis(lab):
    root, bank_id, _ = lab
    cid = compute.compute(bank_id)
    aid = analyse.analyse(cid)
    source = inputs.source(root, aid, "analyse")
    path = source.export / "results.json"
    result = load_json(path)
    result["perturbation"].pop()
    write_json_atomic(path, result)
    resign(source.directory)
    with pytest.raises(PingstoreError, match="incomplete"):
        present.present(aid)
    assert not list((root / ".pingstore/runs").glob("*-present"))


def test_independent_stages_preserve_measurements_and_never_publish(lab, monkeypatch):
    root, bank, calls = lab
    cid = compute.compute(bank)
    assert len(calls) == 60
    source = inputs.source(root, cid, "compute")
    for path in source.export.rglob("recording.npz"):
        with np.load(path) as data:
            assert set(data.files) == set(recipe.SNAPSHOT_ARRAYS)
    monkeypatch.setattr(
        compute, "run_inference", lambda *a, **k: pytest.fail("downstream simulation")
    )
    aid = analyse.analyse(cid)
    analysis = inputs.source(root, aid, "analyse")
    result = load_json(analysis.export / "results.json")
    assert len(result["perturbation"]) == 42
    assert len(result["baseline_results"]) == 36
    assert {r["rate_e"] for r in result["baseline_results"]} == {25.0}
    assert result["plot_data"]["baseline_e_rate_hz"] == {"coba": 20.0, "ping": 20.0}
    assert {r["role"] for r in result["checkpoint_provenance"]} == {"best_validation"}
    for name in (
        "raster",
        "plot_data",
        "baseline_rows",
        "summarize_accuracy",
        "summarize_perturbation_rows",
    ):
        monkeypatch.setattr(
            measurements, name, lambda *a, **k: pytest.fail("presentation measurement")
        )
    pid = present.present(aid)
    output = inputs.source(root, pid, "present")
    assert {f.name for f in output.export.iterdir()} == {
        "numbers.json",
        *recipe.FIGURES,
    }
    assert output.record["inputs"] == {"analysis": analysis.reference}
    assert not (root / ".artifacts").exists()
    assert load_json(output.export / "numbers.json")["illustrative_labels"] == [7] * 12


def test_recipe_preserves_production_and_smoke_grids():
    for smoke, total, sweeps in ((False, 210, 192), (True, 60, 42)):
        jobs = recipe.jobs(recipe.configuration(smoke=smoke))
        assert len(jobs) == total
        assert len({j["id"] for j in jobs}) == total
        assert len({j["path"] for j in jobs}) == total
        assert sum(j["kind"] == "sweep" for j in jobs) == sweeps
        assert {j["seed"] for j in jobs if j["kind"] == "sweep"} == {42, 43, 44}
        rasters = [j for j in jobs if j["kind"] == "raster"]
        assert len(rasters) == 12
        assert {j["seed"] for j in rasters} == {42}
        assert {j["sample_index"] for j in rasters} == {0}
    assert recipe.SHARDS == 6


def test_frontier_registry_and_multiseed_summaries_are_complete(tmp_path):
    expected = {
        cell["name"] for cell in exp022.CELLS if cell["training_run_id"] == "TR-02"
    }
    observed = {
        recipe.cell_name(model, target, seed)
        for model in recipe.MODELS
        for target in recipe.RATE_TARGET_GRID_HZ
        for seed in recipe.seeds_for(target)
    }
    assert observed == expected
    rows = [
        {
            "cell_name": f"ping__off__seed{seed}",
            "model": "ping",
            "rate_target_display": "off",
            "rate_target_hz": None,
            "seed": seed,
            "best_acc": value + 1,
            "final_acc": value,
            "rate_e": value / 10,
        }
        for seed, value in zip((42, 43, 44), (80.0, 82.0, 84.0), strict=True)
    ]
    point = measurements.summarize_frontier(rows)[0]
    assert point["seeds"] == [42, 43, 44]
    assert point["n_seeds"] == 3
    assert point["statistic"] == "mean_across_independent_seeds"
    assert point["uncertainty"] == "sem_across_independent_seeds"
    assert point["final_acc"] == 82.0
    np.testing.assert_allclose(point["final_acc_sem"], 2 / np.sqrt(3))

    summary_rows = [
        {"level": level, "seed": seed, "acc": value}
        for level, values in ((0.0, (80, 82, 84)), (0.5, (40, 50, 60)))
        for seed, value in zip(recipe.SEEDS_BASELINE, values, strict=True)
    ]
    xs, means, sds = measurements.summarize_accuracy(summary_rows, "level")
    np.testing.assert_allclose(xs, [0.0, 0.5])
    np.testing.assert_allclose(means, [82.0, 50.0])
    np.testing.assert_allclose(sds, [2.0, 10.0])

    assert callable(recipe.inference_request)


def test_perturbation_summary_retains_seed_and_sample_provenance():
    rows = [
        {
            "model": "ping",
            "seed": seed,
            "mode": "drop",
            "level": 0.5,
            "acc": acc,
            "e_rate_hz": rate,
            "n_total": 1400,
        }
        for seed, acc, rate in zip(
            recipe.SEEDS_BASELINE, (70, 80, 90), (8, 9, 10), strict=True
        )
    ]
    assert measurements.summarize_perturbation_rows(rows) == [
        {
            "model": "ping",
            "mode": "drop",
            "level": 0.5,
            "acc": 80.0,
            "acc_sd": 10.0,
            "e_rate_hz": 9.0,
            "e_rate_hz_sd": 1.0,
            "seeds": [42, 43, 44],
            "n_total_per_seed": [1400, 1400, 1400],
        }
    ]


def test_writeup_anchor_levels_remain_in_the_recipe():
    assert {0.0, 0.8, 1.0} <= set(recipe.PERTURB_DROP_LEVELS)


def test_raster_selection_preserves_dtype_sum_and_rng(tmp_path):
    e = (np.arange(20 * 256).reshape(20, 256) % 3 == 0).astype(np.float32)
    i = (np.arange(20 * 128).reshape(20, 128) % 5 == 0).astype(np.float32)
    np.savez(
        tmp_path / "recording.npz", spk_e=e[:, None, :], spk_i=i[:, None, :], label=7
    )
    job = {"model": "ping", "seed": 42, "mode": "drop", "level": 0.5}
    result = measurements.raster(tmp_path, {"dt": 0.1, "t_ms": 2.0}, job)
    rng = np.random.default_rng(0)
    ei = np.sort(rng.choice(256, 200, replace=False))
    ii = np.sort(rng.choice(128, 64, replace=False))
    et, en = np.where(e[:, ei].astype(bool))
    it, inn = np.where(i[:, ii].astype(bool))
    for key, expected in (
        ("e_t", et * 0.1),
        ("e_n", en),
        ("i_t", it * 0.1),
        ("i_n", inn + 206),
    ):
        np.testing.assert_array_equal(result[key], expected)
    assert result["e_rate_hz"] == float(e.sum() / (256 * 0.002))


def test_shards_collect_without_reexecuting_and_resume_verified_work(lab, monkeypatch):
    root, bank, calls = lab
    monkeypatch.setattr(
        compute, "capture_code", lambda *a: {"git_commit": "fixture", "dirty": False}
    )
    rid = core.reserve_stage(root / ".pingstore", "exp037", "compute")
    for index in range(6):
        compute.shard(bank, run_id=rid, index=index)
    assert len(calls) == 60
    compute.shard(bank, run_id=rid, index=0)
    assert len(calls) == 60
    compute.compute(bank, run_id=rid, collect=True)
    assert len(calls) == 60
    source = inputs.source(root, rid, "compute")
    execution = source.record["execution"]
    assert len(execution["shards"]) == 6
    assert execution["started_at"] == min(w["started_at"] for w in execution["shards"])
    assert (
        execution["started_at"]
        <= execution["workers_completed_at"]
        <= execution["completed_at"]
    )
    assert all(
        w["host"] and w["device"]["type"] in ("cuda", "cpu")
        for w in execution["shards"]
    )
    assert not (source.directory / ".scratch").exists()
    with pytest.raises(PingstoreError):
        compute.shard(bank, run_id=rid, index=0)


@pytest.mark.parametrize("fault", ["payload", "attachment", "profile"])
def test_shard_resume_rejects_changed_evidence(lab, monkeypatch, fault):
    root, bank, _ = lab
    monkeypatch.setattr(
        compute, "capture_code", lambda *a: {"git_commit": "fixture", "dirty": False}
    )
    rid = core.reserve_stage(root / ".pingstore", "exp037", "compute")
    compute.shard(bank, run_id=rid, index=0)
    directory = root / ".pingstore/runs" / f".{rid}.tmp"
    if fault == "profile":
        monkeypatch.setenv("PINGLAB_SMOKE", "0")
    else:
        path = next(
            (
                directory / ("export" if fault == "payload" else ".scratch/simulations")
            ).rglob("*.json")
        )
        path.write_text("{}")
    with pytest.raises(PingstoreError):
        compute.shard(bank, run_id=rid, index=0)
    assert not (root / ".pingstore/runs" / rid).exists()


def test_collect_rejects_missing_shards_and_busy_reservation(lab, monkeypatch):
    root, bank, _ = lab
    rid = core.reserve_stage(root / ".pingstore", "exp037", "compute")
    with pytest.raises((OSError, PingstoreError)):
        compute.compute(bank, run_id=rid, collect=True)
    directory = root / ".pingstore/runs" / f".{rid}.tmp"
    with (
        compute._compute_lock(directory, exclusive=False),
        pytest.raises(PingstoreError, match="busy"),
    ):
        compute.compute(bank, run_id=rid)


def test_reviewed_figures_keep_coordinates_show_full_range_and_omit_run_ids(
    tmp_path, monkeypatch
):
    from experiments.exp037 import present as plots

    captured = []
    monkeypatch.setattr(plots, "save_figure", lambda fig, *a, **k: captured.append(fig))
    data = {"use_pct": True, "panels": {}}
    for mode in ("drop", "add"):
        data["panels"][mode] = {
            model: {
                "x": [0, 80, 100] if mode == "drop" else [0, 100, 201.426],
                "mean": [90, 89, 10.6],
                "lo": [89, 88, 10],
                "hi": [91, 90, 11.2],
            }
            for model in recipe.MODELS
        }
    plots.plot_perturbation_curves(data, tmp_path / "curves", "exp037-r999-present")
    figure = captured[0]
    assert not figure.texts
    for axis, mode in zip(figure.axes, ("drop", "add"), strict=True):
        for line, model in zip(axis.lines[:2], recipe.MODELS, strict=True):
            np.testing.assert_array_equal(
                line.get_xdata(), data["panels"][mode][model]["x"]
            )
            np.testing.assert_array_equal(
                line.get_ydata(), data["panels"][mode][model]["mean"]
            )
    assert figure.axes[1].get_xlim()[1] > 201.426
    assert "reference E rate" in figure.axes[1].get_xlabel()
    assert "Poisson" not in figure.axes[1].get_title(loc="left")
    assert "probability" in figure.axes[0].get_xlabel()
    assert [text.get_text() for text in figure.axes[1].get_legend().get_texts()] == [
        "COBA",
        "PING",
        "±1 SD band",
    ]


def test_relative_grid_uses_each_seed_calibration_and_complete_shards():
    cfg = recipe.configuration()
    covered = []
    for index in range(6):
        jobs = recipe.shard_jobs(cfg, index)
        assert jobs[0]["kind"] == "calibration"
        assert len({(j["model"], j["seed"]) for j in jobs}) == 1
        baseline = 20.0 + index * 30
        sweep = [j for j in jobs if j["kind"] == "sweep" and j["mode"] == "add"]
        assert [j["level"] for j in sweep] == list(range(0, 201, 10))
        for job in sweep:
            resolved = recipe.resolve_job(job, baseline, 0.1)
            assert resolved["applied_level"] == job["level"] * baseline / 100
            assert resolved["baseline_e_rate_hz"] == baseline
        covered.extend(j["id"] for j in jobs)
    assert len(covered) == len(set(covered)) == 210
    assert set(covered) == {j["id"] for j in recipe.jobs(cfg)}


@pytest.mark.parametrize("baseline", [0, -1, float("nan"), float("inf"), 10000])
def test_unusable_calibration_rejected(baseline):
    job = next(
        j
        for j in recipe.jobs(recipe.configuration())
        if j["mode"] == "add" and j["level"] == 200
    )
    with pytest.raises(ValueError):
        recipe.resolve_job(job, baseline, 0.1)


def test_analysis_groups_by_percentage_not_applied_hz():
    points, baselines = [], []
    for model in recipe.MODELS:
        for seed, baseline in zip((42, 43, 44), (20.0, 100.0, 150.0), strict=True):
            baselines.append({"model": model, "seed": seed, "e_rate_hz": baseline})
            for mode, levels in [("drop", (0.0, 1.0)), ("add", (0.0, 100.0, 200.0))]:
                for level in levels:
                    points.append(
                        {
                            "model": model,
                            "seed": seed,
                            "mode": mode,
                            "level": level,
                            "acc": 90.0 - (seed - 42),
                            "applied_level": baseline * level / 100,
                        }
                    )
    data = measurements.plot_data([], points, baselines=baselines)
    for model in recipe.MODELS:
        assert data["panels"]["add"][model]["x"] == [0.0, 100.0, 200.0]
        assert data["panels"]["add"][model]["mean"] == [89.0] * 3
        assert data["panels"]["add"][model]["lo"] == [88.0] * 3


def test_analysis_rejects_resigned_calibration_drift(lab):
    root, bank, _ = lab
    cid = compute.compute(bank)
    source = inputs.source(root, cid, "compute")
    job = recipe.shard_jobs(recipe.configuration(smoke=True), 0)[0]
    path = source.file(job["path"], "metrics.json")
    record = load_json(path)
    record["rates_hz"]["hid"] *= 2
    record["perturbation"]["populations"]["e1"]["raw_spikes"] *= 2
    record["perturbation"]["populations"]["e1"]["transmitted_spikes"] *= 2
    write_json_atomic(path, record)
    resign(source.directory)
    with pytest.raises(PingstoreError, match="evidence differs"):
        analyse.analyse(cid)


def test_hpc_review_is_read_only_and_dependencies_require_live(lab, monkeypatch):
    from argparse import Namespace

    from experiments.exp037.custom import hpc

    root, bank, _ = lab
    monkeypatch.setattr(hpc, "REPO", root)
    monkeypatch.setattr(hpc, "check_plan", lambda _: None)
    source = inputs.source(root, bank, "compute", experiment="exp022")
    identities = {
        stage: core.reserve_stage(root / ".pingstore", "exp037", stage)
        for stage in hpc.STAGES
    }
    for identity in identities.values():
        path = root / ".pingstore/runs" / f".{identity}.tmp/.reservation.json"
        record = load_json(path)
        record["inputs"] = {"bank": source.reference}
        write_json_atomic(path, record)
    plan = {
        "runs": identities,
        "bank": source.reference,
        "account": "test-account",
        "cpu_account": "test-cpu-account",
        "partition": "ampere",
        "cpu_partition": "icelake",
        "walltime": "02:00:00",
        "cpus": 4,
        "memory_gb": 32,
    }
    path = root / "plan.json"
    write_json_atomic(path, plan)
    monkeypatch.setattr(
        hpc.subprocess,
        "run",
        lambda *a, **k: pytest.fail("dry run contacted scheduler"),
    )
    hpc.review(Namespace(plan=path, live=False, test_only=False))
    assert not path.with_suffix(".submitted.json").exists()
    assert not (root / "logs").exists()
    commands = []

    def submit(cmd, **kwargs):
        commands.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, str(100 + len(commands)), "")

    monkeypatch.setattr(hpc.subprocess, "run", submit)
    hpc.review(Namespace(plan=path, live=True, test_only=False))
    assert "--array=0-5%6" in commands[0]
    assert "--gres=gpu:1" in commands[0]
    for idx, cmd in enumerate(commands[1:], 101):
        assert f"--dependency=afterok:{idx}" in cmd
        assert "--gres=gpu:1" not in cmd
    with pytest.raises(PingstoreError, match="already attempted"):
        hpc.review(Namespace(plan=path, live=True, test_only=False))


def test_shard_identity_tracks_local_inference_and_nested_source(tmp_path):
    from types import SimpleNamespace

    from experiments.exp037.custom import sharding

    recipe_path = tmp_path / "recipe.py"
    recipe_path.write_text("SEED = 42\n")
    implementation = tmp_path / "custom" / "inference.py"
    implementation.parent.mkdir()
    implementation.write_text("VERSION = 1\n")
    bank = SimpleNamespace(reference={"run_id": "fixture", "payload_digest": "fixture"})
    recipe_module = SimpleNamespace(__file__=str(recipe_path))
    dataset = (np.zeros((2, 4), dtype=np.float32), np.array([0, 1]))
    before = sharding.shard_identity({}, bank, recipe_module, dataset=dataset)
    assert before["inference_source"] == file_sha256(Path(compute.__file__))
    assert before["experiment_source"][str(implementation)] == file_sha256(
        implementation
    )

    implementation.write_text("VERSION = 2\n")
    after = sharding.shard_identity({}, bank, recipe_module, dataset=dataset)
    assert after["experiment_source"] != before["experiment_source"]
    assert after["dataset"] == before["dataset"]
    assert after["bank"] == before["bank"]
