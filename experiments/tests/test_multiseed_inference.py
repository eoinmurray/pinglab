import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from experiments import exp022, exp037, exp038
from experiments.helpers.checkpoints import sha256_file

REPO = Path(__file__).resolve().parents[2]


def test_frontier_consumers_cover_the_complete_tr02_registry() -> None:
    expected = {
        cell["name"]
        for cell in exp022.CANONICAL_CELLS
        if cell["training_run_id"] == "TR-02"
    }
    for module in (exp037, exp038):
        observed = {
            module.cell_name(model, target, seed)
            for model in module.MODELS
            for target in module.RATE_TARGET_GRID_HZ
            for seed in module.seeds_for(target)
        }
        assert observed == expected
        assert len(observed) == 36


def test_frontier_summary_reports_mean_and_sem_across_seeds() -> None:
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
        for seed, value in zip((42, 43, 44), (80.0, 82.0, 84.0))
    ]
    summary = exp037.summarize_frontier(rows)
    assert len(summary) == 1
    point = summary[0]
    assert point["seeds"] == [42, 43, 44]
    assert point["n_seeds"] == 3
    assert point["statistic"] == "mean_across_independent_seeds"
    assert point["uncertainty"] == "sem_across_independent_seeds"
    assert point["final_acc"] == 82.0
    np.testing.assert_allclose(point["final_acc_sem"], 2 / np.sqrt(3))


def _write_selected_checkpoint(train_dir: Path) -> None:
    path = train_dir / "weights.pth"
    path.write_bytes(b"checkpoint")
    (train_dir / "metrics.json").write_text(json.dumps({
        "best_epoch": 7,
        "config": {"epochs": 50},
        "checkpoints": {
            "best_validation": {
                "filename": "weights.pth",
                "epoch": 7,
                "sha256": sha256_file(path),
            }
        },
    }))


def test_exp037_jobs_cover_every_quantitative_seed() -> None:
    jobs = exp037.infer_jobs()
    quantitative = [job for job in jobs if job.startswith("sweep__")]
    raster = [job for job in jobs if job.startswith("raster__")]
    expected_quantitative = (
        len(exp037.MODELS)
        * len(exp037.SEEDS_BASELINE)
        * (len(exp037.PERTURB_DROP_LEVELS) + len(exp037.PERTURB_ADD_LEVELS))
    )
    expected_raster = len(exp037.MODELS) * (
        len(exp037.PERTURB_RASTER_DROP_LEVELS)
        + len(exp037.PERTURB_RASTER_ADD_LEVELS)
    )
    assert len(quantitative) == expected_quantitative
    assert len(raster) == expected_raster
    assert {exp037._parse_job(job)[2] for job in quantitative} == {42, 43, 44}
    assert {exp037._parse_job(job)[2] for job in raster} == {42}


def test_exp037_raster_jobs_are_model_seed_mode_and_level_specific():
    from experiments.exp037 import recipe
    jobs=recipe.jobs(recipe.configuration())
    assert len({j["path"] for j in jobs}) == len(jobs)
    rasters=[j for j in jobs if j["kind"]=="raster"]
    assert len(rasters)==12 and {j["seed"] for j in rasters}=={42}


def test_exp037_accuracy_summary_is_across_seed_mean_and_sample_sd() -> None:
    rows = [
        {"level": level, "seed": seed, "acc": value}
        for level, values in ((0.0, (80, 82, 84)), (0.5, (40, 50, 60)))
        for seed, value in zip(exp037.SEEDS_BASELINE, values)
    ]
    xs, means, sds = exp037.summarize_accuracy(rows, "level")
    np.testing.assert_allclose(xs, [0.0, 0.5])
    np.testing.assert_allclose(means, [82.0, 50.0])
    np.testing.assert_allclose(sds, [2.0, 10.0])


def test_exp037_publication_rows_retain_seed_and_sample_provenance() -> None:
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
            exp037.SEEDS_BASELINE, (70, 80, 90), (8, 9, 10)
        )
    ]
    summary = exp037.summarize_perturbation_rows(rows)
    assert summary == [{
        "model": "ping",
        "mode": "drop",
        "level": 0.5,
        "acc": 80.0,
        "acc_sd": 10.0,
        "e_rate_hz": 9.0,
        "e_rate_hz_sd": 1.0,
        "seeds": [42, 43, 44],
        "n_total_per_seed": [1400, 1400, 1400],
    }]


def test_exp037_quantitative_inference_uses_publication_subset(tmp_path):
    from experiments.exp037 import recipe
    job=next(j for j in recipe.jobs(recipe.configuration()) if j["kind"]=="sweep")
    args=recipe.inference_args(tmp_path,tmp_path/"weights.pth",tmp_path/"out",job)
    assert args[args.index("--max-samples")+1]=="1000"
    assert args[args.index("--load-weights")+1].endswith("weights.pth")


def test_exp038_ei_summary_is_across_seed_mean_and_sample_sd() -> None:
    points = [
        {
            "seed": seed,
            "ei_strength": ei,
            "acc": acc,
            "hid_rate_hz": rate,
            "inh_rate_hz": rate / 2,
            "n_total": 1400,
        }
        for ei, accs, rates in (
            (0.0, (88, 90, 92), (120, 130, 140)),
            (1.0, (50, 55, 60), (8, 9, 10)),
        )
        for seed, acc, rate in zip(exp038.SEEDS_BASELINE, accs, rates)
    ]
    summary = exp038.summarize_ei_points(points)
    assert [row["ei_strength"] for row in summary] == [0.0, 1.0]
    assert summary[0]["acc"] == 90.0
    assert summary[0]["acc_sd"] == 2.0
    assert summary[1]["hid_rate_hz"] == 9.0
    assert summary[1]["hid_rate_hz_sd"] == 1.0
    assert {point["n_total"] for point in points} == {1400}


def test_exp038_quantitative_inference_uses_reduced_eval_subset(tmp_path: Path) -> None:
    from experiments.exp038 import recipe
    job = next(j for j in recipe.jobs(recipe.configuration()) if j["kind"] == "ei_sweep")
    args = recipe.inference_args(tmp_path, tmp_path / "weights.pth", tmp_path / "out", job)
    assert args[args.index("--max-samples") + 1] == "1000"
    assert args[args.index("--load-weights") + 1].endswith("weights.pth")
    assert args[args.index("--skip-load") + 1:args.index("--skip-load") + 3] == ["W_ei.", "W_ie."]


def test_downstream_runners_honor_isolated_runner_paths(tmp_path: Path) -> None:
    active = REPO / "artifacts"
    before = {
        path.relative_to(active): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in active.rglob("*")
        if path.is_file()
    }
    for slug in (
        "exp033",
        "exp082",
    ):
        root = tmp_path / slug
        env = {
            **os.environ,
            "PINGLAB_REQUIRE_ISOLATED": "1",
            "PINGLAB_RUN_STATE_DIR": str((root / "state").resolve()),
            "PINGLAB_RUN_DERIVED_DIR": str((root / "derived").resolve()),
            "PINGLAB_RUN_LOG_DIR": str((root / "logs").resolve()),
            "PINGLAB_TRAINING_ROOT": str((tmp_path / "training").resolve()),
        }
        code = (
            f"from experiments import {slug} as m; import json; "
            "print(json.dumps({'state': str(m.RUN_PATHS.state), "
            "'derived': str(m.FIGURES), 'logs': str(m.RUN_PATHS.logs)}))"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=REPO,
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )
        paths = json.loads(result.stdout)
        assert paths == {
            "state": str((root / "state").resolve()),
            "derived": str((root / "derived").resolve()),
            "logs": str((root / "logs").resolve()),
        }
    after = {
        path.relative_to(active): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in active.rglob("*")
        if path.is_file()
    }
    assert after == before
