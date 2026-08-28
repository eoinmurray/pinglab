from __future__ import annotations

from pathlib import Path

from experiments import exp022, exp025


def test_frontier_strength_is_owned_by_exp022() -> None:
    assert exp025.FR_STRENGTH_UPPER == exp022.FR_STRENGTH_UPPER == 0.041


def _frontier_rows() -> list[dict]:
    return [
        {
            "cell_name": exp025.cell_name(model, rate_target_hz, seed),
            "model": model,
            "rate_target_hz": rate_target_hz,
            "rate_target_display": exp025.rate_target_display(rate_target_hz),
            "seed": seed,
            "final_acc": 80.0 + seed / 100.0,
            "rate_e": 10.0 + seed / 100.0,
            "evaluation_partition": "official_mnist_test",
            "evaluation_samples": exp025.EVAL_MAX_SAMPLES,
            "checkpoint_role": exp025.CHECKPOINT_ROLE,
        }
        for model in exp025.MODELS
        for rate_target_hz in exp025.RATE_TARGET_GRID_HZ
        for seed in exp025.seeds_for(rate_target_hz)
    ]


def test_frontier_consumes_all_36_registered_tr02_cells() -> None:
    rows = _frontier_rows()
    assert len(rows) == 36
    assert len({row["cell_name"] for row in rows}) == 36
    assert {row["seed"] for row in rows} == {42, 43, 44}


def test_frontier_statistics_record_three_seed_provenance_per_point() -> None:
    stats = exp025.aggregate_frontier(_frontier_rows())
    assert len(stats) == 12
    for point in stats:
        assert point["statistic"] == "mean_across_independent_seeds"
        assert point["uncertainty"] == "sem_across_independent_seeds"
        assert point["n_seeds"] == 3
        assert point["seeds"] == [42, 43, 44]
        assert len(point["cell_names"]) == 3


def test_smoke_scaled_inference_caps_dataset(tmp_path: Path) -> None:
    from experiments.exp025 import recipe

    jobs = [
        j for j in recipe.jobs(recipe.configuration(smoke=True)) if j["kind"] == "scale"
    ]
    assert len(jobs) == 6
    args = recipe.inference_args(
        tmp_path, tmp_path / "weights_final.pth", tmp_path / "out", jobs[0]
    )
    assert args[args.index("--max-samples") + 1] == "100"
    assert args[args.index("--scale-w-in") + 1] == "0.5"
    assert args[args.index("--outputs") + 1] == "per_cell_rates"
    assert args[args.index("--output-fields") + 1 :] == ["rate_e_per_sample"]


def test_frontier_endpoint_requests_one_official_test_forward_pass(
    tmp_path: Path,
) -> None:
    from experiments.exp025 import recipe

    jobs = [j for j in recipe.jobs(recipe.configuration()) if j["kind"] == "frontier"]
    assert len(jobs) == 36
    assert len({j["cell_name"] for j in jobs}) == 36
    args = recipe.inference_args(
        tmp_path, tmp_path / "weights_final.pth", tmp_path / "out", jobs[0]
    )
    assert args[args.index("--max-samples") + 1] == "1000"
    assert "--outputs" not in args


def test_low_w_in_cells_are_owned_by_exp022() -> None:
    assert tuple(exp025.LOW_W_IN_VALUES) == (0.05, 0.1, 0.3, 0.9)
    assert (
        exp025.low_w_in_cell_name(0.9, 43)
        == exp022.training_run_cell("TR-07", w_in=0.9, seed=43)["name"]
    )


def test_low_w_in_production_grid_uses_three_seeds() -> None:
    assert exp025.LOW_W_IN_SEEDS == [42, 43, 44]
    paths = {
        exp025.low_w_in_cell_name(w_in, seed)
        for w_in in exp025.LOW_W_IN_VALUES
        for seed in exp025.LOW_W_IN_SEEDS
    }
    assert len(paths) == 12


def test_low_w_in_aggregation_reports_mean_and_sem() -> None:
    rows = [
        {"seed": 42, "final_acc": 80.0, "rate_e": 10.0, "rate_i": 5.0},
        {"seed": 43, "final_acc": 82.0, "rate_e": 12.0, "rate_i": 7.0},
        {"seed": 44, "final_acc": 84.0, "rate_e": 14.0, "rate_i": 9.0},
    ]
    result = exp025.aggregate_low_w_in_seed_rows(0.3, rows)
    assert result["n_seeds"] == 3
    assert result["seeds"] == [42, 43, 44]
    assert result["final_acc"] == 82.0
    assert result["final_acc_sem"] > 0
    assert result["statistic"] == "mean_across_independent_seeds"
    assert result["uncertainty"] == "sem_across_independent_seeds"
