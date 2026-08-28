from __future__ import annotations

from pathlib import Path

import numpy as np
from experiments import exp022, exp082
from experiments.collections.gamma_gated_sparsity.plan import build_plan


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
        exp022._stamp_training_run_identity.__globals__,
        "cell_dir",
        lambda _name: directory,
    )

    exp022._stamp_training_run_identity(cell)

    config = exp022.json.loads((directory / "config.json").read_text())
    metrics = exp022.json.loads((directory / "metrics.json").read_text())
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
