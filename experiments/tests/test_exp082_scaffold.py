from __future__ import annotations

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
        tuple(cell["input_rates_hz"]) == exp082.TRAINING_RATES_HZ
        for cell in cells
    )
    assert all(cell in exp022.CANONICAL_CELLS for cell in cells)


def test_exp022_variable_rate_args() -> None:
    cell = exp022.PLANNED_VARIABLE_RATE_CELLS[0]
    args = exp022.build_train_args(cell, exp082.training_dir(42), 7000, 50)
    assert args[args.index("--readout") + 1] == "spike-count"
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
        family: {cell["training_run_id"] for cell in exp022.CANONICAL_CELLS
                 if cell["family"] == family}
        for family in exp022.TRAINING_RUN_IDS
    }
    assert by_family == {
        family: {run_id}
        for family, run_id in exp022.TRAINING_RUN_IDS.items()
    }


def test_completed_cell_artifacts_are_stamped_with_training_run_id(
    tmp_path, monkeypatch,
) -> None:
    cell = exp022.PLANNED_VARIABLE_RATE_CELLS[0]
    directory = tmp_path / cell["name"]
    directory.mkdir()
    (directory / "config.json").write_text('{"mode": "train"}\n')
    (directory / "metrics.json").write_text('{"config": {"epochs": 50}}\n')
    monkeypatch.setattr(exp022, "cell_dir", lambda _name: directory)

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
    assert exp082.EVALUATION_PROFILE in {"smoke", "pilot", "production"}


def test_collection_requires_exp082_measurements_and_figures(tmp_path) -> None:
    plan = build_plan(tmp_path / "campaign", "exp082-contract")
    row = next(
        row
        for stage in plan["stages"]
        for row in stage["experiments"]
        if row["slug"] == "exp082"
    )
    assert [path.rsplit("/", 1)[-1] for path in row["required_outputs"]] == [
        "numbers.json",
        "measurements.npz",
        "matched_stream.png",
        "variable_stream.png",
        "psychometric_200ms.svg",
        "duration_rate_summary.png",
    ]


def test_saved_measurements_replot_every_figure(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(exp082, "FIGURES", tmp_path)
    stream = {
        "boundaries": [0, 2],
        "conditions": [[0.2, 5.0]],
        "labels": [1],
        "predictions": [1],
        "correct": [1],
        "spikes_e": np.zeros((2, 4), dtype=np.int8),
        "spikes_i": np.zeros((2, 2), dtype=np.int8),
        "spikes_out": np.zeros((2, 10), dtype=np.int8),
        "probabilities": np.full((2, 10), 0.1, dtype=np.float32),
    }
    exp082.save_measurements(stream, stream)
    payload = {
        "run_id": "test",
        "matched_stream": {key: value for key, value in stream.items() if not isinstance(value, np.ndarray)},
        "variable_stream": {key: value for key, value in stream.items() if not isinstance(value, np.ndarray)},
        "grid_per_seed": [
            {"seed": seed, "duration_ms": duration, "rate_hz": rate,
             "n_correct": 1, "n_total": 1, "accuracy": 1.0}
            for duration in exp082.DURATIONS_MS
            for rate in exp082.PSYCHOMETRIC_RATES_HZ
            for seed in exp082.SEEDS
        ],
    }
    payload["duration_200ms_psychometric"] = [
        row for row in payload["grid_per_seed"]
        if row["duration_ms"] == exp082.MATCHED_DURATION_MS
    ]
    numbers = tmp_path / "numbers.json"
    numbers.write_text(exp082.json.dumps(payload))
    exp082.replot_results(numbers, tmp_path / exp082.MEASUREMENTS_FILE)
    for filename in (
        "matched_stream.png", "variable_stream.png",
        "psychometric_200ms.svg", "duration_rate_summary.png",
    ):
        assert (tmp_path / filename).is_file()
    first_hashes = {
        filename: (tmp_path / filename).read_bytes()
        for filename in (
            "matched_stream.png", "variable_stream.png",
            "psychometric_200ms.svg", "duration_rate_summary.png",
        )
    }
    exp082.replot_results(numbers, tmp_path / exp082.MEASUREMENTS_FILE)
    assert {
        filename: (tmp_path / filename).read_bytes()
        for filename in first_hashes
    } == first_hashes
