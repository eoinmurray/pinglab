from __future__ import annotations

import numpy as np
from experiments import exp022, exp082


def test_exp022_planned_bank_targets_exp082() -> None:
    cells = exp022.PLANNED_VARIABLE_RATE_CELLS
    assert [cell["name"] for cell in cells] == [
        "ping__variable_rate__seed42",
        "ping__variable_rate__seed43",
        "ping__variable_rate__seed44",
    ]
    assert all(cell["consumer"] == "exp082" for cell in cells)
    assert all(cell["readout"] == "spike-rate" for cell in cells)
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
    assert args[args.index("--readout") + 1] == "spike-rate"
    start = args.index("--input-rates") + 1
    assert tuple(map(float, args[start : start + 6])) == exp082.TRAINING_RATES_HZ


def test_exp022_wilkes_resource_tiers_partition_registry() -> None:
    expected = {
        "standard": 78,
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


def test_spike_rate_logits_use_only_matched_window_and_duration() -> None:
    spikes = np.asarray(
        [
            [1, 0],
            [0, 1],
            [1, 1],
            [9, 9],
        ],
        dtype=np.float32,
    )
    logits = exp082.spike_rate_logits(spikes, start=1, stop=3, dt_ms=100.0)
    np.testing.assert_array_equal(logits, np.asarray([5, 10], dtype=np.float32))


def test_psychometric_is_fixed_at_training_duration() -> None:
    assert exp082.MATCHED_DURATION_MS == 200.0
    assert exp082.PSYCHOMETRIC_RATES_HZ == exp082.TRAINING_RATES_HZ
