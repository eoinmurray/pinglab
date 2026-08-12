from __future__ import annotations

import json
from pathlib import Path

from experiments import exp025


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


def test_smoke_scaled_inference_caps_dataset(monkeypatch, tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    (train_dir / "config.json").write_text(json.dumps({"t_ms": 200.0}))
    infer_dir = tmp_path / "infer"
    infer_dir.mkdir()
    (infer_dir / "metrics.json").write_text(json.dumps({
        "best_acc": 10.0,
        "ce_loss": 2.3,
        "rates_hz": {"hid": 1.0, "inh": 0.5},
    }))
    observed: dict[str, object] = {}

    def fake_infer(
        _train_dir: Path,
        extra_args: list[str] | None = None,
        out_name: str = "infer",
        max_samples: int | None = None,
    ) -> Path:
        observed.update(
            extra_args=extra_args,
            out_name=out_name,
            max_samples=max_samples,
        )
        return infer_dir

    monkeypatch.setattr(exp025, "_infer_cell", fake_infer)
    monkeypatch.setattr(exp025, "SMOKE", True)
    monkeypatch.setattr(exp025, "MAX_SAMPLES", 100)
    exp025._eval_scaled(train_dir, scale_w_in=0.5)

    assert observed["max_samples"] == 100
    assert observed["extra_args"] == [
        "--scale-w-in", "0.5", "--outputs", "per_cell_rates",
    ]


def test_low_w_in_training_uses_direct_readout_initializer(tmp_path: Path) -> None:
    args = exp025.build_low_w_in_args(0.9, tmp_path / "cell")
    assert "--readout-w-out-scale" not in args
    assert args[args.index("--readout-w-init-mean") + 1] == "1.12060546875"
    assert args[args.index("--readout-w-init-std") + 1] == "0.8349609375"
    assert exp025.LOW_W_IN_VALUES == [0.05, 0.1, 0.3, 0.9]
