from __future__ import annotations

import json
from pathlib import Path

import pytest
from experiments import exp044


def _common_config() -> dict:
    return {
        "model": "ping", "dataset": "mnist", "max_samples": 7000,
        "epochs": 50, "t_ms": 200.0, "tau_ampa_ms": 2.0,
        "tau_gaba_ms": 6.0, "input_rate": 25.0,
        "input_rate_sampling": "fixed", "hidden_sizes": [1024],
        "n_in": 784, "n_hidden": 1024, "n_inh": 256, "n_out": 10,
        "ei_strength": 1.0, "w_in": [0.9, 0.09], "w_in_initial_zero_fraction": 0.95,
        "readout_mode": "mem-mean", "readout_w_init_mean": 1.12060546875,
        "readout_w_init_std": 0.8349609375, "surrogate_slope": 1.0,
        "lr": 0.0004, "batch_size": 256, "weight_decay": 0.0,
        "grad_clip": 1.0, "v_grad_dampen": 1000.0, "dales_law": True,
        "trainable_w_ei": False, "trainable_w_ie": False,
    }


def _write_cells(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[tuple[float, int], Path]:
    directories: dict[tuple[float, int], Path] = {}
    for dt_ms in exp044.DT_SWEEP_MS:
        for seed in exp044.SEEDS:
            directory = tmp_path / f"ping__{exp044.dt_label(dt_ms)}__seed{seed}"
            directory.mkdir()
            config = {**_common_config(), "dt": dt_ms, "seed": seed}
            (directory / "config.json").write_text(json.dumps(config))
            directories[(dt_ms, seed)] = directory
    monkeypatch.setattr(exp044, "cell_dir", lambda dt_ms, seed: directories[(dt_ms, seed)])
    return directories


def test_training_contract_verifies_all_15_cells(tmp_path: Path, monkeypatch) -> None:
    _write_cells(tmp_path, monkeypatch)
    contract = exp044.resolve_training_contract()
    assert contract["common"]["batch_size"] == 256
    assert len(contract["cells"]) == 15
    assert {cell["dt_ms"] for cell in contract["cells"]} == set(exp044.DT_SWEEP_MS)
    assert {cell["seed"] for cell in contract["cells"]} == set(exp044.SEEDS)


@pytest.mark.parametrize("field", exp044.TRAINING_COMMON_FIELDS)
def test_training_contract_rejects_each_unregistered_difference(
    field: str, tmp_path: Path, monkeypatch,
) -> None:
    directories = _write_cells(tmp_path, monkeypatch)
    target = directories[(exp044.DT_SWEEP_MS[-1], exp044.SEEDS[-1])]
    config = json.loads((target / "config.json").read_text())
    value = config[field]
    if isinstance(value, bool):
        config[field] = not value
    elif isinstance(value, list):
        config[field] = [*value, 999]
    elif isinstance(value, (int, float)):
        config[field] = value + 1
    else:
        config[field] = f"{value}-mismatch"
    (target / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match=f"config {field}="):
        exp044.resolve_training_contract()


@pytest.mark.parametrize(("field", "value"), [("dt", 0.2), ("seed", 99)])
def test_training_contract_rejects_unregistered_identity(
    field: str, value: object, tmp_path: Path, monkeypatch,
) -> None:
    directories = _write_cells(tmp_path, monkeypatch)
    target = directories[(exp044.DT_SWEEP_MS[0], exp044.SEEDS[0])]
    config = json.loads((target / "config.json").read_text())
    config[field] = value
    (target / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match=f"config {field}"):
        exp044.resolve_training_contract()
