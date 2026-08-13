"""Focused unit tests for the standalone EXP080 empirical runner."""

from __future__ import annotations

import numpy as np
from experiments import exp080


def test_registered_simulator_validations_pass() -> None:
    record = exp080.validate_simulator()
    assert all(record["checks"].values())


def test_early_spike_contributes_more_than_late_spike() -> None:
    assert exp080.probe_single_spike(20.0) > exp080.probe_single_spike(180.0) > 0.0


def test_direct_features_replay_and_zero_input() -> None:
    import torch

    device = exp080.torch_device()
    images = torch.zeros((2, 28, 28), dtype=torch.uint8, device=device)
    rates = torch.tensor([0.5, 25.0], device=device)
    first = exp080.direct_features(
        images,
        rates,
        torch.Generator(device=device).manual_seed(123),
    )
    replay = exp080.direct_features(
        images,
        rates,
        torch.Generator(device=device).manual_seed(123),
    )
    assert torch.equal(first, replay)
    assert np.all(first.cpu().numpy() == 0.0)


def test_rate_grid_brackets_registered_floor() -> None:
    assert exp080.RATES_HZ == (0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0)
    assert exp080.USEFUL_ACCURACY == 0.5


def test_floor_requires_every_decoder_to_cross_criterion(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(exp080, "FIGURES", tmp_path)
    correctness = np.zeros((len(exp080.RATES_HZ), len(exp080.SEEDS), 10), dtype=bool)
    correctness[1, :, :6] = True
    correctness[1, 2, 4:6] = False
    correctness[2:, :, :6] = True

    decision = exp080.analyze(correctness)

    assert decision["r_train_hz"] == 0.5
    assert decision["criterion_crossed"] is True
    assert decision["rows"][1]["accuracy"] > exp080.USEFUL_ACCURACY
    assert decision["rows"][1]["minimum_seed_accuracy"] < exp080.USEFUL_ACCURACY


def test_no_crossing_is_recorded_as_a_censored_result(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(exp080, "FIGURES", tmp_path)
    correctness = np.zeros((len(exp080.RATES_HZ), len(exp080.SEEDS), 10), dtype=bool)

    decision = exp080.analyze(correctness)

    assert decision["criterion_crossed"] is False
    assert decision["r_train_hz"] is None
    assert decision["recommendation"] == {
        "floor_hz": None,
        "ceiling_hz": max(exp080.RATES_HZ),
    }
    assert (tmp_path / "decision.json").is_file()
