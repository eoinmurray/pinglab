from __future__ import annotations

import numpy as np
import pytest
from experiments import exp099


def test_checkpoint_dependency_is_hash_pinned_and_external() -> None:
    directory, record = exp099.require_checkpoint()
    assert directory.name == "ping__canonical__seed42"
    assert record["role"] == "final_epoch"
    assert record["sha256"] == exp099.EXPECTED_CHECKPOINT_SHA256
    assert exp099.REPO / "artifacts" / "data" / "exp099" not in record["path"].parents


def test_rate_sweep_matches_historical_range() -> None:
    assert len(exp099.INPUT_RATES_HZ) == 40
    assert exp099.INPUT_RATES_HZ[0] == 0.0
    assert exp099.INPUT_RATES_HZ[-1] == 100.0
    assert np.all(np.diff(exp099.INPUT_RATES_HZ) > 0)


def test_pixel_inputs_are_paired_and_nested() -> None:
    pixels = np.linspace(0, 1, exp099.N_INPUT, dtype=np.float32)
    bank = exp099.paired_input_bank(pixels)
    assert bank[0.0].shape == (4_000, 1, 784)
    assert not bank[0.0].any()
    assert np.all(bank[25.641025641025642] <= bank[100.0])


def test_off_pixels_remain_silent() -> None:
    pixels = np.ones(exp099.N_INPUT, dtype=np.float32)
    pixels[:10] = 0
    assert not exp099.paired_input_bank(pixels)[100.0][:, :, :10].any()


def test_zero_raster_has_no_state_label() -> None:
    arrays = {
        "e": np.zeros((4_000, exp099.N_E), dtype=np.uint8),
        "i": np.zeros((4_000, exp099.N_I), dtype=np.uint8),
    }
    row = exp099.describe(0.0, arrays)
    assert row["e_rate_hz"] == 0.0
    assert row["median_e_isi_cv"] is None
    assert row["dominant_frequency_hz"] is None
    assert "state" not in row


def test_positive_lag_means_e_leads_i() -> None:
    e = np.zeros((4_000, exp099.N_E), dtype=np.uint8)
    i = np.zeros((4_000, exp099.N_I), dtype=np.uint8)
    for step in (1_200, 1_600, 2_000, 2_400, 2_800, 3_200):
        e[step, :10] = 1
        i[step + 10, :10] = 1
    assert exp099._e_i_lag(e, i) == 1.0


def test_mnist_test_image_identity() -> None:
    pixels, label = exp099.load_test_image()
    assert pixels.shape == (784,)
    assert pixels.min() >= 0 and pixels.max() <= 1
    assert label == 7


def test_scale_records_upstream_and_input_provenance() -> None:
    assert exp099.SCALE["upstream_publication"].startswith("ggs-production-composite")
    assert exp099.SCALE["checkpoint_role"] == "final_epoch"
    assert exp099.SCALE["dataset_split"] == "official_test"
    assert exp099.SCALE["poisson_seed"] == 9900


def test_checkpoint_mismatch_fails_closed(monkeypatch) -> None:
    monkeypatch.setattr(exp099, "EXPECTED_CHECKPOINT_SHA256", "bad")
    with pytest.raises(RuntimeError, match="does not match"):
        exp099.require_checkpoint()
