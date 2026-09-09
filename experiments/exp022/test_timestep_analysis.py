"""Physical-time checks for the collection's nonintegral analysis bins."""

import numpy as np
import pytest
from experiments.exp022.analyse import _gamma_psd, bank_composition, measure_snapshot
from experiments.helpers import rhythmicity as figure_metrics
from pingstore.contracts import PingstoreError


def periodic_raster(dt, period_ms=30.0, duration_ms=1800.0):
    raster = np.zeros((round(duration_ms / dt), 10), dtype=bool)
    events = np.rint(np.arange(0, duration_ms, period_ms) / dt).astype(int)
    raster[events] = True
    return raster


@pytest.mark.parametrize("dt", [0.05, 0.1, 0.2, 0.3, 0.6])
def test_spectrum_reports_physical_frequency_across_timestep_grid(dt):
    frequencies, power, peak = _gamma_psd(periodic_raster(dt), dt)
    assert frequencies.shape == power.shape
    assert peak == pytest.approx(1000 / 30, abs=0.7)


@pytest.mark.parametrize("dt", [0.1, 0.3, 0.6])
def test_autocorrelation_lags_and_scalar_lookup_use_physical_time(dt):
    raster = periodic_raster(dt)
    lags, ac = figure_metrics.spike_autocorrelogram(raster, dt, max_lag_ms=60)
    window = (lags >= 20) & (lags <= 40)
    peak_lag = lags[window][np.argmax(ac[window])]
    assert peak_lag == pytest.approx(30, abs=1.2)
    # IEI bins remain physical milliseconds and can differ from the AC bins.
    result = figure_metrics.rhythmicity_scalars(
        lags, ac, np.array([30.0]), np.array([1]), bio_lag_ms=30.0
    )
    expected = ac[round(30 / (lags[1] - lags[0]))]
    assert result["biophysical"] == expected
    assert result["iei_anchored"] == expected


@pytest.mark.parametrize("dt,steps", [(0.1, 2000), (0.3, 666), (0.6, 333)])
def test_snapshot_rate_integral_preserves_spike_counts_in_partial_bin(tmp_path, dt, steps):
    spikes = np.zeros((steps, 2), dtype=bool)
    spikes[-1] = True
    source, destination = tmp_path / "recording.npz", tmp_path / "rasters.npz"
    np.savez(source, spk_e=spikes, spk_i=spikes, dt=np.float32(dt))
    measure_snapshot(source, destination)
    with np.load(destination) as measured:
        assert measured["bin_widths_ms"].sum() == pytest.approx(steps * dt)
        assert len(measured["bin_widths_ms"]) == 200
        assert measured["bin_widths_ms"][-1] == pytest.approx(1.0 if dt == 0.1 else 0.8)
        for population in ("e", "i"):
            integrated = np.sum(measured[f"{population}_rate"] * measured["bin_widths_ms"]) / 1000
            assert integrated == pytest.approx(1.0)
            assert measured[f"{population}_hz"] == pytest.approx(1000 / (steps * dt))


def test_bank_composition_rejects_overlap_or_missing_cells():
    record = {
        "inputs": {"retained_bank": {"run_id": "source", "payload_digest": "digest"}},
        "bank_reuse": {"plan": {
            "reused_cells": ["old"], "new_cells": ["new"],
            "diagnostic_policy": "regenerate_all",
        }},
    }
    result = bank_composition(record, {"old", "new"})
    assert result["reused_cells"] == result["new_cells"] == 1
    with pytest.raises(PingstoreError, match="partition"):
        bank_composition(record, {"old", "new", "missing"})
    record["bank_reuse"]["plan"]["new_cells"].append("old")
    with pytest.raises(PingstoreError, match="partition"):
        bank_composition(record, {"old", "new"})
