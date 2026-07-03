"""Coverage for pure metrics/runlog helpers not hit by the integration tests:
the compact scan-frame strings, PSD/pop-rate wrappers, and the human-readable
formatters + provenance/file-listing helpers.
"""

from __future__ import annotations

import numpy as np
import runlog
from metrics import compute_pop_rate, compute_psd, metrics_str

# ── metrics.py ────────────────────────────────────────────────────────────


def _poisson_raster(T, n, rate=0.1, seed=0):
    rng = np.random.RandomState(seed)
    return (rng.random((T, n)) < rate).astype(np.float32)


def test_metrics_str_reports_rates_and_cv():
    spk_e = _poisson_raster(400, 32, rate=0.1, seed=1)
    spk_i = _poisson_raster(400, 8, rate=0.2, seed=2)
    s = metrics_str(spk_e, spk_i, dt=0.25, n_e=32, n_i=8)
    assert s.startswith("E=")
    assert "I=" in s and "CV=" in s and "I/E=" in s


def test_metrics_str_no_spikes_when_zero_duration():
    empty = np.zeros((0, 16), dtype=np.float32)
    assert metrics_str(empty, None, dt=0.25, n_e=16, n_i=4) == "no spikes"


def test_metrics_str_single_bin_has_zero_cv():
    # Fewer steps than one 2 ms bin at dt=0.25 (bin_steps=8) → n_bins<=1 branch.
    spk_e = _poisson_raster(4, 16, rate=0.3, seed=3)
    s = metrics_str(spk_e, None, dt=0.25, n_e=16, n_i=4)
    assert "CV=0.00" in s
    assert "I=0" in s  # spk_i is None → I rate reported as 0


def test_compute_pop_rate_shape():
    spk = _poisson_raster(200, 20, rate=0.1, seed=4)
    t, rate = compute_pop_rate(spk, 20, dt=0.25, bin_ms=2.0)
    assert t.shape == rate.shape
    assert (rate >= 0).all()


def test_compute_psd_normal_is_peak_normalised():
    spk = _poisson_raster(400, 20, rate=0.15, seed=5)
    freqs, psd = compute_psd(spk, 20, dt=0.25, bin_ms=2.0)
    assert freqs.shape == psd.shape
    assert psd.max() <= 1.0 + 1e-9


def test_compute_psd_too_short_returns_zeros():
    spk = np.zeros((1, 8), dtype=np.float32)
    freqs, psd = compute_psd(spk, 8, dt=0.25)
    assert freqs.tolist() == [0.0]
    assert psd.tolist() == [0.0]


# ── runlog.py formatters + helpers ────────────────────────────────────────


def test_format_eta_covers_all_ranges():
    assert runlog.format_eta(45) == "45s"
    assert runlog.format_eta(510) == "8m30s"
    assert runlog.format_eta(4320) == "1h12m"


def test_format_bytes_covers_all_units():
    assert runlog.format_bytes(512) == "512 B"
    assert runlog.format_bytes(2048).endswith("KB")
    assert runlog.format_bytes(5 * 1024**2).endswith("MB")
    assert runlog.format_bytes(3 * 1024**3).endswith("GB")
    assert runlog.format_bytes(2 * 1024**4).endswith("TB")


def test_provenance_returns_expected_keys():
    # provenance() calls _git_sha() and _env_hash() internally.
    p = runlog.provenance()
    assert {"git_sha", "python_env_hash", "run_id", "torch_version"} <= p.keys()


def test_list_output_files_lists_written_files(tmp_path):
    (tmp_path / "a.json").write_text("{}")
    (tmp_path / "b.npz").write_bytes(b"\x00\x01")
    files = runlog.list_output_files(tmp_path)  # list of (name, size_bytes) tuples
    names = {name for name, _ in files}
    assert {"a.json", "b.npz"} <= names
