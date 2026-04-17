import numpy as np
import pytest

from metrics import (
    find_fundamental_nondiff,
    format_metrics,
    population_rate_nondiff,
)


class TestPopulationRate:
    def test_constant_rate_from_known_spike_train(self):
        """Every neuron fires exactly once per 1ms bin at dt=0.25ms
        => expect 1000 Hz/neuron."""
        n_neurons = 10
        dt = 0.25
        bin_ms = 1.0
        bin_steps = int(bin_ms / dt)  # 4
        n_bins = 50
        spikes = np.zeros((n_bins * bin_steps, n_neurons))
        spikes[::bin_steps, :] = 1  # one spike per bin per neuron
        t, rate = population_rate_nondiff(spikes, n_neurons, bin_ms, dt)
        assert t.shape == (n_bins,)
        assert rate.shape == (n_bins,)
        np.testing.assert_allclose(rate, 1000.0)

    def test_silent_network(self):
        spikes = np.zeros((400, 8))
        _, rate = population_rate_nondiff(spikes, 8, bin_ms=1.0, dt_ms=0.25)
        assert np.all(rate == 0.0)

    def test_time_bin_positions(self):
        spikes = np.zeros((200, 4))
        t, _ = population_rate_nondiff(spikes, 4, bin_ms=2.0, dt_ms=0.25)
        assert t[0] == 0.0
        assert t[1] == 2.0


class TestFindFundamental:
    def _make_psd_with_peak(self, peak_hz, sub_ratio=0.0, snr=10.0,
                            f_hi=80.0, n=257, bin_ms=2.0):
        freqs = np.fft.rfftfreq(2 * (n - 1), d=bin_ms / 1000.0)
        # Background noise at median=1
        psd = np.ones_like(freqs)
        # Peak
        peak_idx = int(np.argmin(np.abs(freqs - peak_hz)))
        psd[peak_idx] = snr
        if sub_ratio > 0:
            sub_idx = int(np.argmin(np.abs(freqs - peak_hz / 2)))
            psd[sub_idx] = snr * sub_ratio
        return psd, freqs

    def test_pure_peak_returned(self):
        psd, freqs = self._make_psd_with_peak(40.0)
        f0 = find_fundamental_nondiff(psd, freqs)
        assert f0 == pytest.approx(40.0, abs=2.0)

    def test_subharmonic_takes_precedence(self):
        psd, freqs = self._make_psd_with_peak(60.0, sub_ratio=0.5)
        f0 = find_fundamental_nondiff(psd, freqs)
        assert f0 == pytest.approx(30.0, abs=2.0)

    def test_weak_subharmonic_ignored(self):
        psd, freqs = self._make_psd_with_peak(60.0, sub_ratio=0.1)
        f0 = find_fundamental_nondiff(psd, freqs)
        assert f0 == pytest.approx(60.0, abs=2.0)

    def test_flat_psd_returns_zero(self):
        freqs = np.linspace(0, 100, 200)
        psd = np.ones_like(freqs)
        assert find_fundamental_nondiff(psd, freqs) == 0.0

    def test_empty_band_returns_zero(self):
        freqs = np.linspace(100.0, 200.0, 50)
        psd = np.ones_like(freqs)
        assert find_fundamental_nondiff(psd, freqs, f_lo=5.0, f_hi=80.0) == 0.0


class TestFormatMetrics:
    def test_f0_shown_only_when_positive(self):
        base = {"rate_e": 30.0, "rate_i": 80.0, "cv": 0.5, "act": 0.6, "f0": 0.0}
        assert "f0" not in format_metrics(base)
        base["f0"] = 42.0
        assert "f0=" in format_metrics(base)

    def test_basic_fields_present(self):
        m = {"rate_e": 30.0, "rate_i": 80.0, "cv": 0.5, "act": 0.6, "f0": 0.0}
        s = format_metrics(m)
        assert "E=" in s and "I=" in s and "CV=" in s and "act=" in s
