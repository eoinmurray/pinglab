import numpy as np
import pytest

from pinglab.types import Spikes, InstrumentsResults, NetworkResult
from pinglab.analysis import (
    population_rate,
    population_isi_cv,
    population_fano_factor,
    mean_firing_rates,
    plv_from_phase_series,
    population_plv,
    synchrony_index,
    ei_lag_stats,
    crosscorr,
    conductance_stats,
    energy_metrics,
    pairwise_spike_count_corr,
    rate_coherence,
    base_metrics,
)


def make_spikes(times, ids):
    return Spikes(
        times=np.array(times, dtype=float),
        ids=np.array(ids, dtype=int),
    )


def test_population_rate_basic():
    spikes = make_spikes([1, 2, 11], [0, 1, 2])
    t_ms, rate_e = population_rate(spikes, T_ms=20.0, dt_ms=10.0, pop="E", N_E=2)
    _, rate_i = population_rate(spikes, T_ms=20.0, dt_ms=10.0, pop="I", N_E=2, N_I=1)
    _, rate_all = population_rate(spikes, T_ms=20.0, dt_ms=10.0, pop="all", N_E=2, N_I=1)

    assert t_ms.size == 2
    np.testing.assert_allclose(rate_e, [100.0, 0.0])
    np.testing.assert_allclose(rate_i, [0.0, 100.0])
    np.testing.assert_allclose(rate_all, [66.6666667, 33.3333333], rtol=1e-6)


def test_population_isi_cv_zero():
    spikes = make_spikes([0, 10, 20, 5, 15, 25], [0, 0, 0, 1, 1, 1])
    cv_pop, cv_per = population_isi_cv(spikes)
    assert cv_pop == 0.0
    np.testing.assert_allclose(cv_per, [0.0, 0.0])


def test_population_isi_cv_empty():
    spikes = make_spikes([], [])
    cv_pop, cv_per = population_isi_cv(spikes)
    assert cv_pop == 0.0
    assert cv_per.size == 0


def test_population_fano_factor_constant_counts():
    spikes = make_spikes([1, 11, 21], [0, 0, 0])
    fano_e, fano_i = population_fano_factor(spikes, T=30.0, N_E=1, N_I=1, window_ms=10.0)
    assert fano_e == 0.0
    assert fano_i == 0.0


def test_mean_firing_rates_basic():
    spikes = make_spikes([0, 1000, 0, 500], [0, 0, 1, 1])
    e_rate, i_rate = mean_firing_rates(spikes, N_E=1, N_I=1)
    assert e_rate == 1.0
    assert i_rate == 2.0


def test_mean_firing_rates_population_mean():
    spikes = make_spikes([0, 1000, 2000, 0, 500, 1000], [0, 0, 0, 1, 1, 1])
    e_rate, i_rate = mean_firing_rates(spikes, N_E=2, N_I=1)
    assert e_rate == 1.5
    assert i_rate == 0.0


def test_plv_from_phase_series_locked():
    t_ms = np.arange(0.0, 1000.0, 1.0)
    freq = 40.0
    phase = 2.0 * np.pi * freq * t_ms / 1000.0
    spike_times = np.arange(0.0, 1000.0, 25.0)
    plv = plv_from_phase_series(t_ms, phase, spike_times)
    assert plv == pytest.approx(1.0, rel=1e-3)


def test_population_plv_empty():
    spikes = make_spikes([], [])
    plv = population_plv(spikes, T_ms=100.0, dt_ms=5.0, fmin=30.0, fmax=80.0, pop="all", N_E=1, N_I=1)
    assert plv == 0.0


def test_synchrony_index_constant_counts():
    spikes = make_spikes([0, 10, 20, 30, 40, 50, 60, 70, 80, 90], [0] * 10)
    si = synchrony_index(spikes, T=100.0, bin_ms=10.0, N_E=1, N_I=0)
    assert si == 0.0


def test_ei_lag_stats_simple():
    spikes = make_spikes([0, 10, 20, 2, 12, 22], [0, 0, 0, 1, 1, 1])
    mean_lag, std_lag = ei_lag_stats(spikes, N_E=1, max_lag_ms=5.0)
    assert mean_lag == 2.0
    assert std_lag == 0.0


def test_crosscorr_peak_bin():
    spikes = make_spikes([0, 10, 20, 2, 12, 22], [0, 0, 0, 1, 1, 1])
    centers, hist = crosscorr(spikes, N_E=1, bin_ms=1.0, max_lag_ms=5.0)
    peak_idx = int(np.argmax(hist))
    assert centers[peak_idx] == 2.5
    assert hist[peak_idx] == 3


def test_rate_coherence_pairs():
    spikes = make_spikes(
        [5, 25, 45, 65, 85, 15, 35, 55, 75, 95],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    )
    coh = rate_coherence(spikes, T=100.0, bin_ms=10.0, N_E=2, N_I=0)
    assert coh == pytest.approx(-1.0)

    spikes_same = make_spikes(
        [5, 25, 45, 65, 85, 5, 25, 45, 65, 85],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    )
    coh_same = rate_coherence(spikes_same, T=100.0, bin_ms=10.0, N_E=2, N_I=0)
    assert coh_same == pytest.approx(1.0)


def test_conductance_stats_basic():
    g_e = np.ones((2, 2))
    g_i = np.ones((2, 2)) * 2.0
    g_e_mean, g_i_mean, ratio, g_e_cv, g_i_cv = conductance_stats(g_e, g_i)
    assert g_e_mean == 1.0
    assert g_i_mean == 2.0
    assert ratio == 0.5
    assert g_e_cv == 0.0
    assert g_i_cv == 0.0


def test_energy_metrics_basic():
    g_e = np.ones((2, 2))
    g_i = np.zeros((2, 2))
    spikes = make_spikes([1, 2, 3], [0, 1, 1])
    energy_cond, energy_spk, energy_tot, energy_eff = energy_metrics(
        g_e=g_e,
        g_i=g_i,
        spikes=spikes,
        dt=1.0,
        N_E=1,
        N_I=1,
    )
    assert energy_cond == 4.0
    assert energy_spk == 3.0
    assert energy_tot == 7.0
    np.testing.assert_allclose(energy_eff, 3.0 / 7.0)


def test_pairwise_spike_count_corr_signs():
    spikes = make_spikes([1, 21, 2, 22, 12], [0, 0, 1, 1, 2])
    corr_ee, corr_ii, corr_ei = pairwise_spike_count_corr(
        spikes=spikes,
        T=30.0,
        bin_ms=10.0,
        N_E=2,
        N_I=1,
    )
    assert corr_ee > 0.9
    assert corr_ii == pytest.approx(1.0)
    assert corr_ei < -0.9


def test_base_metrics_smoke(tmp_path):
    spikes = make_spikes([5, 15, 35, 6, 16, 36], [0, 0, 0, 1, 1, 1])
    instruments = InstrumentsResults(
        times=np.arange(10, dtype=float),
        neuron_ids=np.array([0]),
        g_e_mean_E=np.ones(10),
        g_i_mean_E=np.ones(10) * 2.0,
    )
    run_result = NetworkResult(spikes=spikes, instruments=instruments)

    class Config:
        class Base:
            T = 100.0
            dt = 1.0
            N_E = 1
            N_I = 1
        base = Base()

    metrics = base_metrics(Config(), run_result, tmp_path, label="test")
    assert "mean_rate_E" in metrics
    assert "energy_total" in metrics
    assert "regime" in metrics
    assert (tmp_path / "metrics_test.yaml").exists()
