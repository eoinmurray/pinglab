
from pathlib import Path
import numpy as np
from pinglab.inputs.tonic import tonic
from pinglab.run.run_network import run_network
from pinglab.types import NetworkResult, InstrumentsConfig
import yaml

from pinglab.plots.raster import save_raster
from pinglab.analysis import (
    population_mean_rate,
    population_isi_cv,
    pairwise_spike_count_corr,
    ei_lag_stats,
    gamma_metrics,
    population_fano_factor,
    synchrony_index,
    conductance_stats,
    calculate_regime,
)


def run_regime(
      config,
      data_path: Path,
      label: str = "regime",
  ) -> None:
    # Convenience aliases
    T = float(config.base.T)
    dt = float(config.base.dt)
    N_E = int(config.base.N_E)
    N_I = int(config.base.N_I)

    external_input = tonic(
        N_E=N_E,
        N_I=N_I,
        I_E=config.default_inputs.I_E,
        I_I=config.default_inputs.I_I,
        noise_std=config.default_inputs.noise,
        num_steps=int(np.ceil(T / dt)),
        seed=config.base.seed if config.base.seed is not None else 0,
    )

    # Enable conductance recording for all neurons
    instruments_config = InstrumentsConfig(
        variables=["g_e", "g_i"],
        all_neurons=True,
    )
    run_cfg = config.base.model_copy(update={"instruments": instruments_config})

    result: NetworkResult = run_network(run_cfg, external_input=external_input)
    save_raster(result.spikes, data_path / f"{label}_raster.png")

    metrics: dict[str, float | int | str | None] = {}

    # === Mean firing rates ===
    mean_rate_E, mean_rate_I = population_mean_rate(
        result.spikes,
        T=T,
        N_E=N_E,
        N_I=N_I,
    )
    metrics["mean_rate_E"] = mean_rate_E
    metrics["mean_rate_I"] = mean_rate_I

    # === ISI CV ===
    E_ids = np.arange(N_E)
    I_ids = np.arange(N_E, N_E + N_I)

    cv_pop_E, cv_E = population_isi_cv(result.spikes, neuron_ids=E_ids)
    cv_pop_I, cv_I = population_isi_cv(result.spikes, neuron_ids=I_ids)
    metrics["cv_pop_E"] = cv_pop_E
    metrics["cv_pop_I"] = cv_pop_I
    metrics["cv_mean_E"] = float(np.nanmean(cv_E)) if cv_E.size > 0 else float("nan")
    metrics["cv_mean_I"] = float(np.nanmean(cv_I)) if cv_I.size > 0 else float("nan")

    # === Pairwise spike-count correlations ===
    corr_EE, corr_II, corr_EI = pairwise_spike_count_corr(
        spikes=result.spikes,
        T=T,
        bin_ms=5.0,
        N_E=N_E,
        N_I=N_I,
    )
    metrics["corr_EE_mean"] = corr_EE
    metrics["corr_II_mean"] = corr_II
    metrics["corr_EI_mean"] = corr_EI

    # === E→I lag stats ===
    lag_mean_ms, lag_std_ms = ei_lag_stats(
        spikes=result.spikes,
        N_E=N_E,
        max_lag_ms=10.0,
    )
    metrics["lag_EI_mean_ms"] = lag_mean_ms
    metrics["lag_EI_std_ms"] = lag_std_ms

    # === Gamma-band metrics ===
    fs = 1000.0  # Hz, bin = 1 ms
    gamma_peak_freq, gamma_peak_power, gamma_Q = gamma_metrics(
        spikes=result.spikes,
        T=T,
        fs=fs,
        fmin=30.0,
        fmax=90.0,
        data_path=data_path / f"{label}_psd.png",
    )
    metrics["gamma_peak_freq"] = gamma_peak_freq
    metrics["gamma_peak_power"] = gamma_peak_power
    metrics["gamma_Q"] = gamma_Q

    # === Fano factors ===
    fano_E, fano_I = population_fano_factor(
        spikes=result.spikes,
        T=T,
        N_E=N_E,
        N_I=N_I,
        window_ms=50.0,
    )
    metrics["fano_E"] = fano_E
    metrics["fano_I"] = fano_I

    # === Synchrony index ===
    si = synchrony_index(
        spikes=result.spikes,
        T=T,
        bin_ms=5.0,
        N_E=N_E,
        N_I=N_I,
    )
    metrics["synchrony"] = si

    # === Conductance statistics ===
    if result.instruments is None:
        print("Conductance traces not available; skipping conductance statistics and regime classification.")
        return

    g_e_traces = getattr(result.instruments, "g_e", None)
    g_i_traces = getattr(result.instruments, "g_i", None)

    if g_e_traces is None or g_i_traces is None:
        print("Conductance traces not available; skipping conductance statistics and regime classification.")
        return

    g_e_mean, g_i_mean, g_ei_ratio, g_e_cv, g_i_cv = conductance_stats(
        g_e=g_e_traces,
        g_i=g_i_traces,
    )
    metrics["g_e_mean"] = g_e_mean
    metrics["g_i_mean"] = g_i_mean
    metrics["g_ei_ratio"] = g_ei_ratio
    metrics["g_e_cv"] = g_e_cv
    metrics["g_i_cv"] = g_i_cv

    # === Regime classification ===
    # Use E CV distribution as the main cv_per_neuron input
    cv_per_neuron = cv_E if cv_E.size > 0 else cv_I

    regime = calculate_regime(
        mean_rate_E=mean_rate_E,
        mean_rate_I=mean_rate_I,
        cv_population=float(cv_pop_E),
        cv_per_neuron=cv_per_neuron,
        corr_EE_mean=corr_EE,
        corr_II_mean=corr_II,
        corr_EI_mean=corr_EI,
        lag_mean_ms=lag_mean_ms,
        lag_std_ms=lag_std_ms,
        gamma_peak_freq=gamma_peak_freq,
        gamma_peak_power=gamma_peak_power,
        gamma_Q=gamma_Q,
        fano_E=fano_E,
        fano_I=fano_I,
        synchrony=si,
        g_e_mean=g_e_mean,
        g_i_mean=g_i_mean,
        g_ei_ratio=g_ei_ratio,
        g_e_cv=g_e_cv,
        g_i_cv=g_i_cv,
    )

    metrics["regime"] = regime

    # Dump metrics to YAML for later analysis
    # Convert numpy types to native Python types for YAML serialization
    def to_python(v):
        if v is None:
            return None
        if isinstance(v, (np.floating, np.integer)):
            return float(v) if isinstance(v, np.floating) else int(v)
        return v

    metrics_clean = {k: to_python(v) for k, v in metrics.items()}
    metrics_path = data_path / f"{label}_metrics.yaml"
    with metrics_path.open("w") as f:
        yaml.safe_dump(metrics_clean, f)
