
from pathlib import Path
import numpy as np

from pinglab.types import NetworkResult
import yaml

from pinglab.analysis import (
    mean_firing_rates,
    population_isi_cv,
    pairwise_spike_count_corr,
    ei_lag_stats,
    gamma_metrics,
    population_fano_factor,
    synchrony_index,
    conductance_stats,
    energy_metrics,
    calculate_regime_label,
)


def base_metrics(
    config,
    run_result: NetworkResult,
    data_path: Path,
    label: str = "regime",
) -> dict[str, float | int | str | None]:
  """
    Unified metrics computation for PING network simulations.

    This module provides the `base_metrics` function, which computes a comprehensive
    set of diagnostics from a single simulation run. These metrics are designed to:

    1. **Characterize dynamical regime** — Distinguish asynchronous irregular (AI)
    from oscillatory (PING) states using multiple redundant indicators.

    2. **Enable rate-matched comparisons** — All subsequent experiments compare
    conditions at matched mean firing rates, isolating timing and structure effects.

    3. **Enable energy-matched comparisons** — Conductance integrals and spike counts
    provide energy proxies for efficiency analysis.

    Metrics computed
    ----------------
    **Firing rates:**
        - `mean_rate_E`, `mean_rate_I`: Population mean firing rates (Hz).

    **ISI variability:**
        - `cv_pop_E`, `cv_pop_I`: CV of pooled inter-spike intervals.
        - `cv_mean_E`, `cv_mean_I`: Mean CV across individual neurons.

    **Pairwise correlations:**
        - `corr_EE_mean`, `corr_II_mean`, `corr_EI_mean`: Mean spike-count correlations.

    **E→I timing:**
        - `lag_EI_mean_ms`, `lag_EI_std_ms`: Mean and std of E→I spike lag.

    **Spectral (gamma):**
        - `gamma_peak_freq`: Peak frequency in gamma band (Hz).
        - `gamma_peak_power`: Power at peak.
        - `gamma_Q`: Quality factor (sharpness).

    **Trial variability:**
        - `fano_E`, `fano_I`: Fano factor of spike counts.
        - `synchrony`: Population synchrony index.

    **Conductance:**
        - `g_e_mean`, `g_i_mean`: Mean excitatory/inhibitory conductances.
        - `g_ei_ratio`: E/I conductance ratio.
        - `g_e_cv`, `g_i_cv`: Temporal variability of conductances.

    **Energy:**
        - `energy_conductance`: Time-integrated total conductance.
        - `energy_spikes`: Total spike count.
        - `energy_total`: Combined energy proxy.
        - `energy_per_spike`: Efficiency (spikes per unit energy).

    **Regime:**
        - `regime`: Classification label (e.g., 'AI', 'PING', 'silent').
        - `regime_reason`: Human-readable explanation of classification.

    Requirements
    ------------
    The simulation must be run with `instruments` configured to record conductances.
    Either `population_means=True` (recommended) or per-neuron `g_e`/`g_i` traces.

    Example config.yaml:
        base:
        instruments:
            variables: [V, g_e, g_i]
            population_means: true
            downsample: 10
    """

  # Convenience aliases
  T = float(config.base.T)
  dt = float(config.base.dt)
  N_E = int(config.base.N_E)
  N_I = int(config.base.N_I)

  metrics: dict[str, float | int | str | None] = {}

  # === Mean firing rates ===
  mean_rate_E, mean_rate_I = mean_firing_rates(
      run_result.spikes,
      N_E=N_E,
      N_I=N_I,
  )
  metrics["mean_rate_E"] = mean_rate_E
  metrics["mean_rate_I"] = mean_rate_I

  # === ISI CV ===
  E_ids = np.arange(N_E)
  I_ids = np.arange(N_E, N_E + N_I)

  cv_pop_E, cv_E = population_isi_cv(run_result.spikes, neuron_ids=E_ids)
  cv_pop_I, cv_I = population_isi_cv(run_result.spikes, neuron_ids=I_ids)
  metrics["cv_pop_E"] = cv_pop_E
  metrics["cv_pop_I"] = cv_pop_I
  metrics["cv_mean_E"] = float(np.nanmean(cv_E)) if cv_E.size > 0 else float("nan")
  metrics["cv_mean_I"] = float(np.nanmean(cv_I)) if cv_I.size > 0 else float("nan")

  # === Pairwise spike-count correlations ===
  corr_EE, corr_II, corr_EI = pairwise_spike_count_corr(
      spikes=run_result.spikes,
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
      spikes=run_result.spikes,
      N_E=N_E,
      max_lag_ms=10.0,
  )
  metrics["lag_EI_mean_ms"] = lag_mean_ms
  metrics["lag_EI_std_ms"] = lag_std_ms

  # === Gamma-band metrics ===
  fs = 1000.0  # Hz, bin = 1 ms
  gamma_peak_freq, gamma_peak_power, gamma_Q = gamma_metrics(
      spikes=run_result.spikes,
      T=T,
      fs=fs,
      fmin=30.0,
      fmax=90.0,
      data_path=data_path / f"psd_{label}.png",
  )
  metrics["gamma_peak_freq"] = gamma_peak_freq
  metrics["gamma_peak_power"] = gamma_peak_power
  metrics["gamma_Q"] = gamma_Q

  # === Fano factors ===
  fano_E, fano_I = population_fano_factor(
      spikes=run_result.spikes,
      T=T,
      N_E=N_E,
      N_I=N_I,
      window_ms=50.0,
  )
  metrics["fano_E"] = fano_E
  metrics["fano_I"] = fano_I

  # === Synchrony index ===
  si = synchrony_index(
      spikes=run_result.spikes,
      T=T,
      bin_ms=5.0,
      N_E=N_E,
      N_I=N_I,
  )
  metrics["synchrony"] = si

  # === Conductance statistics ===
  # Try population means first (more efficient), fall back to per-neuron traces
  g_e_mean_E = getattr(run_result.instruments, "g_e_mean_E", None)
  g_i_mean_E = getattr(run_result.instruments, "g_i_mean_E", None)

  if g_e_mean_E is not None and g_i_mean_E is not None:
      # Use population mean traces (1D arrays)
      g_e_traces = g_e_mean_E
      g_i_traces = g_i_mean_E
  else:
      # Fall back to per-neuron traces
      g_e_traces = getattr(run_result.instruments, "g_e", None)
      g_i_traces = getattr(run_result.instruments, "g_i", None)

  if g_e_traces is None or g_i_traces is None:
      raise RuntimeError("Conductance traces not available; skipping conductance statistics and regime classification.")

  g_e_mean, g_i_mean, g_ei_ratio, g_e_cv, g_i_cv = conductance_stats(
      g_e=g_e_traces,
      g_i=g_i_traces,
  )
  metrics["g_e_mean"] = g_e_mean
  metrics["g_i_mean"] = g_i_mean
  metrics["g_ei_ratio"] = g_ei_ratio
  metrics["g_e_cv"] = g_e_cv
  metrics["g_i_cv"] = g_i_cv

  # === Energy metrics ===
  energy_cond, energy_spk, energy_tot, energy_eff = energy_metrics(
      g_e=g_e_traces,
      g_i=g_i_traces,
      spikes=run_result.spikes,
      dt=dt,
      N_E=N_E,
      N_I=N_I,
  )
  metrics["energy_conductance"] = energy_cond
  metrics["energy_spikes"] = energy_spk
  metrics["energy_total"] = energy_tot
  metrics["energy_per_spike"] = energy_eff

  # === Regime classification ===
  # Use E CV distribution as the main cv_per_neuron input
  cv_per_neuron = cv_E if cv_E.size > 0 else cv_I

  regime, reason = calculate_regime_label(
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
  metrics["regime_reason"] = reason

  # Dump metrics to YAML for later analysis
  # Convert numpy types to native Python types for YAML serialization
  def to_python(v):
      if v is None:
          return None
      if isinstance(v, (np.floating, np.integer)):
          return float(v) if isinstance(v, np.floating) else int(v)
      return v

  metrics_clean = {k: to_python(v) for k, v in metrics.items()}
  metrics_path = data_path / f"metrics_{label}.yaml"
  with metrics_path.open("w") as f:
      yaml.safe_dump(metrics_clean, f)

  return metrics_clean
