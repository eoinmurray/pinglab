from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from pinglab.io import compile_graph_to_runtime
from pinglab.service import build_weights_preview as build_weights_preview_service
from pinglab.service import run_simulation as run_simulation_service


class ConfigOverrides(BaseModel):
    model_config = ConfigDict(extra="ignore")

    pulse_onset_ms: float | None = None
    pulse_duration_ms: float | None = None
    pulse_interval_ms: float | None = None
    pulse_amplitude_E: float | None = None
    pulse_amplitude_I: float | None = None
    dt: float | None = None
    T: float | None = None
    N_E: int | None = None
    N_I: int | None = None
    seed: int | None = None
    neuron_model: Literal["lif", "mqif"] | None = None
    delay_ei: float | None = None
    delay_ie: float | None = None
    delay_ee: float | None = None
    delay_ii: float | None = None
    V_init: float | None = None
    E_L: float | None = None
    E_e: float | None = None
    E_i: float | None = None
    C_m_E: float | None = None
    g_L_E: float | None = None
    C_m_I: float | None = None
    g_L_I: float | None = None
    V_th: float | None = None
    V_reset: float | None = None
    t_ref_E: float | None = None
    t_ref_I: float | None = None
    tau_ampa: float | None = None
    tau_gaba: float | None = None
    mqif_a: list[float] | None = None
    mqif_Vr: list[float] | None = None
    mqif_w_a: list[float] | None = None
    mqif_w_Vr: list[float] | None = None
    mqif_w_tau: list[float] | None = None
    g_L_heterogeneity_sd: float | None = None
    C_m_heterogeneity_sd: float | None = None
    V_th_heterogeneity_sd: float | None = None
    t_ref_heterogeneity_sd: float | None = None


class InputsSpec(BaseModel):
    model_config = ConfigDict(extra="ignore")

    input_type: Literal["ramp", "sine", "external_spike_train", "pulse", "pulses"] = "ramp"
    input_population: Literal["all", "e", "i"] | None = None
    I_E_start: float | None = None
    I_E_end: float | None = None
    I_I_start: float | None = None
    I_I_end: float | None = None
    I_E_base: float | None = None
    I_I_base: float | None = None
    noise_std: float | None = None
    noise_std_E: float | None = None
    noise_std_I: float | None = None
    seed: int | None = None
    sine_freq_hz: float | None = None
    sine_amp: float | None = None
    sine_y_offset: float | None = None
    sine_phase: float | None = None
    sine_phase_offset_i: float | None = None
    lambda0_hz: float | None = None
    mod_depth: float | None = None
    envelope_freq_hz: float | None = None
    phase_rad: float | None = None
    w_in: float | None = None
    tau_in_ms: float | None = None
    pulse_t_ms: float | None = None
    pulse_width_ms: float | None = None
    pulse_interval_ms: float | None = None
    pulse_amp_E: float | None = None
    pulse_amp_I: float | None = None
    targeted_subset_enabled: bool | None = None
    target_population: Literal["all", "e", "i"] | None = None
    target_strategy: Literal["random", "first"] | None = None
    target_fraction: float | None = None
    target_seed: int | None = None


class WeightsSpec(BaseModel):
    model_config = ConfigDict(extra="ignore")

    ee: dict | None = None
    ei: dict | None = None
    ie: dict | None = None
    ii: dict | None = None
    clamp_min: float | None = None
    seed: int | None = None


class RunRequest(BaseModel):
    graph: dict | None = None
    config: ConfigOverrides | None = None
    inputs: InputsSpec | None = None
    weights: WeightsSpec | None = None
    performance_mode: bool = True
    max_spikes: int | None = Field(default=30000, ge=1000)
    burn_in_ms: float | None = Field(default=None, ge=0.0)


class SpikesResponse(BaseModel):
    times: list[float]
    ids: list[int]
    types: list[int]


class RunResponse(BaseModel):
    spikes: SpikesResponse
    core_sim_ms: float
    runtime_ms: float
    num_steps: int
    num_spikes: int
    total_e_spikes: int = 0
    total_i_spikes: int = 0
    spikes_truncated: bool = False
    mean_rate_E: float
    mean_rate_I: float
    isi_cv_E: float
    isi_mean_E_ms: float = 0.0
    isi_inverse_E_hz: float = 0.0
    autocorr_peak: float
    xcorr_peak: float
    coherence_peak: float
    lagged_coherence: float
    population_rate_t_ms: list[float]
    population_rate_hz_E: list[float]
    population_rate_hz_I: list[float]
    population_rate_hz_layers: list[list[float]] = []
    decode_lowpass_hz_E: list[float] = []
    decode_lowpass_hz_I: list[float] = []
    decode_lowpass_hz_layers: list[list[float]] = []
    membrane_t_ms: list[float]
    membrane_V_E: list[float]
    membrane_V_I: list[float]
    membrane_V_layers: list[list[float]] = []
    membrane_g_e_E: list[float]
    membrane_g_i_E: list[float]
    membrane_g_e_I: list[float]
    membrane_g_i_I: list[float]
    autocorr_lags_ms: list[float]
    autocorr_corr: list[float]
    autocorr_lags_layers_ms: list[list[float]] = []
    autocorr_corr_layers: list[list[float]] = []
    xcorr_lags_ms: list[float]
    xcorr_corr: list[float]
    xcorr_lags_layers_ms: list[list[float]] = []
    xcorr_corr_layers: list[list[float]] = []
    coherence_lags_ms: list[float]
    coherence_corr: list[float]
    weights_hist_bins: list[float]
    weights_hist_counts_ee: list[float]
    weights_hist_counts_ei: list[float]
    weights_hist_counts_ie: list[float]
    weights_hist_counts_ii: list[float]
    weights_hist_blocks_ee: list[list[float]] = []
    weights_hist_blocks_ei: list[list[float]] = []
    weights_hist_blocks_ie: list[list[float]] = []
    weights_hist_blocks_ii: list[list[float]] = []
    weights_heatmap: list[list[float]]
    psd_freqs_hz: list[float]
    psd_power: list[float]
    psd_power_layers: list[list[float]] = []
    input_t_ms: list[float]
    input_mean_E: list[float]
    input_mean_I: list[float]
    input_mean_layers: list[list[float]] = []
    input_raw_spike_fraction_E: list[float] = []
    input_raw_spike_fraction_I: list[float] = []
    input_raw_spike_fraction_layers: list[list[float]] = []
    input_raw_raster_times_ms_layers: list[list[float]] = []
    input_raw_raster_ids_layers: list[list[int]] = []
    input_envelope_hz: list[float] = []
    input_envelope_hz_layers: list[list[float]] = []
    input_spike_fraction_E: list[float] = []
    input_spike_fraction_I: list[float] = []
    input_spike_fraction_layers: list[list[float]] = []
    decode_envelope_hz: list[float] = []
    decode_envelope_hz_layers: list[list[float]] = []
    decode_corr: float = 0.0
    decode_rmse: float = 0.0
    decode_corr_layers: list[float] = []
    decode_rmse_layers: list[float] = []
    psd_peak_freq_hz: float = 0.0
    psd_peak_bandwidth_hz: float = 0.0
    psd_peak_q_factor: float = 0.0
    layer_labels: list[str] = []
    input_prep_ms: float = 0.0
    weights_build_ms: float = 0.0
    analysis_ms: float = 0.0
    response_build_ms: float = 0.0
    server_compute_ms: float = 0.0


class WeightsRequest(BaseModel):
    graph: dict | None = None
    config: ConfigOverrides | None = None
    weights: WeightsSpec | None = None


class WeightsResponse(BaseModel):
    weights_hist_bins: list[float]
    weights_hist_counts_ee: list[float]
    weights_hist_counts_ei: list[float]
    weights_hist_counts_ie: list[float]
    weights_hist_counts_ii: list[float]
    weights_hist_blocks_ee: list[list[float]] = []
    weights_hist_blocks_ei: list[list[float]] = []
    weights_hist_blocks_ie: list[list[float]] = []
    weights_hist_blocks_ii: list[list[float]] = []
    weights_heatmap: list[list[float]]


def build_weights_preview(request: WeightsRequest | None = None) -> WeightsResponse:
    graph_payload = request.graph if request else None
    runtime_overrides = None
    if graph_payload is not None:
        runtime_overrides = compile_graph_to_runtime(graph_payload)
    config_overrides = (
        request.config.model_dump(exclude_none=True) if request and request.config else None
    )
    weights_overrides = (
        request.weights.model_dump(exclude_none=True) if request and request.weights else None
    )
    payload = build_weights_preview_service(
        config_overrides=config_overrides,
        weights_overrides=weights_overrides,
        runtime_overrides=runtime_overrides,
    )
    return WeightsResponse.model_validate(payload)


def run_simulation(request: RunRequest | None = None) -> RunResponse:
    request = request or RunRequest()
    runtime_overrides = None
    if request.graph is not None:
        runtime_overrides = compile_graph_to_runtime(request.graph)
    config_overrides = request.config.model_dump(exclude_none=True) if request.config else None
    inputs_overrides = request.inputs.model_dump(exclude_none=True) if request.inputs else None
    weights_overrides = request.weights.model_dump(exclude_none=True) if request.weights else None
    payload = run_simulation_service(
        config_overrides=config_overrides,
        inputs_overrides=inputs_overrides,
        weights_overrides=weights_overrides,
        runtime_overrides=runtime_overrides,
        performance_mode=request.performance_mode,
        max_spikes=request.max_spikes,
        burn_in_ms=request.burn_in_ms,
    )
    return RunResponse.model_validate(payload)


def run_timing_headers(
    result: RunResponse,
    *,
    serialize_ms: float | None = None,
    response_bytes: int | None = None,
) -> dict[str, str]:
    headers = {
        "X-Pinglab-Core-Sim-Ms": f"{result.core_sim_ms:.3f}",
        "X-Pinglab-Input-Prep-Ms": f"{result.input_prep_ms:.3f}",
        "X-Pinglab-Weights-Build-Ms": f"{result.weights_build_ms:.3f}",
        "X-Pinglab-Analysis-Ms": f"{result.analysis_ms:.3f}",
        "X-Pinglab-Response-Build-Ms": f"{result.response_build_ms:.3f}",
        "X-Pinglab-Server-Compute-Ms": f"{result.server_compute_ms:.3f}",
    }
    if serialize_ms is not None:
        headers["X-Pinglab-Serialize-Ms"] = f"{serialize_ms:.3f}"
    if response_bytes is not None:
        headers["X-Pinglab-Response-Bytes"] = str(response_bytes)
    return headers
