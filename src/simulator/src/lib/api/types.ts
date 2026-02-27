export type SpikesResponse = {
  times: number[]
  ids: number[]
  types: number[]
}

export type RunResponse = {
  spikes: SpikesResponse
  core_sim_ms: number
  runtime_ms: number
  num_steps: number
  num_spikes: number
  total_e_spikes: number
  total_i_spikes: number
  spikes_truncated?: boolean
  mean_rate_E: number
  mean_rate_I: number
  isi_cv_E: number
  isi_mean_E_ms: number
  isi_inverse_E_hz: number
  autocorr_peak: number
  xcorr_peak: number
  coherence_peak: number
  lagged_coherence: number
  population_rate_t_ms: number[]
  population_rate_hz_E: number[]
  population_rate_hz_I: number[]
  population_rate_hz_layers: number[][]
  decode_lowpass_hz_E: number[]
  decode_lowpass_hz_I: number[]
  decode_lowpass_hz_layers: number[][]
  membrane_t_ms: number[]
  membrane_V_E: number[]
  membrane_V_I: number[]
  membrane_V_layers: number[][]
  membrane_g_e_E: number[]
  membrane_g_i_E: number[]
  membrane_g_e_I: number[]
  membrane_g_i_I: number[]
  autocorr_lags_ms: number[]
  autocorr_corr: number[]
  autocorr_lags_layers_ms: number[][]
  autocorr_corr_layers: number[][]
  xcorr_lags_ms: number[]
  xcorr_corr: number[]
  xcorr_lags_layers_ms: number[][]
  xcorr_corr_layers: number[][]
  coherence_lags_ms: number[]
  coherence_corr: number[]
  weights_hist_bins: number[]
  weights_hist_counts_ee: number[]
  weights_hist_counts_ei: number[]
  weights_hist_counts_ie: number[]
  weights_hist_counts_ii: number[]
  weights_hist_blocks_ee: number[][]
  weights_hist_blocks_ei: number[][]
  weights_hist_blocks_ie: number[][]
  weights_hist_blocks_ii: number[][]
  weights_heatmap: number[][]
  psd_freqs_hz: number[]
  psd_power: number[]
  psd_power_layers: number[][]
  input_t_ms: number[]
  input_mean_E: number[]
  input_mean_I: number[]
  input_mean_layers: number[][]
  input_raw_spike_fraction_E: number[]
  input_raw_spike_fraction_I: number[]
  input_raw_spike_fraction_layers: number[][]
  input_raw_raster_times_ms_layers: number[][]
  input_raw_raster_ids_layers: number[][]
  input_envelope_hz: number[]
  input_envelope_hz_layers: number[][]
  input_spike_fraction_E: number[]
  input_spike_fraction_I: number[]
  input_spike_fraction_layers: number[][]
  decode_envelope_hz: number[]
  decode_envelope_hz_layers: number[][]
  decode_corr: number
  decode_rmse: number
  decode_corr_layers: number[]
  decode_rmse_layers: number[]
  psd_peak_freq_hz: number
  psd_peak_bandwidth_hz: number
  psd_peak_q_factor: number
  layer_labels: string[]
  input_prep_ms: number
  weights_build_ms: number
  analysis_ms: number
  response_build_ms: number
  server_compute_ms: number
}

export type WeightsResponse = {
  weights_hist_bins: number[]
  weights_hist_counts_ee: number[]
  weights_hist_counts_ei: number[]
  weights_hist_counts_ie: number[]
  weights_hist_counts_ii: number[]
  weights_hist_blocks_ee: number[][]
  weights_hist_blocks_ei: number[][]
  weights_hist_blocks_ie: number[][]
  weights_hist_blocks_ii: number[][]
  weights_heatmap: number[][]
}

export type RunRequest = {
  graph: Record<string, unknown>
}

export type WeightsRequest = {
  graph: Record<string, unknown>
}

export type RunTimingHeaders = {
  coreSimMs: number | null
  inputPrepMs: number | null
  weightsBuildMs: number | null
  analysisMs: number | null
  responseBuildMs: number | null
  serverComputeMs: number | null
  serializeMs: number | null
  responseBytes: number | null
}
