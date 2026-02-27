---
title: docs.4-api
description: API contract for the simulator backend
---

# API

This is the real contract implemented in `src/api/src/api/web_app.py`.

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/run` | Run simulation with default request |
| `POST` | `/run` | Run simulation with request payload |
| `GET` | `/weights` | Build weight preview with default request |
| `POST` | `/weights` | Build weight preview with request payload |

All routes support:

- JSON responses (`application/json`)
- Arrow IPC stream responses (`application/vnd.apache.arrow.stream`) when requested via `Accept`

## Run it

From repo root:

```bash
task api:dev
```

Equivalent command:

```bash
uv run --package pinglab-api uvicorn api.api:app --reload
```

Default local base URL is `http://localhost:8000`.

## Content negotiation

If the request `Accept` header contains `application/vnd.apache.arrow.stream`, the API returns a one-row Arrow IPC stream. Otherwise it returns JSON.

Examples:

```bash
# JSON
curl -s http://localhost:8000/run \
  -H 'Accept: application/json'

# Arrow
curl -s http://localhost:8000/run \
  -H 'Accept: application/vnd.apache.arrow.stream' \
  -o run.arrow
```

## `/run` request

Top-level request shape:

```json
{
  "config": {"...": "optional config overrides"},
  "inputs": {"...": "optional input overrides"},
  "weights": {"...": "optional weight overrides"},
  "performance_mode": true,
  "max_spikes": 30000,
  "burn_in_ms": null
}
```

Top-level keys:

- `config`
- `inputs`
- `weights`
- `performance_mode`: bool (default `true`)
- `max_spikes`: int or null, minimum `1000` when set (default `30000`)
- `burn_in_ms`: float or null, minimum `0.0` when set (default `null`)

`config` override keys:

```text
C_m_E, C_m_I, C_m_heterogeneity_sd, E_K, E_L, E_Na, E_e, E_i,
N_E, N_I, T, V_init, V_reset, V_th, V_th_heterogeneity_sd,
adex_V_T, adex_V_peak, adex_a, adex_b, adex_delta_T, adex_tau_w,
delay_ee, delay_ei, delay_ie, delay_ii, dt,
fhn_a, fhn_b, fhn_tau_w,
g_A, g_K, g_L_E, g_L_I, g_L_heterogeneity_sd, g_Na,
izh_a, izh_b, izh_c, izh_d,
mqif_Vr, mqif_a, mqif_w_Vr, mqif_w_a, mqif_w_tau,
neuron_model,
pulse_amplitude_E, pulse_amplitude_I, pulse_duration_ms, pulse_interval_ms, pulse_onset_ms,
qif_Vr, qif_Vt, qif_a,
seed,
t_ref_E, t_ref_I, t_ref_heterogeneity_sd,
tau_ampa, tau_gaba
```

`inputs` override keys:

```text
I_E_base, I_E_end, I_E_start, I_I_base, I_I_end, I_I_start,
envelope_freq_hz, lambda0_hz, mod_depth, phase_rad,
input_population, input_type,
noise_std, noise_std_E, noise_std_I,
pulse_amp_E, pulse_amp_I, pulse_interval_ms, pulse_t_ms, pulse_width_ms,
seed,
sine_amp, sine_freq_hz, sine_phase, sine_phase_offset_i, sine_y_offset,
w_in, tau_in_ms,
target_fraction, target_population, target_seed, target_strategy, targeted_subset_enabled
```

`weights` override keys:

```text
ee, ei, ie, ii, clamp_min, seed
```

## `/run` response

`/run` returns spikes, summary metrics, traces, spectra, histogram artifacts, readout metrics, and timing fields.

Core fields include:

- Spikes and counts: `spikes`, `num_steps`, `num_spikes`, `spikes_truncated`
- Main metrics: `mean_rate_E`, `mean_rate_I`, `isi_cv_E`, `autocorr_peak`, `xcorr_peak`, `coherence_peak`, `lagged_coherence`
- Trace arrays: population rates, membrane traces, autocorr/xcorr/coherence curves
- Weight artifacts: histogram bins/counts/blocks and `weights_heatmap`
- Spectral fields: PSD arrays and peak summaries
- Timing fields: `core_sim_ms`, `runtime_ms`, `input_prep_ms`, `weights_build_ms`, `analysis_ms`, `response_build_ms`, `server_compute_ms`

If you need the exact full field list, check `RunResponse` in `src/api/src/api/functions.py`.

## `/run` timing headers

`/run` responses include timing headers:

- `X-Pinglab-Core-Sim-Ms`
- `X-Pinglab-Input-Prep-Ms`
- `X-Pinglab-Weights-Build-Ms`
- `X-Pinglab-Analysis-Ms`
- `X-Pinglab-Response-Build-Ms`
- `X-Pinglab-Server-Compute-Ms`
- `X-Pinglab-Serialize-Ms`
- `X-Pinglab-Response-Bytes`

## `/weights` request

Top-level request shape:

```json
{
  "config": {"...": "optional config overrides"},
  "weights": {"...": "optional weight overrides"}
}
```

`config` and `weights` use the same override keys listed above.

## `/weights` response

`/weights` returns:

- `weights_hist_bins`
- `weights_hist_counts_ee`
- `weights_hist_counts_ei`
- `weights_hist_counts_ie`
- `weights_hist_counts_ii`
- `weights_hist_blocks_ee`
- `weights_hist_blocks_ei`
- `weights_hist_blocks_ie`
- `weights_hist_blocks_ii`
- `weights_heatmap`

## Example requests

```bash
curl -s http://localhost:8000/run \
  -X POST \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json' \
  -d '{
    "config": {"T": 800, "dt": 0.1, "N_E": 80, "N_I": 20, "seed": 1},
    "inputs": {
      "input_type": "external_spike_train",
      "input_population": "e",
      "lambda0_hz": 30,
      "mod_depth": 0.7,
      "envelope_freq_hz": 5,
      "phase_rad": 0.0,
      "w_in": 0.25,
      "tau_in_ms": 3.0
    },
    "performance_mode": true,
    "max_spikes": 30000
  }'
```

```bash
curl -s http://localhost:8000/weights \
  -X POST \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json' \
  -d '{
    "config": {"N_E": 80, "N_I": 20, "seed": 1},
    "weights": {"seed": 1, "clamp_min": 0.0}
  }'
```

## Errors and validation

- Unknown keys inside `config`, `inputs`, and `weights` are ignored.
- Invalid enum values or type mismatches return `422 Unprocessable Entity`.
- If no body is sent to `POST /run` or `POST /weights`, defaults are used.

## Modal deployment

Deploy the same API app to Modal:

```bash
task api:deploy
```

Local and Modal use the same route contract.
