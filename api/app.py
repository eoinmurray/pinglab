from __future__ import annotations

import time
from pathlib import Path
from typing import Literal

import numpy as np
import yaml
from fastapi import Body, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from pinglab.analysis import mean_firing_rates, population_rate
from pinglab.analysis.autocorr_peak import autocorr_peak
from pinglab.analysis.coherence import coherence_peak
from pinglab.analysis.mean_pairwise_xcorr_peak import mean_pairwise_xcorr_peak
from pinglab.inputs import add_pulse_to_input, add_pulse_train_to_input, ramp
from pinglab.lib import build_adjacency_matrices
from pinglab.run.neuron_models import build_model_from_config
from pinglab.run.run_network import run_network
from pinglab.types import InstrumentsConfig, NetworkConfig, Spikes
from pinglab.utils.slice_spikes import slice_spikes


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
    neuron_model: Literal[
        "lif",
        "hh",
        "adex",
        "connor_stevens",
        "fitzhugh",
        "mqif",
        "qif",
        "izhikevich",
    ] | None = None
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
    g_Na: float | None = None
    g_K: float | None = None
    E_Na: float | None = None
    E_K: float | None = None
    adex_V_T: float | None = None
    adex_delta_T: float | None = None
    adex_tau_w: float | None = None
    adex_a: float | None = None
    adex_b: float | None = None
    adex_V_peak: float | None = None
    g_A: float | None = None
    fhn_a: float | None = None
    fhn_b: float | None = None
    fhn_tau_w: float | None = None
    t_ref_E: float | None = None
    t_ref_I: float | None = None
    tau_ampa: float | None = None
    tau_gaba: float | None = None
    mqif_a: list[float] | None = None
    mqif_Vr: list[float] | None = None
    mqif_w_a: list[float] | None = None
    mqif_w_Vr: list[float] | None = None
    mqif_w_tau: list[float] | None = None
    qif_a: float | None = None
    qif_Vr: float | None = None
    qif_Vt: float | None = None
    izh_a: float | None = None
    izh_b: float | None = None
    izh_c: float | None = None
    izh_d: float | None = None
    g_L_heterogeneity_sd: float | None = None
    C_m_heterogeneity_sd: float | None = None
    V_th_heterogeneity_sd: float | None = None
    t_ref_heterogeneity_sd: float | None = None


class InputsSpec(BaseModel):
    model_config = ConfigDict(extra="ignore")

    input_type: Literal["ramp", "pulse", "pulses"] = "ramp"
    I_E_start: float | None = None
    I_E_end: float | None = None
    I_I_start: float | None = None
    I_I_end: float | None = None
    I_E_base: float | None = None
    I_I_base: float | None = None
    noise_std: float | None = None
    seed: int | None = None
    pulse_t_ms: float | None = None
    pulse_width_ms: float | None = None
    pulse_interval_ms: float | None = None
    pulse_amp_E: float | None = None
    pulse_amp_I: float | None = None


class WeightsSpec(BaseModel):
    model_config = ConfigDict(extra="ignore")

    ee: dict | None = None
    ei: dict | None = None
    ie: dict | None = None
    ii: dict | None = None
    clamp_min: float | None = None
    seed: int | None = None


class RunRequest(BaseModel):
    config: ConfigOverrides | None = None
    inputs: InputsSpec | None = None
    weights: WeightsSpec | None = None
    connectivity_backend: Literal["event", "dense"] = "event"


class SpikesResponse(BaseModel):
    times: list[float]
    ids: list[int]
    types: list[int]


class RunResponse(BaseModel):
    spikes: SpikesResponse
    runtime_ms: float
    num_steps: int
    num_spikes: int
    rhythmicity: float
    mean_rate_E: float
    mean_rate_I: float
    isi_cv_E: float
    population_rate_t_ms: list[float]
    population_rate_hz_E: list[float]
    population_rate_hz_I: list[float]
    membrane_t_ms: list[float]
    membrane_V_E: list[float]
    membrane_V_I: list[float]
    autocorr_lags_ms: list[float]
    autocorr_corr: list[float]
    xcorr_lags_ms: list[float]
    xcorr_corr: list[float]
    coherence_lags_ms: list[float]
    coherence_corr: list[float]
    weights_hist_bins: list[float]
    weights_hist_counts_ee: list[float]
    weights_hist_counts_ei: list[float]
    weights_hist_counts_ie: list[float]
    weights_hist_counts_ii: list[float]
    input_t_ms: list[float]
    input_mean_E: list[float]
    input_mean_I: list[float]


DEFAULT_CONFIG = NetworkConfig(
    dt=0.1,
    T=1000.0,
    N_E=800,
    N_I=200,
    seed=0,
    neuron_model="lif",
    delay_ei=0.5,
    delay_ie=1.2,
    delay_ee=0.5,
    delay_ii=0.5,
    V_init=-65.0,
    E_L=-65.0,
    E_e=0.0,
    E_i=-80.0,
    C_m_E=1.0,
    g_L_E=0.05,
    C_m_I=1.0,
    g_L_I=0.1,
    V_th=-50.0,
    V_reset=-65.0,
    t_ref_E=3.0,
    t_ref_I=1.5,
    tau_ampa=2.0,
    tau_gaba=6.5,
    mqif_a=[0.02],
    mqif_Vr=[-55.0],
    mqif_w_a=[0.02],
    mqif_w_Vr=[-55.0],
    mqif_w_tau=[100.0],
    g_L_heterogeneity_sd=0.15,
    C_m_heterogeneity_sd=0.10,
    V_th_heterogeneity_sd=1.2,
    t_ref_heterogeneity_sd=0.3,
    instruments=InstrumentsConfig(
        variables=[],
        all_neurons=False,
        neuron_ids=[],
    ),
)

DEFAULT_INPUTS = {
    "input_type": "ramp",
    "I_E_start": 0.7,
    "I_E_end": 0.7,
    "I_I_start": 0.7,
    "I_I_end": 0.7,
    "I_E_base": 0.7,
    "I_I_base": 0.7,
    "noise_std": 0.5,
    "seed": 0,
    "pulse_t_ms": 200.0,
    "pulse_width_ms": 20.0,
    "pulse_interval_ms": 100.0,
    "pulse_amp_E": 1.0,
    "pulse_amp_I": 1.0,
}

DEFAULT_WEIGHTS = {
    "ee": {"p": 0.02, "dist": {"name": "normal", "params": {"mean": 0.02, "std": 0.0}}},
    "ei": {"p": 0.18, "dist": {"name": "normal", "params": {"mean": 0.015, "std": 0.0}}},
    "ie": {"p": 0.04, "dist": {"name": "normal", "params": {"mean": 0.015, "std": 0.0}}},
    "ii": {"p": 0.06, "dist": {"name": "normal", "params": {"mean": 0.02, "std": 0.0}}},
    "clamp_min": 0.0,
    "seed": 0,
}


def _weights_histograms(
    weights: WeightMatrices, bins: int = 40
) -> tuple[list[float], list[float], list[float], list[float], list[float]]:
    ee = weights.W_ee.ravel()
    ei = weights.W_ei.ravel()
    ie = weights.W_ie.ravel()
    ii = weights.W_ii.ravel()
    max_val = 0.0
    for arr in (ee, ei, ie, ii):
        if arr.size:
            max_val = max(max_val, float(np.max(arr)))
    if max_val <= 0:
        return [0.0, 1.0], [0.0], [0.0], [0.0], [0.0]

    edges = np.linspace(0.0, max_val, bins + 1)
    centers = ((edges[:-1] + edges[1:]) / 2.0).tolist()

    def _hist(arr: np.ndarray) -> list[float]:
        if arr.size == 0:
            return [0.0] * len(centers)
        counts, _ = np.histogram(arr, bins=edges)
        return counts.astype(float).tolist()

    return centers, _hist(ee), _hist(ei), _hist(ie), _hist(ii)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/configs")
def list_configs() -> dict[str, list[str]]:
    configs = sorted([p.name for p in CONFIG_DIR.glob("*.yaml")])
    return {"configs": configs}


@app.get("/configs/{name}")
def load_config(name: str) -> dict:
    filename = _safe_config_filename(name)
    path = CONFIG_DIR / filename
    if not path.exists():
        return {"error": "Config not found"}
    with path.open("r") as f:
        data = yaml.safe_load(f) or {}
    return {"name": filename, "config": data}


@app.post("/configs/{name}")
def save_config(name: str, payload: dict = Body(...)) -> dict:
    filename = _safe_config_filename(name)
    path = CONFIG_DIR / filename
    with path.open("w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    return {"ok": True, "name": filename}

RHYTHM_BIN_MS = 5.0
RHYTHM_BURN_IN_MS = 200.0
RHYTHM_TAU_MIN_MS = 5.0
RHYTHM_TAU_MAX_MS = 200.0

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def _safe_config_filename(name: str) -> str:
    clean = Path(name).name
    if clean != name:
        raise ValueError("Invalid config name")
    if not clean:
        raise ValueError("Config name cannot be empty")
    if not clean.endswith(".yaml"):
        clean = f"{clean}.yaml"
    if clean.startswith(".") or ".." in clean or "/" in clean or "\\" in clean:
        raise ValueError("Invalid config name")
    return clean


def _autocorr_rhythmicity(
    rate_hz: np.ndarray,
    dt_ms: float,
    tau_min_ms: float,
    tau_max_ms: float,
) -> float:
    if rate_hz.size == 0:
        return 0.0

    mean = float(np.mean(rate_hz))
    std = float(np.std(rate_hz))
    if std == 0.0:
        return 0.0

    x = (rate_hz - mean) / std
    n = x.size
    corr = np.correlate(x, x, mode="full")[n - 1 :]
    norm = np.arange(n, 0, -1, dtype=float)
    C = corr / norm

    lag_min = max(1, int(np.ceil(tau_min_ms / dt_ms)))
    lag_max = min(n - 1, int(np.floor(tau_max_ms / dt_ms)))
    if lag_max < lag_min:
        return 0.0

    window = C[lag_min : lag_max + 1]
    peak_idx = int(np.argmax(window))
    lag_idx = lag_min + peak_idx
    return float(C[lag_idx])


def _isi_cv(spikes: Spikes, N_E: int, N_I: int, pop: Literal["E", "I"] = "E") -> float:
    if spikes.times.size == 0:
        return 0.0
    if pop == "E":
        neuron_ids = np.arange(N_E)
    else:
        neuron_ids = np.arange(N_E, N_E + N_I)
    cvs: list[float] = []
    for nid in neuron_ids:
        t = spikes.times[spikes.ids == nid]
        if t.size < 2:
            continue
        t = np.sort(t)
        isis = np.diff(t)
        if isis.size == 0:
            continue
        mean_isi = float(np.mean(isis))
        if mean_isi <= 0.0:
            continue
        cv = float(np.std(isis) / mean_isi)
        cvs.append(cv)
    if not cvs:
        return 0.0
    return float(np.mean(cvs))


def _deep_merge(base: dict, update: dict) -> dict:
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _merge_defaults(defaults: dict, overrides: BaseModel | None) -> dict:
    if overrides is None:
        return defaults.copy()
    update = overrides.model_dump(exclude_none=True)
    return _deep_merge(defaults, update)


@app.post("/run", response_model=RunResponse)
def run_simulation(request: RunRequest | None = Body(default=None)) -> RunResponse:
    if request is None:
        request = RunRequest()
    config_overrides = {}
    if request.config is not None:
        config_overrides = request.config.model_dump(exclude_none=True)
    config = DEFAULT_CONFIG.model_copy(update=config_overrides)
    neuron_ids: list[int] = []
    if config.N_E > 0:
        neuron_ids.append(0)
    if config.N_I > 0:
        neuron_ids.append(config.N_E)
    config = config.model_copy(
        update={
            "instruments": InstrumentsConfig(
                variables=["V"],
                neuron_ids=neuron_ids,
                all_neurons=False,
            )
        }
    )

    inputs = _merge_defaults(DEFAULT_INPUTS, request.inputs)
    weights = _merge_defaults(DEFAULT_WEIGHTS, request.weights)

    num_steps = int(np.ceil(config.T / config.dt))

    input_seed = inputs["seed"] if inputs["seed"] is not None else config.seed
    weights_seed = weights["seed"] if weights["seed"] is not None else config.seed

    input_type = inputs.get("input_type", "ramp")
    if input_type == "ramp":
        external_input = ramp(
            config.N_E,
            config.N_I,
            inputs["I_E_start"],
            inputs["I_E_end"],
            inputs["I_I_start"],
            inputs["I_I_end"],
            inputs["noise_std"],
            num_steps,
            config.dt,
            int(input_seed) if input_seed is not None else 0,
        )
    else:
        baseline = ramp(
            config.N_E,
            config.N_I,
            inputs["I_E_base"],
            inputs["I_E_base"],
            inputs["I_I_base"],
            inputs["I_I_base"],
            inputs["noise_std"],
            num_steps,
            config.dt,
            int(input_seed) if input_seed is not None else 0,
        )
        e_ids = np.arange(config.N_E)
        i_ids = np.arange(config.N_E, config.N_E + config.N_I)
        pulse_t_ms = inputs["pulse_t_ms"]
        pulse_width_ms = inputs["pulse_width_ms"]
        pulse_amp_E = inputs["pulse_amp_E"]
        pulse_amp_I = inputs["pulse_amp_I"]
        if input_type == "pulse":
            if config.N_E > 0:
                baseline = add_pulse_to_input(
                    baseline,
                    target_neurons=e_ids,
                    pulse_t=pulse_t_ms,
                    pulse_width_ms=pulse_width_ms,
                    pulse_amp=pulse_amp_E,
                    dt=config.dt,
                    num_steps=num_steps,
                )
            if config.N_I > 0:
                baseline = add_pulse_to_input(
                    baseline,
                    target_neurons=i_ids,
                    pulse_t=pulse_t_ms,
                    pulse_width_ms=pulse_width_ms,
                    pulse_amp=pulse_amp_I,
                    dt=config.dt,
                    num_steps=num_steps,
                )
            external_input = baseline
        elif input_type == "pulses":
            if config.N_E > 0:
                baseline = add_pulse_train_to_input(
                    baseline,
                    target_neurons=e_ids,
                    pulse_t=pulse_t_ms,
                    pulse_width_ms=pulse_width_ms,
                    pulse_amp=pulse_amp_E,
                    pulse_interval_ms=inputs["pulse_interval_ms"],
                    dt=config.dt,
                    num_steps=num_steps,
                )
            if config.N_I > 0:
                baseline = add_pulse_train_to_input(
                    baseline,
                    target_neurons=i_ids,
                    pulse_t=pulse_t_ms,
                    pulse_width_ms=pulse_width_ms,
                    pulse_amp=pulse_amp_I,
                    pulse_interval_ms=inputs["pulse_interval_ms"],
                    dt=config.dt,
                    num_steps=num_steps,
                )
            external_input = baseline
        else:
            raise ValueError(f"Unsupported input_type: {input_type}")

    weight_mats = build_adjacency_matrices(
        N_E=config.N_E,
        N_I=config.N_I,
        ee=weights["ee"],
        ei=weights["ei"],
        ie=weights["ie"],
        ii=weights["ii"],
        clamp_min=weights["clamp_min"],
        seed=int(weights_seed) if weights_seed is not None else None,
    )
    (
        weights_hist_bins,
        weights_hist_counts_ee,
        weights_hist_counts_ei,
        weights_hist_counts_ie,
        weights_hist_counts_ii,
    ) = _weights_histograms(weight_mats)

    model = build_model_from_config(config)

    t0 = time.perf_counter()
    result = run_network(
        config,
        external_input,
        model=model,
        weights=weight_mats,
        connectivity_backend=request.connectivity_backend,
    )
    runtime_ms = (time.perf_counter() - t0) * 1000.0

    spikes = result.spikes
    spikes_response = SpikesResponse(
        times=spikes.times.tolist(),
        ids=spikes.ids.tolist(),
        types=(spikes.types.tolist() if spikes.types is not None else []),
    )

    analysis_start = min(RHYTHM_BURN_IN_MS, max(0.0, config.T - config.dt))
    analysis_stop = config.T
    analysis_T = analysis_stop - analysis_start
    rhythmicity = 0.0
    mean_rate_E = 0.0
    mean_rate_I = 0.0
    isi_cv_E = 0.0
    population_rate_t_ms: list[float] = []
    population_rate_hz_E: list[float] = []
    population_rate_hz_I: list[float] = []
    membrane_t_ms: list[float] = []
    membrane_V_E: list[float] = []
    membrane_V_I: list[float] = []
    autocorr_lags_ms: list[float] = []
    autocorr_corr: list[float] = []
    xcorr_lags_ms: list[float] = []
    xcorr_corr: list[float] = []
    coherence_lags_ms: list[float] = []
    coherence_corr: list[float] = []
    input_t_ms: list[float] = []
    input_mean_E: list[float] = []
    input_mean_I: list[float] = []
    if analysis_T > 0:
        sliced = slice_spikes(spikes, analysis_start, analysis_stop)
        shifted = Spikes(
            times=sliced.times - analysis_start,
            ids=sliced.ids,
            types=sliced.types,
        )
        mean_rate_E, mean_rate_I = mean_firing_rates(
            shifted,
            config.N_E,
            config.N_I,
        )
        isi_cv_E = _isi_cv(shifted, config.N_E, config.N_I, pop="E")
        t_ms_rhythm, rate_hz_rhythm = population_rate(
            shifted,
            T_ms=analysis_T,
            dt_ms=RHYTHM_BIN_MS,
            pop="E",
            N_E=config.N_E,
            N_I=config.N_I,
        )
        rhythmicity = _autocorr_rhythmicity(
            rate_hz_rhythm,
            dt_ms=RHYTHM_BIN_MS,
            tau_min_ms=RHYTHM_TAU_MIN_MS,
            tau_max_ms=RHYTHM_TAU_MAX_MS,
        )
        _, _, _, auto_lags, auto_corr = autocorr_peak(
            shifted,
            T_ms=analysis_T,
            dt_ms=RHYTHM_BIN_MS,
            pop="E",
            N_E=config.N_E,
            N_I=config.N_I,
        )
        _, xcorr_lags, xcorr_corr_vals = mean_pairwise_xcorr_peak(
            shifted,
            T_ms=analysis_T,
            N_E=config.N_E,
            dt_ms=RHYTHM_BIN_MS,
        )
        _, coh_lags, coh_vals = coherence_peak(
            shifted,
            T_ms=analysis_T,
            N_E=config.N_E,
            dt_ms=RHYTHM_BIN_MS,
        )
        autocorr_lags_ms = auto_lags.tolist()
        autocorr_corr = auto_corr.tolist()
        xcorr_lags_ms = xcorr_lags.tolist()
        xcorr_corr = xcorr_corr_vals.tolist()
        coherence_lags_ms = coh_lags.tolist()
        coherence_corr = coh_vals.tolist()

    t_ms_full, rate_hz_full_E = population_rate(
        spikes,
        T_ms=config.T,
        dt_ms=RHYTHM_BIN_MS,
        pop="E",
        N_E=config.N_E,
        N_I=config.N_I,
    )
    _, rate_hz_full_I = population_rate(
        spikes,
        T_ms=config.T,
        dt_ms=RHYTHM_BIN_MS,
        pop="I",
        N_E=config.N_E,
        N_I=config.N_I,
    )
    population_rate_t_ms = t_ms_full.tolist()
    population_rate_hz_E = rate_hz_full_E.tolist()
    population_rate_hz_I = rate_hz_full_I.tolist()

    if external_input.ndim == 1:
        input_t_ms = (np.arange(num_steps) * config.dt).tolist()
        input_mean_E = external_input.tolist()
        input_mean_I = external_input.tolist()
    else:
        input_t_ms = (np.arange(num_steps) * config.dt).tolist()
        input_mean_E = np.mean(external_input[:, : config.N_E], axis=1).tolist()
        input_mean_I = np.mean(external_input[:, config.N_E :], axis=1).tolist()

    instruments = result.instruments
    if instruments.V is not None and instruments.times.size > 0:
        membrane_t_ms = instruments.times.tolist()
        v_matrix = instruments.V
        types = instruments.types
        if types is not None:
            e_idx = np.where(types == 0)[0]
            i_idx = np.where(types == 1)[0]
            if e_idx.size > 0:
                membrane_V_E = v_matrix[:, int(e_idx[0])].tolist()
            if i_idx.size > 0:
                membrane_V_I = v_matrix[:, int(i_idx[0])].tolist()

    return RunResponse(
        spikes=spikes_response,
        runtime_ms=runtime_ms,
        num_steps=num_steps,
        num_spikes=len(spikes.times),
        rhythmicity=rhythmicity,
        mean_rate_E=mean_rate_E,
        mean_rate_I=mean_rate_I,
        isi_cv_E=isi_cv_E,
        population_rate_t_ms=population_rate_t_ms,
        population_rate_hz_E=population_rate_hz_E,
        population_rate_hz_I=population_rate_hz_I,
        membrane_t_ms=membrane_t_ms,
        membrane_V_E=membrane_V_E,
        membrane_V_I=membrane_V_I,
        autocorr_lags_ms=autocorr_lags_ms,
        autocorr_corr=autocorr_corr,
        xcorr_lags_ms=xcorr_lags_ms,
        xcorr_corr=xcorr_corr,
        coherence_lags_ms=coherence_lags_ms,
        coherence_corr=coherence_corr,
        weights_hist_bins=weights_hist_bins,
        weights_hist_counts_ee=weights_hist_counts_ee,
        weights_hist_counts_ei=weights_hist_counts_ei,
        weights_hist_counts_ie=weights_hist_counts_ie,
        weights_hist_counts_ii=weights_hist_counts_ii,
        input_t_ms=input_t_ms,
        input_mean_E=input_mean_E,
        input_mean_I=input_mean_I,
    )
