from __future__ import annotations

import time
from typing import Literal

import numpy as np
from fastapi import Body, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from pinglab.analysis import autocorr_rhythmicity, mean_firing_rates, population_rate
from pinglab.inputs.tonic import tonic
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

    I_E: float | None = None
    I_I: float | None = None
    noise_std: float | None = None
    seed: int | None = None


class WeightsSpec(BaseModel):
    model_config = ConfigDict(extra="ignore")

    mean_ee: float | None = None
    mean_ei: float | None = None
    mean_ie: float | None = None
    mean_ii: float | None = None
    std_ee: float | None = None
    std_ei: float | None = None
    std_ie: float | None = None
    std_ii: float | None = None
    p_ee: float | None = None
    p_ei: float | None = None
    p_ie: float | None = None
    p_ii: float | None = None
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
    population_rate_t_ms: list[float]
    population_rate_hz_E: list[float]
    population_rate_hz_I: list[float]
    membrane_t_ms: list[float]
    membrane_V_E: list[float]
    membrane_V_I: list[float]


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
    "I_E": 0.7,
    "I_I": 0.7,
    "noise_std": 2.0,
    "seed": 0,
}

DEFAULT_WEIGHTS = {
    "mean_ee": 0.02,
    "mean_ei": 0.015,
    "mean_ie": 0.015,
    "mean_ii": 0.02,
    "std_ee": 0.0,
    "std_ei": 0.0,
    "std_ie": 0.0,
    "std_ii": 0.0,
    "p_ee": 0.02,
    "p_ei": 0.18,
    "p_ie": 0.04,
    "p_ii": 0.06,
    "clamp_min": 0.0,
    "seed": 0,
}

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

RHYTHM_BIN_MS = 5.0
RHYTHM_BURN_IN_MS = 200.0
RHYTHM_TAU_MIN_MS = 5.0
RHYTHM_TAU_MAX_MS = 200.0


def _merge_defaults(defaults: dict, overrides: BaseModel | None) -> dict:
    merged = defaults.copy()
    if overrides is None:
        return merged
    merged.update(overrides.model_dump(exclude_none=True))
    return merged


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

    external_input = tonic(
        config.N_E,
        config.N_I,
        inputs["I_E"],
        inputs["I_I"],
        inputs["noise_std"],
        num_steps,
        int(input_seed) if input_seed is not None else 0,
    )

    weight_mats = build_adjacency_matrices(
        N_E=config.N_E,
        N_I=config.N_I,
        mean_ee=weights["mean_ee"],
        mean_ei=weights["mean_ei"],
        mean_ie=weights["mean_ie"],
        mean_ii=weights["mean_ii"],
        std_ee=weights["std_ee"],
        std_ei=weights["std_ei"],
        std_ie=weights["std_ie"],
        std_ii=weights["std_ii"],
        p_ee=weights["p_ee"],
        p_ei=weights["p_ei"],
        p_ie=weights["p_ie"],
        p_ii=weights["p_ii"],
        clamp_min=weights["clamp_min"],
        seed=int(weights_seed) if weights_seed is not None else None,
    )

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
    population_rate_t_ms: list[float] = []
    population_rate_hz_E: list[float] = []
    population_rate_hz_I: list[float] = []
    membrane_t_ms: list[float] = []
    membrane_V_E: list[float] = []
    membrane_V_I: list[float] = []
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
        t_ms_rhythm, rate_hz_rhythm = population_rate(
            shifted,
            T_ms=analysis_T,
            dt_ms=RHYTHM_BIN_MS,
            pop="E",
            N_E=config.N_E,
            N_I=config.N_I,
        )
        rhythmicity = autocorr_rhythmicity(
            rate_hz_rhythm,
            dt_ms=RHYTHM_BIN_MS,
            tau_min_ms=RHYTHM_TAU_MIN_MS,
            tau_max_ms=RHYTHM_TAU_MAX_MS,
        )

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
        population_rate_t_ms=population_rate_t_ms,
        population_rate_hz_E=population_rate_hz_E,
        population_rate_hz_I=population_rate_hz_I,
        membrane_t_ms=membrane_t_ms,
        membrane_V_E=membrane_V_E,
        membrane_V_I=membrane_V_I,
    )
