from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from pinglab.types import InstrumentsConfig, InstrumentsResults


@dataclass
class InstrumentState:
    neuron_ids: np.ndarray
    downsample: int
    variables: set[str]
    population_means: bool
    times: list[float]
    V: list[np.ndarray] | None
    g_e: list[np.ndarray] | None
    g_i: list[np.ndarray] | None
    V_mean_E: list[float] | None
    V_mean_I: list[float] | None
    g_e_mean_E: list[float] | None
    g_e_mean_I: list[float] | None
    g_i_mean_E: list[float] | None
    g_i_mean_I: list[float] | None
    step_counter: int = 0


def init_instruments(config: InstrumentsConfig, N: int) -> InstrumentState:
    if config.all_neurons or config.neuron_ids is None:
        neuron_ids = np.arange(N)
    else:
        neuron_ids = np.array(config.neuron_ids, dtype=int)

    variables = set(config.variables)
    population_means = config.population_means

    return InstrumentState(
        neuron_ids=neuron_ids,
        downsample=config.downsample,
        variables=variables,
        population_means=population_means,
        times=[],
        V=[] if "V" in variables else None,
        g_e=[] if "g_e" in variables else None,
        g_i=[] if "g_i" in variables else None,
        V_mean_E=[] if "V" in variables and population_means else None,
        V_mean_I=[] if "V" in variables and population_means else None,
        g_e_mean_E=[] if "g_e" in variables and population_means else None,
        g_e_mean_I=[] if "g_e" in variables and population_means else None,
        g_i_mean_E=[] if "g_i" in variables and population_means else None,
        g_i_mean_I=[] if "g_i" in variables and population_means else None,
    )


def record_instruments(
    state: InstrumentState,
    t: float,
    V: np.ndarray,
    g_e: np.ndarray,
    g_i: np.ndarray,
    N_E: int,
) -> None:
    if state.step_counter % state.downsample == 0:
        state.times.append(t)
        if state.V is not None:
            state.V.append(V[state.neuron_ids].copy())
        if state.g_e is not None:
            state.g_e.append(g_e[state.neuron_ids].copy())
        if state.g_i is not None:
            state.g_i.append(g_i[state.neuron_ids].copy())
        if state.population_means:
            if state.V_mean_E is not None:
                state.V_mean_E.append(float(np.mean(V[:N_E])))
            if state.V_mean_I is not None:
                state.V_mean_I.append(float(np.mean(V[N_E:])))
            if state.g_e_mean_E is not None:
                state.g_e_mean_E.append(float(np.mean(g_e[:N_E])))
            if state.g_e_mean_I is not None:
                state.g_e_mean_I.append(float(np.mean(g_e[N_E:])))
            if state.g_i_mean_E is not None:
                state.g_i_mean_E.append(float(np.mean(g_i[:N_E])))
            if state.g_i_mean_I is not None:
                state.g_i_mean_I.append(float(np.mean(g_i[N_E:])))
    state.step_counter += 1


def finalize_instruments(state: InstrumentState, N_E: int) -> InstrumentsResults:
    types = np.array([0 if nid < N_E else 1 for nid in state.neuron_ids], dtype=np.uint8)
    return InstrumentsResults(
        times=np.array(state.times),
        neuron_ids=state.neuron_ids,
        V=np.array(state.V) if state.V is not None else None,
        g_e=np.array(state.g_e) if state.g_e is not None else None,
        g_i=np.array(state.g_i) if state.g_i is not None else None,
        types=types,
        V_mean_E=np.array(state.V_mean_E) if state.V_mean_E is not None else None,
        V_mean_I=np.array(state.V_mean_I) if state.V_mean_I is not None else None,
        g_e_mean_E=np.array(state.g_e_mean_E) if state.g_e_mean_E is not None else None,
        g_e_mean_I=np.array(state.g_e_mean_I) if state.g_e_mean_I is not None else None,
        g_i_mean_E=np.array(state.g_i_mean_E) if state.g_i_mean_E is not None else None,
        g_i_mean_I=np.array(state.g_i_mean_I) if state.g_i_mean_I is not None else None,
    )
