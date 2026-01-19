from __future__ import annotations

import warnings
import numpy as np

from pinglab.lib import decay_exponential, WeightMatrices
from pinglab.run.apply_heterogeneity import apply_heterogeneity
from pinglab.run.connectivity import build_connectivity, build_event_connectivity
from pinglab.run.instruments import init_instruments, record_instruments, finalize_instruments
from pinglab.run.neuron_models import BaseNeuronModel
from pinglab.run.validation import validate_external_input, validate_dt
from pinglab.types import Spikes, NetworkConfig, NetworkResult

# Tolerance for firing rate validation (1% above theoretical max)
RATE_VALIDATION_TOLERANCE = 1.01


def _post_simulation_checks(
    config: NetworkConfig,
    spike_times: list[float],
    spike_types: list[int],
    ref_steps_arr: np.ndarray,
) -> None:
    if not spike_times:
        return

    max_possible_rate_E = 1.0 / (config.t_ref_E / 1000.0)
    max_possible_rate_I = 1.0 / (config.t_ref_I / 1000.0)
    rate_E = (
        len([s for s in spike_types if s == 0]) / (config.N_E * (config.T / 1000.0))
        if config.N_E > 0
        else 0.0
    )
    rate_I = (
        len([s for s in spike_types if s == 1]) / (config.N_I * (config.T / 1000.0))
        if config.N_I > 0
        else 0.0
    )

    if rate_E > max_possible_rate_E * RATE_VALIDATION_TOLERANCE:
        warnings.warn(
            f"E firing rate {rate_E:.1f} Hz exceeds theoretical max {max_possible_rate_E:.1f} Hz",
            UserWarning,
            stacklevel=2,
        )
    if rate_I > max_possible_rate_I * RATE_VALIDATION_TOLERANCE:
        warnings.warn(
            f"I firing rate {rate_I:.1f} Hz exceeds theoretical max {max_possible_rate_I:.1f} Hz",
            UserWarning,
            stacklevel=2,
        )

    n_total = len(spike_times)
    min_ref_steps = int(np.min(ref_steps_arr))
    num_steps = int(np.ceil(config.T / config.dt))
    max_spikes_per_neuron = num_steps / max(1, min_ref_steps)
    expected_max = (config.N_E + config.N_I) * max_spikes_per_neuron
    if n_total > expected_max * RATE_VALIDATION_TOLERANCE:
        warnings.warn(
            f"Spike count {n_total} exceeds theoretical max {expected_max:.0f}",
            UserWarning,
            stacklevel=2,
        )


def _select_external_input(external_input: np.ndarray, step: int, N: int) -> np.ndarray:
    if external_input.ndim == 1:
        return np.full(N, external_input[step])
    return external_input[step, :]


def run_network(
    config: NetworkConfig,
    external_input: np.ndarray,
    *,
    model: BaseNeuronModel,
    weights: WeightMatrices | np.ndarray,
    connectivity_backend: str = "event",
) -> NetworkResult:
    """
    Run a conductance-based E/I network simulation.

    Parameters:
        config: Network configuration including neuron counts, time constants,
                coupling strengths, and simulation parameters
        external_input: External input current array of shape (num_steps, N) or (num_steps,)
        model: Initialized neuron model with step() and get_state()
        weights: Adjacency matrices (full N x N or block) for explicit connectivity
        connectivity_backend: "event" (spike-driven) or "dense" (matrix multiply)

    Returns:
        NetworkResult containing spike data and instrument recordings
    """
    v = config
    N = v.N_E + v.N_I
    num_steps = int(np.ceil(v.T / v.dt))

    validate_external_input(v, external_input)
    validate_dt(v)

    if weights is None:
        raise ValueError("weights must be provided for adjacency-only connectivity.")

    rng = np.random.RandomState(v.seed)

    V_th_arr, g_L_arr, C_m_arr, t_ref_arr = apply_heterogeneity(
        rng,
        v.N_E,
        v.N_I,
        v.V_th_heterogeneity_sd,
        v.g_L_heterogeneity_sd,
        v.C_m_heterogeneity_sd,
        v.t_ref_heterogeneity_sd,
        v.g_L_E,
        v.g_L_I,
        v.C_m_E,
        v.C_m_I,
        v.V_th,
        v.t_ref_E,
        v.t_ref_I,
    )

    ref_steps_arr = np.round(t_ref_arr / v.dt).astype(int)

    V = np.full(N, v.V_init)
    g_e = np.zeros(N)
    g_i = np.zeros(N)
    refractory_countdown = np.zeros(N, dtype=int)

    model.initialize(V)
    if connectivity_backend == "dense":
        connectivity = build_connectivity(v, weights)
    elif connectivity_backend == "event":
        connectivity = build_event_connectivity(v, weights)
    else:
        raise ValueError(f"Unknown connectivity_backend: {connectivity_backend}")
    instruments = init_instruments(v.instruments, N)

    spike_times: list[float] = []
    spike_ids: list[int] = []
    spike_types: list[int] = []

    for step in range(num_steps):
        t = step * v.dt
        connectivity.apply(v, g_e, g_i)

        I_ext = _select_external_input(external_input, step, N)

        g_e = decay_exponential(g_e, v.tau_ampa, v.dt)
        g_i = decay_exponential(g_i, v.tau_gaba, v.dt)

        refractory_countdown[refractory_countdown > 0] -= 1
        can_spike = refractory_countdown == 0

        V, spiked = model.step(
            V,
            g_e,
            g_i,
            I_ext,
            C_m_arr,
            g_L_arr,
            V_th_arr,
            can_spike,
        )

        if spiked.any():
            idxs = np.nonzero(spiked)[0]
            refractory_countdown[idxs] = ref_steps_arr[idxs]
            if model.requires_reset:
                V[idxs] = v.V_reset
            for idx in idxs:
                spike_times.append(t)
                spike_ids.append(idx)
                spike_types.append(0 if idx < v.N_E else 1)
            connectivity.schedule(spiked, v.N_E)

        record_instruments(instruments, t, V, g_e, g_i, v.N_E)
        connectivity.advance()

    _post_simulation_checks(v, spike_times, spike_types, ref_steps_arr)

    spikes = Spikes(
        times=np.array(spike_times),
        ids=np.array(spike_ids),
        types=np.array(spike_types),
    )

    instruments_result = finalize_instruments(instruments, v.N_E)

    return NetworkResult(spikes=spikes, instruments=instruments_result)
