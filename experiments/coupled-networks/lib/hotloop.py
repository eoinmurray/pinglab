from __future__ import annotations
import numpy as np

from pinglab import run_network, inputs
from pinglab.analysis import mean_firing_rates
from pinglab.types import NetworkResult


def hotloop_burnin(
    *,
    config,
    burnin_ms: float,
    added_current_E: dict[int, float],
    I_E_override: float,
    seed: int,
) -> float:
    """
    Run a short simulation and return mean E firing rate.

    Used by the homeostatic controller to quickly measure network rate
    without running the full experiment.

    Args:
        config: Experiment config.
        burnin_ms: Simulation duration (ms).
        added_current_E: Image current dict {neuron_id: amplitude}.
        I_E_override: Override value for I_E baseline current.
        seed: Random seed.

    Returns:
        Mean firing rate of excitatory population (Hz).
    """
    dt = config.base.dt
    N_E = config.base.N_E
    N_I = config.base.N_I
    num_steps = int(burnin_ms / dt)

    # Build input with overridden I_E
    I_ext = inputs.tonic(
        N_E=N_E,
        N_I=N_I,
        I_E=I_E_override,
        I_I=config.default_inputs.I_I,
        noise_std=config.default_inputs.noise,
        num_steps=num_steps,
        seed=seed,
    )

    # Add image current throughout
    if added_current_E:
        ids = np.array(list(added_current_E.keys()), dtype=np.int64)
        amps = np.array([added_current_E[i] for i in ids], dtype=np.float32)
        I_ext[:, ids] += amps[None, :]

    # Run with minimal config (no instruments needed)
    run_cfg = config.base.model_copy(update={"T": burnin_ms, "instruments": None})
    result: NetworkResult = run_network(run_cfg, external_input=I_ext)

    # Compute mean E rate
    rate_E, _ = mean_firing_rates(result.spikes, N_E=N_E, N_I=N_I)
    return rate_E


def hotloop_single_image(
    *,
    config,
    warmup_ms: float,
    stim_ms: float,
    added_current_E: dict[int, float],
    seed: int,
) -> tuple[NetworkResult, dict]:
    """
    Runs one trial: warmup + image step input.
    We build an external_input array (num_steps, N_E+N_I).

    Returns (result, meta) where meta includes timing indices.
    """
    dt = config.base.dt
    T = config.base.T
    N_E = config.base.N_E
    N_I = config.base.N_I

    num_steps = int(T / dt)
    warmup_steps = int(warmup_ms / dt)
    stim_steps = int(stim_ms / dt)

    stim_start = warmup_steps
    stim_stop = min(num_steps, stim_start + stim_steps)

    # baseline tonic with noise
    I_ext = inputs.tonic(
        N_E=N_E,
        N_I=N_I,
        I_E=config.default_inputs.I_E,
        I_I=config.default_inputs.I_I,
        noise_std=config.default_inputs.noise,
        num_steps=num_steps,
        seed=seed,
    )

    # add the image current to selected E neurons during stim window
    if added_current_E:
        ids = np.array(list(added_current_E.keys()), dtype=np.int64)
        amps = np.array([added_current_E[i] for i in ids], dtype=np.float32)
        I_ext[stim_start:stim_stop, ids] += amps[None, :]

    result: NetworkResult = run_network(config.base, external_input=I_ext)

    meta = dict(
        stim_start_ms=stim_start * dt,
        stim_stop_ms=stim_stop * dt,
        stim_start_step=stim_start,
        stim_stop_step=stim_stop,
        dt=dt,
        T=T,
    )
    return result, meta
