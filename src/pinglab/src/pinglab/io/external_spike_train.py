"""Spike-driven external input using inhomogeneous Poisson sources."""

from __future__ import annotations

import numpy as np


def external_spike_train(
    N_E: int,
    N_I: int,
    I_E_base: float,
    I_I_base: float,
    noise_std_E: float,
    noise_std_I: float,
    num_steps: int,
    dt: float,
    seed: int,
    lambda0_hz: float = 30.0,
    mod_depth: float = 0.5,
    envelope_freq_hz: float = 5.0,
    phase_rad: float = 0.0,
    w_in: float = 0.25,
    tau_in_ms: float = 3.0,
    return_spikes: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate external drive from independent per-neuron Poisson spike sources.

    Each E neuron receives spikes from its own virtual source. Spikes are filtered
    through a one-pole synaptic state before being converted to current.
    """
    # Validate structural inputs early so later array math can assume sane shapes.
    if N_E < 0 or N_I < 0:
        raise ValueError("N_E and N_I must be non-negative")
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    if dt <= 0.0:
        raise ValueError("dt must be positive")

    # Build a per-time-step envelope-modulated rate for the E-population virtual sources.
    rng = np.random.RandomState(seed)
    t_ms = np.arange(num_steps, dtype=float) * float(dt)
    rate_hz = float(lambda0_hz) * (
        1.0
        + float(mod_depth)
        * np.sin(
            2.0 * np.pi * float(envelope_freq_hz) * (t_ms / 1000.0) + float(phase_rad)
        )
    )
    # Rates cannot be negative; clip before converting to Bernoulli spike probability.
    np.maximum(rate_hz, 0.0, out=rate_hz)
    spike_prob = np.clip(rate_hz * float(dt) / 1000.0, 0.0, 1.0)

    # Sample independent spikes for each E neuron at each timestep.
    # I-population spike sources are currently disabled and represented as all-false.
    if N_E > 0:
        spikes_e = rng.random_sample((num_steps, N_E)) < spike_prob[:, np.newaxis]
    else:
        spikes_e = np.zeros((num_steps, 0), dtype=bool)
    spikes_i = np.zeros((num_steps, max(0, N_I)), dtype=bool)

    # Start from tonic baselines, then add spike-driven synaptic drive for E neurons.
    drive_e = np.full((num_steps, max(0, N_E)), float(I_E_base), dtype=float)
    if N_E > 0:
        # One-pole synapse state: x[t+1] = decay * x[t] + spikes[t].
        syn_state = np.zeros(N_E, dtype=float)
        decay = np.exp(-float(dt) / max(1e-9, float(tau_in_ms)))
        for step_idx in range(num_steps):
            syn_state = syn_state * decay + spikes_e[step_idx].astype(float)
            drive_e[step_idx, :] += float(w_in) * syn_state

    # I neurons currently receive only tonic + optional noise input.
    drive_i = np.full((num_steps, max(0, N_I)), float(I_I_base), dtype=float)

    # Add independent Gaussian noise to each population if requested.
    if noise_std_E > 0.0 and N_E > 0:
        drive_e += rng.normal(0.0, float(noise_std_E), size=drive_e.shape)
    if noise_std_I > 0.0 and N_I > 0:
        drive_i += rng.normal(0.0, float(noise_std_I), size=drive_i.shape)

    # Return combined [E | I] current matrix, with optional raw spike rasters.
    drive = np.concatenate([drive_e, drive_i], axis=1)
    if return_spikes:
        return drive, spikes_e, spikes_i
    return drive
