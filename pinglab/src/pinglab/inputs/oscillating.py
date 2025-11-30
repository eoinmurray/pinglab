"""Oscillating input generation with configurable frequency and phase."""

import numpy as np


def oscillating(
    N_E: int,
    N_I: int,
    I_E: float,
    I_I: float,
    noise_std: float,
    num_steps: int,
    dt: float,
    seed: int,
    oscillation_freq: float = 40.0,
    oscillation_amplitude: float = 0.5,
    oscillation_phase: float = 0.0,
    phase_offset_I: float = 0.0,
) -> np.ndarray:
    """
    Generate oscillating input with additive noise.

    Parameters:
        N_E: Number of excitatory neurons
        N_I: Number of inhibitory neurons
        I_E: Baseline input current to E neurons
        I_I: Baseline input current to I neurons
        noise_std: Standard deviation of additive Gaussian noise
        num_steps: Number of simulation time steps
        dt: Time step duration in milliseconds
        seed: Random seed for noise generation
        oscillation_freq: Frequency of oscillation in Hz
        oscillation_amplitude: Amplitude of oscillation (added to baseline)
        oscillation_phase: Initial phase of oscillation in radians
        phase_offset_I: Phase offset for I population relative to E (radians)

    Returns:
        np.ndarray: External input of shape (num_steps, N_E + N_I)
    """
    rng = np.random.RandomState(seed)

    # Convert frequency to angular frequency (rad/ms)
    omega = 2.0 * np.pi * oscillation_freq / 1000.0  # Hz to rad/ms

    # Generate time array
    t = np.arange(num_steps) * dt  # Time in ms

    # Compute oscillations for E and I populations (vectorized)
    osc_E = np.sin(omega * t + oscillation_phase)
    osc_I = np.sin(omega * t + oscillation_phase + phase_offset_I)

    # Compute oscillating baselines for each population
    I_E_osc = I_E + oscillation_amplitude * osc_E  # Shape: (num_steps,)
    I_I_osc = I_I + oscillation_amplitude * osc_I  # Shape: (num_steps,)

    # Build full input array (vectorized)
    I_ext = np.column_stack([
        np.tile(I_E_osc[:, np.newaxis], (1, N_E)),
        np.tile(I_I_osc[:, np.newaxis], (1, N_I)),
    ])

    # Add Gaussian noise
    noise = rng.normal(0, noise_std, (num_steps, N_E + N_I))
    I_ext += noise

    return I_ext
