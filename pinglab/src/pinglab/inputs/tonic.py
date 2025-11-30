"""Tonic (constant baseline) input generation with noise."""

import numpy as np


def tonic(
    N_E: int,
    N_I: int,
    I_E: float,
    I_I: float,
    noise_std: float,
    num_steps: int,
    seed: int,
) -> np.ndarray:
    """
    Generate external tonic input with additive Gaussian noise.

    Parameters:
        N_E: Number of excitatory neurons
        N_I: Number of inhibitory neurons
        I_E: Baseline input current to E neurons
        I_I: Baseline input current to I neurons
        noise_std: Standard deviation of additive Gaussian noise
        num_steps: Number of simulation time steps
        seed: Random seed for noise generation

    Returns:
        np.ndarray: External input of shape (num_steps, N_E + N_I)
    """
    rng = np.random.RandomState(seed)
    N = N_E + N_I

    # Build baseline input (vectorized)
    I_base = np.concatenate([np.full(N_E, I_E), np.full(N_I, I_I)])

    # Generate all noise at once and add to baseline
    noise = rng.normal(0, noise_std, (num_steps, N))
    I_ext = I_base + noise

    return I_ext
