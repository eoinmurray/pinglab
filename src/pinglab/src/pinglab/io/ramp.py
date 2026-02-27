"""Ramp input generation with additive noise."""

import numpy as np


def ramp(
    N_E: int,
    N_I: int,
    I_E_start: float,
    I_E_end: float,
    I_I_start: float,
    I_I_end: float,
    noise_std_E: float,
    noise_std_I: float,
    num_steps: int,
    dt: float,
    seed: int,
) -> np.ndarray:
    """
    Generate a linear ramp input with additive noise.

    Parameters:
        N_E: Number of excitatory neurons
        N_I: Number of inhibitory neurons
        I_E_start: Starting input current for E neurons
        I_E_end: Ending input current for E neurons
        I_I_start: Starting input current for I neurons
        I_I_end: Ending input current for I neurons
        noise_std_E: Standard deviation of additive Gaussian noise for E neurons
        noise_std_I: Standard deviation of additive Gaussian noise for I neurons
        num_steps: Number of simulation time steps
        dt: Time step duration in milliseconds
        seed: Random seed for noise generation

    Returns:
        np.ndarray: External input of shape (num_steps, N_E + N_I)
    """
    if N_E < 0 or N_I < 0:
        raise ValueError("N_E and N_I must be non-negative")
    if noise_std_E < 0 or noise_std_I < 0:
        raise ValueError("noise_std must be non-negative")
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    if dt <= 0:
        raise ValueError("dt must be positive")

    rng = np.random.RandomState(seed)
    t = np.linspace(0.0, 1.0, num_steps, endpoint=False)

    I_E = I_E_start + (I_E_end - I_E_start) * t
    I_I = I_I_start + (I_I_end - I_I_start) * t

    I_ext = np.column_stack(
        [
            np.tile(I_E[:, np.newaxis], (1, N_E)),
            np.tile(I_I[:, np.newaxis], (1, N_I)),
        ]
    )

    if N_E > 0:
        noise_e = rng.normal(0, noise_std_E, (num_steps, N_E))
        I_ext[:, :N_E] += noise_e
    if N_I > 0:
        noise_i = rng.normal(0, noise_std_I, (num_steps, N_I))
        I_ext[:, N_E:] += noise_i

    # Clamp tonic input to be non-negative
    np.maximum(I_ext, 0.0, out=I_ext)

    return I_ext
