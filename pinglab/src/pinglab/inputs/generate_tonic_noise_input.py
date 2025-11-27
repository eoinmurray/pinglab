
import numpy as np

def generate_tonic_noise_input(
    N_E: int,
    N_I: int,
    I_E: float,
    I_I: float,
    noise_std: float,
    num_steps: int,
    seed: int,
) -> np.ndarray:
    """Generate external tonic noise input matching internal implementation."""
    rng = np.random.RandomState(seed)
    N = N_E + N_I

    # Pre-allocate output array
    I_ext = np.zeros((num_steps, N))

    # Generate noise for each timestep
    for step in range(num_steps):
        # Base input
        I_base = np.concatenate([np.full(N_E, I_E), np.full(N_I, I_I)])

        # Add Gaussian noise
        noise = rng.normal(0, noise_std, N)
        I_ext[step, :] = I_base + noise

    return I_ext
