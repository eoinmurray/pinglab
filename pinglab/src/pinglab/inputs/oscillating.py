
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
    oscillation_freq: float = 40.0,  # Oscillation frequency in Hz
    oscillation_amplitude: float = 0.5,  # Amplitude of oscillation (relative to baseline)
    oscillation_phase: float = 0.0,  # Initial phase in radians
    phase_offset: float = 0.0,  # Phase offset between E and I populations (radians)
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
        oscillation_amplitude: Amplitude of oscillation (multiplied by baseline)
        oscillation_phase: Initial phase of oscillation in radians
        population_specific: If True, E and I populations receive different phases
        phase_offset: Phase offset between E and I populations in radians (used if population_specific=True)

    Returns:
        np.ndarray: External input of shape (num_steps, N_E + N_I)
    """
    rng = np.random.RandomState(seed)
    N = N_E + N_I

    # Pre-allocate output array
    I_ext = np.zeros((num_steps, N))

    # Convert frequency to angular frequency (rad/ms)
    omega = 2.0 * np.pi * oscillation_freq / 1000.0  # Hz to rad/ms

    # Generate time array
    t = np.arange(num_steps) * dt  # Time in ms

    osc = np.sin(omega * t + oscillation_phase + phase_offset)
    osc_E = osc
    osc_I = osc

    # Generate input for each timestep
    for step in range(num_steps):
        # Base input with oscillation
        I_E_osc = I_E + oscillation_amplitude * osc_E[step]
        I_I_osc = I_I + oscillation_amplitude * osc_I[step]

        I_base = np.concatenate([
            np.full(N_E, I_E_osc),
            np.full(N_I, I_I_osc)
        ])

        # Add Gaussian noise
        noise = rng.normal(0, noise_std, N)
        I_ext[step, :] = I_base + noise

    return I_ext
