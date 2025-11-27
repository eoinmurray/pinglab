"""
Synapse models with exponential conductance decay.
"""
import numpy as np



def decay_exponential(g: np.ndarray, tau: float, dt: float) -> np.ndarray:
    """
    Exponentially decay conductance array g with time constant tau.
    """
    return g * np.exp(-dt / tau)