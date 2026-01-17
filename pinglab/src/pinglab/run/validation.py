from __future__ import annotations

import warnings
import numpy as np

from pinglab.types import NetworkConfig

# Numerical stability thresholds for Euler integration
DT_STABILITY_FACTOR = 5
DT_ACCURACY_FACTOR = 10


def validate_external_input(config: NetworkConfig, external_input: np.ndarray) -> None:
    num_steps = int(np.ceil(config.T / config.dt))
    N = config.N_E + config.N_I

    if external_input.ndim == 1:
        if external_input.shape[0] != num_steps:
            raise ValueError(
                f"external_input has {external_input.shape[0]} steps, expected {num_steps}"
            )
    elif external_input.ndim == 2:
        if external_input.shape[0] != num_steps:
            raise ValueError(
                f"external_input has {external_input.shape[0]} steps, expected {num_steps}"
            )
        if external_input.shape[1] != N:
            raise ValueError(
                f"external_input has {external_input.shape[1]} neurons, expected {N}"
            )
    else:
        raise ValueError(
            f"external_input must be 1D or 2D, got {external_input.ndim}D"
        )


def validate_dt(config: NetworkConfig) -> None:
    tau_mem_E = config.C_m_E / config.g_L_E
    tau_mem_I = config.C_m_I / config.g_L_I
    tau_min = min(tau_mem_E, tau_mem_I)

    if config.dt > tau_min / DT_STABILITY_FACTOR:
        raise ValueError(
            f"Time step dt={config.dt}ms is too large for numerical stability. "
            f"Minimum membrane time constant is tau_min={tau_min:.2f}ms. "
            f"Require dt < tau_min/{DT_STABILITY_FACTOR} = {tau_min/DT_STABILITY_FACTOR:.2f}ms"
        )

    if config.dt > tau_min / DT_ACCURACY_FACTOR:
        warnings.warn(
            f"dt={config.dt}ms is large relative to tau_min={tau_min:.2f}ms. "
            f"Consider dt < {tau_min/DT_ACCURACY_FACTOR:.2f}ms for better accuracy.",
            UserWarning,
            stacklevel=2,
        )
