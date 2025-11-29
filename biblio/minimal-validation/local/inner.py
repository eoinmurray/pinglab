
import numpy as np

from pinglab import run_network
from pinglab.inputs import generate_tonic_noise_input
from pinglab.types import NetworkResult


def inner(cfg):
    config = cfg["config"]
    g_ei = cfg["g_ei"]
    I_E = cfg["I_E"]

    external_input = generate_tonic_noise_input(
        N_E=int(config.base.N_E),
        N_I=int(config.base.N_E * 0.25),
        I_E=I_E,
        I_I=config.inputs.I_I,
        noise_std=config.inputs.noise,
        num_steps=int(np.ceil(config.base.T / config.base.dt)),
        seed=config.base.seed if config.base.seed is not None else 0,
    )

    result: NetworkResult = run_network(
        config.base.model_copy(
            update={
                "external_input": external_input,
                "g_ei": g_ei,
            }
        )
    )

    return result
