
import numpy as np

from pinglab import run_network
from pinglab.inputs import tonic
from pinglab.types import NetworkResult

from .model import LocalConfig


def hotloop(cfg):
    config: LocalConfig = cfg["config"]
    g_ei = cfg["g_ei"]
    I_E = cfg["I_E"]

    external_input = tonic(
        N_E=int(config.base.N_E),
        N_I=int(config.base.N_E * 0.25),
        I_E=I_E,
        I_I=config.default_inputs.I_I,
        noise_std=config.default_inputs.noise,
        num_steps=int(np.ceil(config.base.T / config.base.dt)),
        seed=config.base.seed if config.base.seed is not None else 0,
    )

    run_cfg = config.base.model_copy(update={ "I_E": I_E, "g_ei": g_ei })

    result: NetworkResult = run_network(run_cfg, external_input=external_input) 

    return result
