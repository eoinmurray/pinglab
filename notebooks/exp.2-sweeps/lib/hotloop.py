
import numpy as np

from pinglab.lib.weights_builder import build_adjacency_matrices
from pinglab.run import run_network, build_model_from_config
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

    run_cfg = config.base
    if config.weights is None:
        raise ValueError("weights must be provided for adjacency-only runs.")
    matrices = build_adjacency_matrices(
        N_E=run_cfg.N_E,
        N_I=run_cfg.N_I,
        mean_ee=config.weights.mean_ee,
        mean_ei=float(g_ei),
        mean_ie=config.weights.mean_ie,
        mean_ii=config.weights.mean_ii,
        std_ee=config.weights.std_ee,
        std_ei=config.weights.std_ei,
        std_ie=config.weights.std_ie,
        std_ii=config.weights.std_ii,
        p_ee=config.weights.p_ee,
        p_ei=config.weights.p_ei,
        p_ie=config.weights.p_ie,
        p_ii=config.weights.p_ii,
        clamp_min=config.weights.clamp_min,
        seed=run_cfg.seed,
    )

    model = build_model_from_config(run_cfg)
    result: NetworkResult = run_network(
        run_cfg,
        external_input=external_input,
        model=model,
        weights=matrices.W,
    )

    return result
