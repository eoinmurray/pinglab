import numpy as np

from pinglab.lib.weights_builder import build_adjacency_matrices


def test_build_adjacency_matrices_lognormal_positive():
    weights = build_adjacency_matrices(
        N_E=10,
        N_I=5,
        ee={"p": 1.0, "dist": {"name": "lognormal", "params": {"mean": 0.0, "sigma": 0.5}}},
        ei={"p": 1.0, "dist": {"name": "lognormal", "params": {"mean": 0.0, "sigma": 0.5}}},
        ie={"p": 1.0, "dist": {"name": "lognormal", "params": {"mean": 0.0, "sigma": 0.5}}},
        ii={"p": 1.0, "dist": {"name": "lognormal", "params": {"mean": 0.0, "sigma": 0.5}}},
        clamp_min=0.0,
        seed=123,
    )
    assert weights.W.shape == (15, 15)
    assert np.isfinite(weights.W).all()
    assert np.min(weights.W) >= 0.0
