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


def test_build_adjacency_matrices_scaling_by_sqrt_n():
    weights = build_adjacency_matrices(
        N_E=100,
        N_I=25,
        ee={"p": 1.0, "dist": {"name": "normal", "params": {"mean": 1.0, "std": 0.0}}},
        ei={"p": 1.0, "dist": {"name": "normal", "params": {"mean": 1.0, "std": 0.0}}},
        ie={"p": 1.0, "dist": {"name": "normal", "params": {"mean": 1.0, "std": 0.0}}},
        ii={"p": 1.0, "dist": {"name": "normal", "params": {"mean": 1.0, "std": 0.0}}},
        clamp_min=0.0,
        seed=0,
    )

    expected_e = 1.0 / np.sqrt(100)
    expected_i = 1.0 / np.sqrt(25)
    assert np.allclose(weights.W_ee, expected_e)
    assert np.allclose(weights.W_ei, expected_e)
    assert np.allclose(weights.W_ie, expected_i)
    assert np.allclose(weights.W_ii, expected_i)


def test_build_adjacency_matrices_ee_feedforward_blocks():
    weights = build_adjacency_matrices(
        N_E=6,
        N_I=0,
        ee={"p": 1.0, "dist": {"name": "normal", "params": {"mean": 1.0, "std": 0.0}}},
        ei={"p": 0.0, "dist": {"name": "normal", "params": {"mean": 0.0, "std": 0.0}}},
        ie={"p": 0.0, "dist": {"name": "normal", "params": {"mean": 0.0, "std": 0.0}}},
        ii={"p": 0.0, "dist": {"name": "normal", "params": {"mean": 0.0, "std": 0.0}}},
        clamp_min=0.0,
        seed=1,
        ee_template={"name": "feedforward_blocks", "blocks": 3},
    )
    expected = 1.0 / np.sqrt(6)
    W = weights.W_ee
    # Block 0 (0-1): upper triangle only.
    assert np.isclose(W[0, 0], expected)
    assert np.isclose(W[1, 1], expected)
    assert np.isclose(W[0, 1], expected)
    assert np.isclose(W[1, 0], 0.0)
    # Block 1 (2-3): upper triangle only, cross-block zero.
    assert np.isclose(W[2, 2], expected)
    assert np.isclose(W[3, 3], expected)
    assert np.isclose(W[2, 3], expected)
    assert np.isclose(W[3, 2], 0.0)
    assert np.isclose(W[0, 2], 0.0)
    assert np.isclose(W[2, 0], 0.0)
    # Block 2 (4-5): upper triangle only.
    assert np.isclose(W[4, 4], expected)
    assert np.isclose(W[5, 5], expected)
    assert np.isclose(W[4, 5], expected)
    assert np.isclose(W[5, 4], 0.0)
