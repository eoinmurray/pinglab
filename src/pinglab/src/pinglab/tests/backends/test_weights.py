import numpy as np

from pinglab.io.weights import build_adjacency_matrices


def _blocks(
    *,
    ee_mean: float = 1.0,
    ei_mean: float = 1.0,
    ie_mean: float = 1.0,
    ii_mean: float = 1.0,
) -> dict[str, dict]:
    return {
        "ee": {"mean": ee_mean, "std": 0.0},
        "ei": {"mean": ei_mean, "std": 0.0},
        "ie": {"mean": ie_mean, "std": 0.0},
        "ii": {"mean": ii_mean, "std": 0.0},
    }


def test_build_adjacency_matrices_normal_positive_when_clamped():
    weights = build_adjacency_matrices(
        N_E=10,
        N_I=5,
        ee={"mean": 0.0, "std": 0.5},
        ei={"mean": 0.0, "std": 0.5},
        ie={"mean": 0.0, "std": 0.5},
        ii={"mean": 0.0, "std": 0.5},
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
        **_blocks(),
        clamp_min=0.0,
        seed=0,
    )

    expected_e = 1.0 / np.sqrt(100)
    expected_i = 1.0 / np.sqrt(25)
    assert np.allclose(weights.W_ee, expected_e)
    assert np.allclose(weights.W_ei, expected_e)
    assert np.allclose(weights.W_ie, expected_i)
    assert np.allclose(weights.W_ii, expected_i)


def test_delay_matrices_are_unset_without_templates():
    weights = build_adjacency_matrices(
        N_E=6,
        N_I=2,
        **_blocks(),
        clamp_min=0.0,
        seed=7,
    )
    assert np.isnan(weights.D_ee).all()
    assert np.isnan(weights.D_ei).all()
    assert np.isnan(weights.D_ie).all()
    assert np.isnan(weights.D_ii).all()
