"""
Utility for building adjacency matrices from simple distributions.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class WeightMatrices:
    W: np.ndarray
    W_ee: np.ndarray
    W_ei: np.ndarray
    W_ie: np.ndarray
    W_ii: np.ndarray


def split_weight_matrix(W: np.ndarray, N_E: int) -> WeightMatrices:
    """
    Split a full (N x N) weight matrix into E/I blocks (targets x sources).
    """
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError("Weight matrix W must be square (N x N).")
    N = W.shape[0]
    N_I = N - N_E
    if N_I < 0:
        raise ValueError("N_E cannot exceed total N in weight matrix.")
    W_ee = W[:N_E, :N_E]
    W_ei = W[N_E:, :N_E]
    W_ie = W[:N_E, N_E:]
    W_ii = W[N_E:, N_E:]
    return WeightMatrices(W=W, W_ee=W_ee, W_ei=W_ei, W_ie=W_ie, W_ii=W_ii)


def assemble_weight_matrix(
    W_ee: np.ndarray,
    W_ei: np.ndarray,
    W_ie: np.ndarray,
    W_ii: np.ndarray,
) -> np.ndarray:
    """
    Assemble a full (N x N) weight matrix from E/I blocks (targets x sources).
    """
    if W_ee.ndim != 2 or W_ei.ndim != 2 or W_ie.ndim != 2 or W_ii.ndim != 2:
        raise ValueError("All weight blocks must be 2D arrays.")
    N_E = W_ee.shape[0]
    if W_ee.shape[1] != N_E:
        raise ValueError("W_ee must be square (N_E x N_E).")
    N_I = W_ii.shape[0]
    if W_ii.shape[1] != N_I:
        raise ValueError("W_ii must be square (N_I x N_I).")
    if W_ei.shape != (N_I, N_E):
        raise ValueError("W_ei must have shape (N_I, N_E).")
    if W_ie.shape != (N_E, N_I):
        raise ValueError("W_ie must have shape (N_E, N_I).")

    N = N_E + N_I
    W = np.zeros((N, N), dtype=float)
    W[:N_E, :N_E] = W_ee
    W[N_E:, :N_E] = W_ei
    W[:N_E, N_E:] = W_ie
    W[N_E:, N_E:] = W_ii
    return W


def _sample_block(
    rng: np.random.RandomState,
    mean: float,
    std: float,
    shape: tuple[int, int],
    p: float,
    clamp_min: float | None,
) -> np.ndarray:
    mask = rng.random_sample(shape) < p
    if std == 0.0:
        weights = np.full(shape, mean, dtype=float)
    else:
        weights = rng.normal(loc=mean, scale=std, size=shape)
    if clamp_min is not None:
        weights = np.clip(weights, clamp_min, None)
    return weights * mask


def build_adjacency_matrices(
    *,
    N_E: int,
    N_I: int,
    mean_ee: float,
    mean_ei: float,
    mean_ie: float,
    mean_ii: float,
    std_ee: float,
    std_ei: float,
    std_ie: float,
    std_ii: float,
    p_ee: float,
    p_ei: float,
    p_ie: float,
    p_ii: float,
    clamp_min: float | None = 0.0,
    seed: int | None = None,
) -> WeightMatrices:
    """
    Build block adjacency matrices with independent Gaussian weights.

    Rows are targets, columns are sources. Blocks are ordered E then I.
    """
    rng = np.random.RandomState(seed)

    W_ee = _sample_block(rng, mean_ee, std_ee, (N_E, N_E), p_ee, clamp_min)
    W_ei = _sample_block(rng, mean_ei, std_ei, (N_I, N_E), p_ei, clamp_min)
    W_ie = _sample_block(rng, mean_ie, std_ie, (N_E, N_I), p_ie, clamp_min)
    W_ii = _sample_block(rng, mean_ii, std_ii, (N_I, N_I), p_ii, clamp_min)

    N = N_E + N_I
    W = np.zeros((N, N), dtype=float)
    W[:N_E, :N_E] = W_ee
    W[N_E:, :N_E] = W_ei
    W[:N_E, N_E:] = W_ie
    W[N_E:, N_E:] = W_ii

    return WeightMatrices(W=W, W_ee=W_ee, W_ei=W_ei, W_ie=W_ie, W_ii=W_ii)
