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


def _parse_dist_params(dist_params: dict | None) -> dict[str, float]:
    if not dist_params:
        return {}
    return {str(k): float(v) for k, v in dist_params.items()}


def _block_spec(block: object) -> tuple[str, dict[str, float], float]:
    if hasattr(block, "dist"):
        dist = getattr(block, "dist")
        dist_name = getattr(dist, "name", "normal")
        dist_params = getattr(dist, "params", None)
        p = getattr(block, "p", 1.0)
    elif isinstance(block, dict):
        dist = block.get("dist", {})
        dist_name = dist.get("name", "normal")
        dist_params = dist.get("params", None)
        p = block.get("p", 1.0)
    else:
        raise ValueError("Weight block must be a WeightBlockSpec or dict.")
    return str(dist_name), _parse_dist_params(dist_params), float(p)


def _sample_distribution(
    rng: np.random.RandomState,
    dist_name: str,
    dist_params: dict[str, float],
    shape: tuple[int, int],
) -> np.ndarray:
    if dist_name == "normal":
        mean = float(dist_params.get("mean", 0.0))
        std = float(dist_params.get("std", 1.0))
        return rng.normal(loc=mean, scale=std, size=shape)
    if dist_name == "lognormal":
        mean = float(dist_params.get("mean", 0.0))
        sigma = float(dist_params.get("sigma", 1.0))
        return rng.lognormal(mean=mean, sigma=sigma, size=shape)
    if dist_name == "gamma":
        shape_k = float(dist_params.get("shape", 1.0))
        if "rate" in dist_params:
            scale = 1.0 / float(dist_params["rate"])
        else:
            scale = float(dist_params.get("scale", 1.0))
        return rng.gamma(shape=shape_k, scale=scale, size=shape)
    if dist_name == "exponential":
        if "rate" in dist_params:
            scale = 1.0 / float(dist_params["rate"])
        else:
            scale = float(dist_params.get("scale", 1.0))
        return rng.exponential(scale=scale, size=shape)
    raise ValueError(f"Unknown weight distribution '{dist_name}'")


def _sample_block(
    rng: np.random.RandomState,
    shape: tuple[int, int],
    p: float,
    clamp_min: float | None,
    dist_name: str,
    dist_params: dict[str, float] | None,
) -> np.ndarray:
    mask = rng.random_sample(shape) < p
    dist_params = _parse_dist_params(dist_params)
    weights = _sample_distribution(
        rng,
        dist_name,
        dist_params,
        shape,
    )
    if clamp_min is not None:
        weights = np.clip(weights, clamp_min, None)
    return weights * mask


def build_adjacency_matrices(
    *,
    N_E: int,
    N_I: int,
    ee: object,
    ei: object,
    ie: object,
    ii: object,
    clamp_min: float | None = 0.0,
    seed: int | None = None,
) -> WeightMatrices:
    """
    Build block adjacency matrices with independent per-block distributions.

    Rows are targets, columns are sources. Blocks are ordered E then I.
    """
    rng = np.random.RandomState(seed)

    ee_name, ee_params, ee_p = _block_spec(ee)
    ei_name, ei_params, ei_p = _block_spec(ei)
    ie_name, ie_params, ie_p = _block_spec(ie)
    ii_name, ii_params, ii_p = _block_spec(ii)

    W_ee = _sample_block(
        rng, (N_E, N_E), ee_p, clamp_min, ee_name, ee_params
    )
    W_ei = _sample_block(
        rng, (N_I, N_E), ei_p, clamp_min, ei_name, ei_params
    )
    W_ie = _sample_block(
        rng, (N_E, N_I), ie_p, clamp_min, ie_name, ie_params
    )
    W_ii = _sample_block(
        rng, (N_I, N_I), ii_p, clamp_min, ii_name, ii_params
    )

    # Scale by 1/sqrt(N_src) to keep input variance stable across sizes.
    if N_E > 0:
        scale_e = 1.0 / np.sqrt(N_E)
        W_ee *= scale_e
        W_ei *= scale_e
    if N_I > 0:
        scale_i = 1.0 / np.sqrt(N_I)
        W_ie *= scale_i
        W_ii *= scale_i

    N = N_E + N_I
    W = np.zeros((N, N), dtype=float)
    W[:N_E, :N_E] = W_ee
    W[N_E:, :N_E] = W_ei
    W[:N_E, N_E:] = W_ie
    W[N_E:, N_E:] = W_ii

    return WeightMatrices(W=W, W_ee=W_ee, W_ei=W_ei, W_ie=W_ie, W_ii=W_ii)
