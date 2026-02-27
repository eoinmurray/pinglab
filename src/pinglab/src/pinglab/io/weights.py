from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ResolvedWeightMatrices:
    W: Any
    W_ee: Any
    W_ei: Any
    W_ie: Any
    W_ii: Any
    D_ee: Any
    D_ei: Any
    D_ie: Any
    D_ii: Any

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


# Backwards-compatible alias for older code paths.
WeightMatrices = ResolvedWeightMatrices


def split_weight_matrix(W: np.ndarray, N_E: int) -> ResolvedWeightMatrices:
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
    D_ee = np.full((N_E, N_E), np.nan, dtype=float)
    D_ei = np.full((N_I, N_E), np.nan, dtype=float)
    D_ie = np.full((N_E, N_I), np.nan, dtype=float)
    D_ii = np.full((N_I, N_I), np.nan, dtype=float)
    return ResolvedWeightMatrices(
        W=W,
        W_ee=W_ee,
        W_ei=W_ei,
        W_ie=W_ie,
        W_ii=W_ii,
        D_ee=D_ee,
        D_ei=D_ei,
        D_ie=D_ie,
        D_ii=D_ii,
    )


def _block_spec(block: object) -> dict[str, float]:
    if hasattr(block, "mean") or hasattr(block, "std"):
        return {
            "mean": float(getattr(block, "mean", 0.0)),
            "std": float(getattr(block, "std", 0.0)),
        }
    if isinstance(block, dict):
        return {
            "mean": float(block.get("mean", 0.0)),
            "std": float(block.get("std", 0.0)),
        }
    raise ValueError("Weight block must be a WeightBlockSpec or dict.")


def _sample_distribution(
    rng: np.random.RandomState,
    dist_params: dict[str, float],
    shape: tuple[int, int],
) -> np.ndarray:
    mean = float(dist_params.get("mean", 0.0))
    std = float(dist_params.get("std", 1.0))
    return rng.normal(loc=mean, scale=std, size=shape)


def _sample_block(
    rng: np.random.RandomState,
    shape: tuple[int, int],
    clamp_min: float | None,
    dist_params: dict[str, float],
) -> np.ndarray:
    weights = _sample_distribution(rng, dist_params, shape)
    if clamp_min is not None:
        weights = np.clip(weights, float(clamp_min), None)
    return weights


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
) -> ResolvedWeightMatrices:
    rng = np.random.RandomState(seed)
    ee_params = _block_spec(ee)
    ei_params = _block_spec(ei)
    ie_params = _block_spec(ie)
    ii_params = _block_spec(ii)

    W_ee = _sample_block(rng, (N_E, N_E), clamp_min, ee_params)
    W_ei = _sample_block(rng, (N_I, N_E), clamp_min, ei_params)
    W_ie = _sample_block(rng, (N_E, N_I), clamp_min, ie_params)
    W_ii = _sample_block(rng, (N_I, N_I), clamp_min, ii_params)

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
    D_ee = np.full((N_E, N_E), np.nan, dtype=float)
    D_ei = np.full((N_I, N_E), np.nan, dtype=float)
    D_ie = np.full((N_E, N_I), np.nan, dtype=float)
    D_ii = np.full((N_I, N_I), np.nan, dtype=float)
    return ResolvedWeightMatrices(
        W=W,
        W_ee=W_ee,
        W_ei=W_ei,
        W_ie=W_ie,
        W_ii=W_ii,
        D_ee=D_ee,
        D_ei=D_ei,
        D_ie=D_ie,
        D_ii=D_ii,
    )


def resolve_weight_matrices_from_full(
    *,
    W: np.ndarray,
    D: np.ndarray,
    n_e: int,
) -> ResolvedWeightMatrices:
    n_total = int(W.shape[0])
    n_i = n_total - int(n_e)
    if n_i < 0:
        raise ValueError("n_e cannot exceed total matrix size")
    return ResolvedWeightMatrices(
        W=W,
        W_ee=W[:n_e, :n_e],
        W_ei=W[n_e:, :n_e],
        W_ie=W[:n_e, n_e:],
        W_ii=W[n_e:, n_e:],
        D_ee=D[:n_e, :n_e],
        D_ei=D[n_e:, :n_e],
        D_ie=D[:n_e, n_e:],
        D_ii=D[n_e:, n_e:],
    )


def to_torch_weights(
    weights: ResolvedWeightMatrices,
    *,
    device: Any,
    dtype: Any,
) -> ResolvedWeightMatrices:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("PyTorch backend requires torch to be installed") from exc

    def _to_torch(value: Any) -> Any:
        if value is None:
            return None
        arr = np.asarray(value)
        if not arr.flags.writeable:
            arr = np.array(arr, copy=True)
        return torch.as_tensor(arr, dtype=dtype, device=device)

    return ResolvedWeightMatrices(
        W=_to_torch(weights.W),
        W_ee=_to_torch(weights.W_ee),
        W_ei=_to_torch(weights.W_ei),
        W_ie=_to_torch(weights.W_ie),
        W_ii=_to_torch(weights.W_ii),
        D_ee=_to_torch(weights.D_ee),
        D_ei=_to_torch(weights.D_ei),
        D_ie=_to_torch(weights.D_ie),
        D_ii=_to_torch(weights.D_ii),
    )
