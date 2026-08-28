"""Validate full-trial sparse evidence and the shared final-epoch training bank."""

import numpy as np
from experiments.exp041.evidence import (
    checkpoints,
    finite,
    measurement,
    training_contract,
)
from pingstore.contracts import PingstoreError

__all__ = ["checkpoints", "finite", "measurement", "training_contract", "recordings"]


def recordings(directory, common, samples):
    with np.load(directory / "rasters.npz", allow_pickle=False) as data:
        required = (
            "dt",
            "n_trials",
            "T",
            "n_e",
            "n_i",
            "e_trial",
            "e_t",
            "e_cell",
            "i_trial",
            "i_t",
            "i_cell",
        )
        raster = {key: np.array(data[key]) for key in required}
    for key, expected in {
        "n_trials": samples,
        "T": round(common["t_ms"] / common["dt"]),
        "n_e": common["n_hidden"],
        "n_i": common["n_inh"],
    }.items():
        a = raster[key]
        if a.ndim != 0 or a.dtype.kind not in "iu" or int(a) != expected:
            raise PingstoreError(
                f"raster {key} differs from complete evaluation recipe"
            )
    if raster["dt"].ndim != 0 or not np.isclose(float(raster["dt"]), common["dt"]):
        raise PingstoreError("raster timestep differs")
    for prefix, population in (("e", common["n_hidden"]), ("i", common["n_inh"])):
        indices = [raster[f"{prefix}_{suffix}"] for suffix in ("trial", "t", "cell")]
        for a, limit in zip(
            indices, (samples, int(raster["T"]), population), strict=True
        ):
            if (
                a.ndim != 1
                or a.dtype.kind not in "iu"
                or np.any(a < 0)
                or np.any(a >= limit)
            ):
                raise PingstoreError(f"invalid {prefix} sparse raster indices")
        if len({a.size for a in indices}) != 1:
            raise PingstoreError("sparse raster index lengths differ")
        tr, ts, cells = [a.astype(np.int64) for a in indices]
        linear = (tr * int(raster["T"]) + ts) * population + cells
        if np.unique(linear).size != linear.size:
            raise PingstoreError(
                "duplicate sparse spikes would be collapsed by cycle reconstruction"
            )
    with np.load(directory / "per_cell_rates.npz", allow_pickle=False) as data:
        rates = np.array(data["rate_e_per_cell"])
    if (
        rates.shape != (common["n_hidden"],)
        or not np.isfinite(rates).all()
        or np.any(rates < 0)
    ):
        raise PingstoreError("invalid per-cell E rates")
    expected_rates = np.bincount(
        raster["e_cell"].astype(int), minlength=common["n_hidden"]
    ) / (samples * common["t_ms"] / 1000)
    if not np.allclose(rates, expected_rates, rtol=1e-5, atol=1e-6):
        raise PingstoreError("per-cell rates disagree with full sparse spike counts")
    return raster, rates
