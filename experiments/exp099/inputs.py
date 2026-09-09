"""Read authenticated v4 evidence with the private-afferent contract."""

from pathlib import Path

import numpy as np
from pingstore.contracts import PingstoreError
from pingstore.stages import source_run

from . import recipe


def source(repo: Path, identity: str, stage: str, *, reference=None):
    return source_run(
        repo / ".pingstore",
        identity,
        stage=stage,
        experiment=recipe.SLUG,
        reference=reference,
    )


def configuration(run):
    cfg = run.record["execution"].get("configuration")
    if not isinstance(cfg, dict) or cfg.get("schema") != "exp099.recipe/v2":
        raise PingstoreError("exp099 requires the private-afferent recipe v2")
    return cfg


def recording(compute):
    cfg = configuration(compute)
    with np.load(compute.export / "recording.npz", allow_pickle=False) as data:
        result = dict(data)
    steps = round(cfg["t_ms"] / cfg["dt_ms"])
    for pop in ("e", "i"):
        for key in (f"spk_{pop}", f"private_{pop}"):
            if key not in result or result[key].shape != (steps, cfg[f"n_{pop}"]):
                raise PingstoreError(f"recording shape disagrees with recipe: {key}")
    for key in (
        "mean_v_e",
        "mean_v_i",
        "mean_E_to_E",
        "mean_E_to_I",
        "mean_I_to_E",
        "mean_I_to_I",
        "mean_private_e_to_E",
        "mean_private_i_to_I",
    ):
        if (
            key not in result
            or result[key].shape != (steps,)
            or not np.isfinite(result[key]).all()
        ):
            raise PingstoreError(f"missing or invalid recorded population mean: {key}")
    return result
