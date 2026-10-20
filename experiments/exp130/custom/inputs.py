"""Read authenticated v4 evidence with the private-afferent contract.

This experiment-local helper is called by the independent stages. It consumes
explicit inputs and resolved settings; importing it does not launch a run or
select scientific evidence.
"""

from pathlib import Path

import numpy as np
from pingstore.contracts import PingstoreError
from pingstore.stages import source_run

from experiments.exp130 import recipe


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
    if not isinstance(cfg, dict) or cfg.get("schema") != "exp130.recipe/v1":
        raise PingstoreError("exp130 requires the theta-afferent recipe v1")
    if cfg.get("conditions") != list(recipe.CONDITIONS):
        raise PingstoreError("condition coverage disagrees with recipe")
    return cfg


def recording(compute, condition):
    cfg = configuration(compute)
    with np.load(compute.unit(condition) / "recording.npz", allow_pickle=False) as data:
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
