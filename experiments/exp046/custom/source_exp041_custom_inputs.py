"""Strict v4 inputs and complete pinned lineage, without historical fallbacks.

The owning experiment calls these local definitions from its explicit stages.
Inputs and outputs retain that stage’s validated scientific contract; importing
this module does not start computation or select a scientific run.
"""

from experiments.exp046.custom import source_exp041_recipe as recipe
from pingstore.contracts import PingstoreError


def configuration(run):
    cfg = run.record["execution"].get("configuration")
    if (
        not isinstance(cfg, dict)
        or cfg.get("schema") not in ("exp041.recipe/v1", "exp041.recipe/v2")
        or cfg.get("profile") not in ("smoke", "production")
        or cfg
        != recipe.configuration(
            smoke=cfg["profile"] == "smoke",
            version=int(cfg["schema"].rsplit("v", 1)[1]),
        )
        or set(run.record["inputs"]) != {"bank"}
    ):
        raise PingstoreError("inconsistent exp041 compute recipe or bank input")
    return cfg
