"""Validated v4 inputs only, with exact upstream pins and no active-run fallback.

The owning experiment calls these local definitions from its explicit stages.
Inputs and outputs retain that stage’s validated scientific contract; importing
this module does not start computation or select a scientific run.
"""

from pathlib import Path

from pingstore.contracts import PingstoreError
from pingstore.stages import SourceRun, source_run

from experiments.exp023 import recipe


def source(
    repo: Path, identity: str, stage: str, *, reference: dict | None = None
) -> SourceRun:
    return source_run(
        repo / ".pingstore",
        identity,
        stage=stage,
        experiment=recipe.SLUG,
        reference=reference,
    )


def configuration(run: SourceRun) -> dict:
    cfg = run.record["execution"].get("configuration")
    if not isinstance(cfg, dict) or cfg.get("schema") not in (
        "exp023.recipe/v4",
        "exp023.recipe/v5",
    ):
        raise PingstoreError("exp023 processing requires unified graph recipe v4 or v5")
    if cfg != recipe.configuration(
        smoke=cfg.get("profile") == "smoke",
        version=int(cfg["schema"].rsplit("v", 1)[1]),
    ):
        raise PingstoreError("exp023 recipe differs from the explicit graph model")
    if run.record["inputs"]:
        raise PingstoreError("exp023 initial compute must not have upstream inputs")
    return cfg
