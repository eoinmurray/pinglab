"""Validated v4 runs for exp081; no legacy or implicit selection fallback.

The owning experiment calls these local definitions from its explicit stages.
Inputs and outputs retain that stage’s validated scientific contract; importing
this module does not start computation or select a scientific run.
"""

from pathlib import Path

from experiments.exp081 import recipe
from pingstore.library import PingstoreError, SourceRun, source_run


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
    cfg = run.details["execution"].get("configuration")
    if not isinstance(cfg, dict) or cfg.get("schema") != "exp081.recipe/v1":
        raise PingstoreError("exp081 requires a retained scientific recipe")
    return cfg
