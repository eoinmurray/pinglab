"""Validated v3 inputs only, with exact upstream pins and no active-run fallback."""

from pathlib import Path

from pingstore.contracts import RUN_SCHEMA, PingstoreError
from pingstore.stages import SourceRun, source_run

from . import recipe


def source(
    repo: Path, identity: str, stage: str, *, reference: dict | None = None
) -> SourceRun:
    run = source_run(
        repo / ".pingstore",
        identity,
        stage=stage,
        experiment=recipe.SLUG,
        reference=reference,
    )
    if run.record["schema"] != RUN_SCHEMA:
        raise PingstoreError("exp023 requires v3 evidence; legacy v2 is not accepted")
    return run


def configuration(run: SourceRun) -> dict:
    cfg = run.record["execution"].get("configuration")
    if not isinstance(cfg, dict) or cfg.get("schema") != "exp023.recipe/v1":
        raise PingstoreError("exp023 requires a retained scientific recipe")
    if run.record["inputs"]:
        raise PingstoreError("exp023 initial compute must not have upstream inputs")
    return cfg
