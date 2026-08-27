"""Validated v3 evidence for exp081; no legacy or implicit selection fallback."""

from pathlib import Path

from pingstore.contracts import RUN_SCHEMA, PingstoreError
from pingstore.stages import SourceRun, source_run

from . import recipe


def source(
    repo: Path, identity: str, stage: str, *, reference: dict | None = None
) -> SourceRun:
    result = source_run(
        repo / ".pingstore",
        identity,
        stage=stage,
        experiment=recipe.SLUG,
        reference=reference,
    )
    if result.record["schema"] != RUN_SCHEMA:
        raise PingstoreError("exp081 requires v3 evidence; legacy v2 is not accepted")
    return result


def configuration(run: SourceRun) -> dict:
    cfg = run.record["execution"].get("configuration")
    if not isinstance(cfg, dict) or cfg.get("schema") != "exp081.recipe/v1":
        raise PingstoreError("exp081 requires a retained scientific recipe")
    return cfg
