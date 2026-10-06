"""Validated v4 inputs only, with exact upstream pins and no active-run fallback."""

from pathlib import Path

from pingstore.contracts import PingstoreError
from pingstore.stages import SourceRun, source_run

from . import recipe


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
    if not isinstance(cfg, dict) or cfg.get("schema") != "exp023.recipe/v3":
        raise PingstoreError("exp023 processing requires native graph recipe v3")
    if cfg != recipe.configuration(smoke=cfg.get("profile") == "smoke"):
        raise PingstoreError("exp023 recipe differs from the explicit graph model")
    if run.record["inputs"]:
        raise PingstoreError("exp023 initial compute must not have upstream inputs")
    return cfg
