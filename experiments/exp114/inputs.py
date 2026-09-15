"""Authenticated v4 input readers for exp114."""

from pathlib import Path

from pingstore.contracts import PingstoreError
from pingstore.stages import source_run

from . import recipe


def source(repo: Path, identity: str, stage: str, *, reference=None):
    return source_run(repo / ".pingstore", identity, stage=stage, experiment=recipe.SLUG, reference=reference)


def configuration(run):
    cfg = run.record["execution"].get("configuration")
    if not isinstance(cfg, dict) or cfg.get("schema") != "exp114.recipe/v1":
        raise PingstoreError("exp114 requires recipe v1")
    return cfg
