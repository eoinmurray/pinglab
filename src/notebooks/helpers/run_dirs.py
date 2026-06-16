"""Prepare a notebook's artifact + figure directories at the start of a run.

Wipes the per-notebook artifact/figure dirs (unless the runner was given
--no-wipe-dir), recreates them, and persists the run-id counter so it survives
the wipe. This is the boilerplate every runner does between parsing args and
the first training cell.

`skip_training=True` keeps the existing artifacts (only the figures dir is
wiped) so figures can be re-rendered from cached sim output. `make_artifacts`
controls whether the artifacts dir is (re)created — runners that only read
another notebook's artifacts leave it False.
"""

from __future__ import annotations

import shutil

from .paths import REPO, artifacts_and_figures
from .run_id import persist as persist_run_id


def prepare(
    slug: str,
    run_id: str,
    *,
    wipe: bool = True,
    skip_training: bool = False,
    make_artifacts: bool = True,
):
    """Wipe (optional) + recreate the slug's dirs and persist its run id.

    Returns (artifacts_dir, figures_dir).
    """
    artifacts, figures = artifacts_and_figures(slug)
    if wipe:
        targets = (figures,) if skip_training else (artifacts, figures)
        for d in targets:
            if d.exists():
                print(f"[wipe] {d.relative_to(REPO)}")
                shutil.rmtree(d)
    if make_artifacts:
        artifacts.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    persist_run_id(slug, run_id)
    return artifacts, figures
