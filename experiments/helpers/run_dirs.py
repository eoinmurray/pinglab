"""Prepare a notebook's artifact + figure directories at the start of a run.

Wipes the per-notebook artifact/figure dirs (unless the runner was given
--no-wipe-dir), recreates them, persists the run-id counter so it survives
the wipe, and stamps the run-provenance manifest (helpers.provenance). This
is the boilerplate every runner does between parsing args and the first
training cell.

`skip_training=True` keeps the existing artifacts (only the figures dir is
wiped) so figures can be re-rendered from cached sim output. `make_artifacts`
controls whether the artifacts dir is (re)created — runners that only read
another notebook's artifacts leave it False.

`scale` is the runner's declared run scale (max_samples, epochs, t_ms, ...),
stamped into the manifest and rendered as the entry's Methods table. Optional
only until the tier-retirement sweep migrates every runner; it then becomes
required for training notebooks. `host` records where the training cells run
("local" or "modal:<GPU>").

Note the manifest describes THIS invocation: on a --skip-training re-render
the figures are fresh but the artifacts they were drawn from may predate the
recorded sha (upstream staleness is a planned phase-2 check).
"""

from __future__ import annotations

import shutil

from .paths import REPO, artifacts_and_figures
from .provenance import write_manifest
from .run_id import persist as persist_run_id


def prepare(
    slug: str,
    run_id: str,
    *,
    wipe: bool = True,
    skip_training: bool = False,
    make_artifacts: bool = True,
    scale: dict | None = None,
    host: str = "local",
):
    """Wipe (optional) + recreate the slug's dirs, persist run id + manifest.

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
    write_manifest(figures, slug=slug, run_id=run_id, scale=scale, host=host)
    return artifacts, figures
