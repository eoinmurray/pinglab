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
("local"; cloud fan-outs run under the RunPod backend, see helpers/runpod.py).

Note the manifest describes THIS invocation: on a --skip-training re-render
the figures are fresh but the artifacts they were drawn from may predate the
recorded sha (upstream staleness is a planned phase-2 check).
"""

from __future__ import annotations

import contextlib
import os
import shutil

from .paths import REPO, artifacts_and_figures
from .provenance import write_manifest
from .run_id import COUNTER_FILE
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
        # A full run wipes the whole artifacts (cached sim output) dir; skip
        # training keeps it. The FIGURES dir is refreshed by removing only its
        # top-level FILES — auxiliary SUBDIRECTORIES (e.g. exp022's rasters/,
        # produced by a separate `--plot-only appendix-rasters` pass and
        # expensive to rebuild) are preserved, so a figure refresh never silently
        # nukes them.
        if not skip_training and artifacts.exists():
            print(f"[wipe] {artifacts.relative_to(REPO)}")
            shutil.rmtree(artifacts)
        if figures.exists():
            print(f"[wipe] {figures.relative_to(REPO)} (top-level files; subdirs kept)")
            for p in figures.iterdir():
                if p.is_file():
                    p.unlink()
    if make_artifacts:
        artifacts.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    persist_run_id(slug, run_id)
    write_manifest(figures, slug=slug, run_id=run_id, scale=scale, host=host)
    return artifacts, figures


# ── Atomic-publish API (the standard; supersedes wipe-at-start `prepare`) ──
# A run writes into a STAGING dir; the published dir is swapped in only when the
# run completes. A failed run never touches the published dir and keeps its
# staging dir for post-mortem. This replaces the destructive rmtree-at-start:
# old data stays visible during a long run, and a crash loses nothing published.
#
# `prepare` above is retained only for runners not yet migrated (e.g. exp037).


def _staging_dir(figures):
    """Sibling of the published figures dir (same filesystem → atomic rename)."""
    return figures.parent / f"{figures.name}.staging"


def prepare_staged(
    slug: str,
    run_id: str,
    *,
    skip_training: bool = False,
    make_artifacts: bool = True,
    scale: dict | None = None,
    host: str = "local",
    plot_only: bool = False,
):
    """Set up a staged run. Returns (artifacts_dir, staging_figures_dir).

    The runner writes ALL figures + numbers.json into the returned staging dir;
    `publish` swaps it into place on success. The artifacts (scratch/cache) dir
    is created but NOT wiped — expensive weights survive across runs; resume is
    the runner's job (e.g. --only-missing). For `plot_only`, the current
    published figures are copied into staging first so redrawing one figure does
    not drop the others.
    """
    artifacts, figures = artifacts_and_figures(slug)
    staging = _staging_dir(figures)
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    if plot_only and figures.exists():
        # Seed staging from the live published dir so an unchanged figure carries
        # over untouched when only one is redrawn.
        shutil.copytree(figures, staging, dirs_exist_ok=True)
    if make_artifacts:
        artifacts.mkdir(parents=True, exist_ok=True)
    # Counter lives in staging so it publishes atomically with the run.
    n = int(run_id.lstrip("r"))
    (staging / COUNTER_FILE).write_text(f"{n}\n")
    write_manifest(staging, slug=slug, run_id=run_id, scale=scale, host=host)
    return artifacts, staging


def publish(slug: str, run_id: str):
    """Atomically swap this run's staging dir into the published location.
    Called only on a successful run. Returns the published figures dir."""
    _artifacts, figures = artifacts_and_figures(slug)
    staging = _staging_dir(figures)
    if not staging.exists():
        raise RuntimeError(f"no staging dir to publish for {slug} (expected {staging})")
    old = figures.parent / f"{figures.name}.old-{run_id}"
    if old.exists():
        shutil.rmtree(old)
    if figures.exists():
        os.rename(figures, old)          # move the live dir aside …
    os.rename(staging, figures)          # … then swap staging in (atomic rename)
    if old.exists():
        shutil.rmtree(old)               # drop the previous run's output
    return figures


@contextlib.contextmanager
def published_run(slug: str, run_id: str, **kwargs):
    """Context manager: stage → (run body) → publish on success, keep on failure.

    Usage:
        with published_run(SLUG, run_id, scale=SCALE) as (artifacts, figures):
            ...  # write everything into `figures`
    """
    artifacts, staging = prepare_staged(slug, run_id, **kwargs)
    try:
        yield artifacts, staging
    except BaseException:
        print(f"[FAILED] run did not publish; staging kept for post-mortem: {staging}")
        raise
    else:
        published = publish(slug, run_id)
        print(f"[published] {published}")
