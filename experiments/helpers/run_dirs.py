"""Build experiments directly inside hidden Pingstore run directories.

Local runners write state, logs, intermediates, and derived output beneath
`.pingstore/runs/.<run-id>.tmp/`. Completion writes `run.json` and
atomically exposes the immutable run; failure leaves the hidden run intact for
post-mortem inspection. `.artifacts/` is refreshed afterward only as a direct
publication view.

`scale` is the runner's declared run scale (max_samples, epochs, t_ms, ...),
stamped into the manifest and rendered as the entry's Methods table. Optional
only until the tier-retirement sweep migrates every runner; it then becomes
required for training notebooks. `host` records where the training cells run
("local"; cloud fan-outs run under the RunPod backend, see helpers/runpod.py).

Skip-training and plot-only runs seed their hidden run from the active immutable
run rather than from shared mutable scratch.
"""

from __future__ import annotations

import contextlib
import functools
import os
import shutil

from pingstore.layout import copy_legacy_derived, initialize_layout
from pingstore.materialize import materialize_run
from pingstore.native import (
    execution_origin,
    finalize_local_run,
    make_run_id,
)

from .paths import (
    FIGURES_ROOT,
    REPO,
    active_run_state,
    artifacts_and_figures,
    runner_paths,
)
from .provenance import write_manifest
from .run_id import COUNTER_FILE


def _display_path(path):
    try:
        return path.relative_to(REPO)
    except ValueError:
        return path


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
    """Create the run's state and derived paths and write its manifest.

    Returns (artifacts_dir, figures_dir).
    """
    if runner_paths(slug).isolated:
        artifacts, figures = artifacts_and_figures(slug)
        if make_artifacts:
            artifacts.mkdir(parents=True, exist_ok=True)
        figures.mkdir(parents=True, exist_ok=True)
        write_manifest(figures, slug=slug, run_id=run_id, scale=scale, host=host)
        return artifacts, figures
    return _prepare_local_working_run(
        slug,
        run_id,
        make_artifacts=make_artifacts,
        scale=scale,
        host=host,
        seed_previous=skip_training,
    )


# Isolated collection execution retains explicit paths until its orchestration
# layer captures them. Direct runners use the hidden run path assembled below.


def _staging_dir(figures):
    """Sibling of the published figures dir (same filesystem → atomic rename)."""
    return figures.parent / f"{figures.name}.staging"


def _prepare_local_working_run(
    slug: str,
    run_id: str,
    *,
    make_artifacts: bool,
    scale: dict | None,
    host: str,
    seed_previous: bool,
    seed_derived: bool = False,
):
    full_run_id = make_run_id(slug, run_id, execution_origin(host))
    temporary = REPO / ".pingstore" / "runs" / f".{full_run_id}.tmp"
    files = temporary / "presentation"
    if (files / "_manifest.json").exists() or (temporary / "run.json").exists():
        raise RuntimeError(f"incomplete run already exists: {temporary}")
    state = temporary / "export/state"
    initialize_layout(temporary, slug)
    files.mkdir(parents=True, exist_ok=True)
    active = FIGURES_ROOT / slug
    if seed_derived and active.is_dir():
        copy_legacy_derived(active, temporary)
    if seed_previous and (active / "_manifest.json").is_file():
        try:
            previous_state = active_run_state(slug)
        except (FileNotFoundError, RuntimeError):
            previous_state = None
        if previous_state is not None and previous_state.is_dir():
            shutil.copytree(previous_state, state, dirs_exist_ok=True)
    if make_artifacts:
        state.mkdir(parents=True, exist_ok=True)
    write_manifest(files, slug=slug, run_id=run_id, scale=scale, host=host)
    return state, files


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
    """Return state and derived paths inside this run's hidden directory."""
    if not runner_paths(slug).isolated:
        return _prepare_local_working_run(
            slug,
            run_id,
            make_artifacts=make_artifacts,
            scale=scale,
            host=host,
            seed_previous=True,
            seed_derived=plot_only,
        )
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
        os.rename(figures, old)  # move the live dir aside …
    os.rename(staging, figures)  # … then swap staging in (atomic rename)
    if old.exists():
        shutil.rmtree(old)  # drop the previous run's output
    return figures


def finalize_prepared_run(
    slug: str,
    run_id: str,
    *,
    scale: dict | None = None,
    host: str = "local",
):
    """Finalize a legacy runner that wrote directly into its hidden run."""
    if runner_paths(slug).isolated:
        return None
    _artifacts, figures = artifacts_and_figures(slug)
    manifest = figures / "_manifest.json"
    if not manifest.is_file():
        write_manifest(figures, slug=slug, run_id=run_id, scale=scale, host=host)
    run = finalize_local_run(REPO, slug, figures.parent)
    _materialize_result(run)
    return FIGURES_ROOT / slug


def _materialize_result(run: dict) -> None:
    """Refresh the active artifact view from the newly completed run."""
    materialize_run(REPO / ".pingstore", run["run_id"], FIGURES_ROOT)


def preserve_active_view(slug: str):
    """Preserve isolated legacy views; local failures keep their hidden run."""

    def decorate(function):
        @functools.wraps(function)
        def guarded(*args, **kwargs):
            if not runner_paths(slug).isolated:
                return function(*args, **kwargs)
            _artifacts, figures = artifacts_and_figures(slug)
            backup = figures.with_name(figures.name + ".pre-run")
            if backup.exists():
                raise RuntimeError(f"legacy run backup already exists: {backup}")
            existed = figures.exists()
            if existed:
                shutil.copytree(figures, backup)
            try:
                result = function(*args, **kwargs)
            except BaseException:
                if figures.exists():
                    shutil.rmtree(figures)
                if existed:
                    os.rename(backup, figures)
                raise
            else:
                if backup.exists():
                    shutil.rmtree(backup)
                return result

        return guarded

    return decorate


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
        # A direct local run becomes an immutable Pingstore ExperimentRun before
        # the historical Demolab artifact view is swapped. Isolated campaign
        # runners already own their run root and are migrated by collection
        # orchestration rather than duplicated here.
        if not runner_paths(slug).isolated:
            run = finalize_local_run(REPO, slug, staging.parent)
            _materialize_result(run)
            published = FIGURES_ROOT / slug
        else:
            published = publish(slug, run_id)
        print(f"[published] {published}")
