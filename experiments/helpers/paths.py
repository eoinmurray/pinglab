"""Canonical artifact + figure directories for a notebook slug.

Every runner writes scratch simulation artifacts (npz, weights, logs) under
`src/artifacts/notebooks/<slug>/` (gitignored) and frozen figures + numbers.json
under `artifacts/data/<slug>/` — the demolab published-data layout the Typst
writings read from. `artifacts_and_figures(slug)` returns that pair so individual
runners don't re-spell the paths.

(The figure root used to be the Astro site's `src/docs/public/figures/notebooks/`;
it moved to `artifacts/data/` when the site migrated to Typst.)
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ARTIFACTS_ROOT = REPO / "temp" / "notebooks"
FIGURES_ROOT = REPO / "artifacts" / "data"


def artifacts_and_figures(slug: str) -> tuple[Path, Path]:
    """Return (artifacts_dir, figures_dir) for a notebook slug (e.g. "nb024")."""
    return ARTIFACTS_ROOT / slug, FIGURES_ROOT / slug
