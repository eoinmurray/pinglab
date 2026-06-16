"""Canonical artifact + figure directories for a notebook slug.

Every runner writes simulation artifacts under
`src/artifacts/notebooks/<slug>/` and frozen figures under
`src/docs/public/figures/notebooks/<slug>/`. `artifacts_and_figures(slug)`
returns that pair so individual runners don't re-spell the paths.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ARTIFACTS_ROOT = REPO / "src" / "artifacts" / "notebooks"
FIGURES_ROOT = REPO / "src" / "docs" / "public" / "figures" / "notebooks"


def artifacts_and_figures(slug: str) -> tuple[Path, Path]:
    """Return (artifacts_dir, figures_dir) for a notebook slug (e.g. "nb024")."""
    return ARTIFACTS_ROOT / slug, FIGURES_ROOT / slug
