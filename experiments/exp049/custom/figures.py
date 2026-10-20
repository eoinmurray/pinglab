"""Figure exports for exp049.

Write SVG and PDF from the same figure by default, preserving the active theme
settings and any formats explicitly requested by this experiment."""

from __future__ import annotations

from pathlib import Path

DEFAULT_FORMATS = ("svg", "pdf")


def save_figure(fig, stem, formats=DEFAULT_FORMATS) -> list[Path]:
    """Write *fig* to <stem>.<ext> for each ext in *formats*.

    *stem* is a path without an extension (a trailing .png/.pdf/.svg is
    stripped, so existing call sites that pass "foo.png" keep working).
    Format is inferred from the extension; rcParams (dpi, bbox, fonttype)
    come from the active theme profile.
    """
    stem = Path(stem)
    if stem.suffix:
        stem = stem.with_suffix("")
    out: list[Path] = []
    for ext in formats:
        p = stem.with_suffix(f".{ext}")
        fig.savefig(p)
        out.append(p)
    return out
