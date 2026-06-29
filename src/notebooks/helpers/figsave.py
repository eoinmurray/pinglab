"""Emit a figure once, in every format the project consumes.

The docs site embeds vector **SVG** (browser-native, themeable, scales without
the font-shrink you get from rescaling a fixed-size PDF); the LaTeX manuscript
embeds vector **PDF**. Both come from the same Matplotlib figure object, so the
notebook stays the single recipe — there is no separate render pipeline that
re-runs the plot at print size.

Sizing/typography is whatever profile is active when the figure was built
(see cli.theme.set_paper_mode): set paper mode before plotting and the same
artifact is print-correct in both places.

Usage:
    save_figure(fig, FIGURES / "dt_sweep")   # -> dt_sweep.svg, dt_sweep.pdf
"""
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
