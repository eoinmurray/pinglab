"""Run-id stamping for experiment figures."""

from __future__ import annotations


def stamp_figure(fig, run_id: str) -> None:
    """Stamp run_id into the bottom-right corner of a matplotlib figure."""
    from . import theme

    fig.text(
        0.995,
        0.005,
        run_id,
        ha="right",
        va="bottom",
        fontsize=theme.SIZE_CAPTION,
        color=theme.LABEL,
        family="monospace",
    )
