"""Meta-flag parsing for experiment runners (bare sys.argv, no argparse)."""

from __future__ import annotations


def replot_target(argv: list[str]) -> str | None:
    """The figure name after `--replot`, or None if the flag is absent.

    `--replot <name>` re-renders a single figure from the run's cached artifacts and
    exits, skipping all compute; the runner maps <name> to its render function. This
    is the one meta flag that unifies the old per-runner selectors (--compound-only,
    --replot-grid, --curves-only, --portrait-only, --accrate-only).
    """
    if "--replot" in argv:
        i = argv.index("--replot")
        if i + 1 < len(argv):
            return argv[i + 1]
    return None
