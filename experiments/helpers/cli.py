"""The closed meta-flag vocabulary for experiment runners + a strict parser.

The demolab invariant is "the runner is the recipe": science parameters are
hardcoded literals in the runner, never CLI flags. Only a small, closed set of
meta flags may change how or whether a run executes or re-renders.

Parsing is bare-argv (no argparse) but strict: an unknown flag is a hard error,
so a typo fails loudly instead of silently changing execution.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

LIFECYCLE_BOOL = ("--skip-training", "--only-missing")
LIFECYCLE_OPTVALUE = ("--plot-only",)
LIFECYCLE_FLAGS = (*LIFECYCLE_BOOL, *LIFECYCLE_OPTVALUE)

ALL_META_FLAGS = frozenset((*LIFECYCLE_FLAGS, "--help"))


@dataclass
class Meta:
    """Parsed lifecycle flags."""

    skip_training: bool = False
    only_missing: bool = False
    plot_only: bool = False
    plot_fig: str | None = None

    @property
    def start_stage(self) -> str:
        """First pipeline stage to run, from the lifecycle flags."""
        if self.plot_only:
            return "plot"
        if self.skip_training:
            return "analyze"
        return "train"


def _usage(prog: str) -> str:
    return "\n".join([
        f"usage: {prog} [meta-flags]",
        "",
        "Pipeline stages run train → analyze → plot. Meta flags pick the suffix;",
        "science parameters are hardcoded in the runner, never flags.",
        "",
        "  (no flag)          full run: train + analyze + plot",
        "  --skip-training    reuse cached weights: analyze + plot",
        "  --plot-only [FIG]  redraw from cache only; bare = all figures, FIG = one",
        "  --only-missing     train only cells lacking a valid marker",
        "  --help             show this message",
    ])


def parse_meta(argv: list[str]) -> Meta:
    """Parse the closed meta vocabulary out of ``argv[1:]``."""
    prog = argv[0].rsplit("/", 1)[-1] if argv else "experiment"
    allowed = set(LIFECYCLE_FLAGS)
    meta = Meta()
    i = 1
    while i < len(argv):
        tok = argv[i]
        if tok == "--help":
            print(_usage(prog))
            sys.exit(0)
        if not tok.startswith("--"):
            raise SystemExit(
                f"{prog}: unexpected argument {tok!r} (positional args are not accepted)"
            )
        if tok not in allowed:
            raise SystemExit(f"{prog}: unknown flag {tok!r}. Try --help.")

        if tok == "--skip-training":
            meta.skip_training = True
        elif tok == "--only-missing":
            meta.only_missing = True
        elif tok == "--plot-only":
            meta.plot_only = True
            if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                meta.plot_fig = argv[i + 1]
                i += 1
        i += 1

    return meta
