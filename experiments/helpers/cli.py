"""The closed meta-flag vocabulary for experiment runners + a strict parser.

The demolab invariant is "the runner is the recipe": science parameters are
hardcoded literals in the runner, never CLI flags. Only a small, CLOSED set of
*meta* flags — which change how/whether a run executes or re-renders, never the
science — may be accepted. This module is the single source of truth for that
set; the arg-allowlist gate imports the same lists so the gate and the parser
can never drift.

Two tiers:
  • lifecycle — pipeline stage selection, valid on any runner. A runner is up to
    three ordered, dependent stages (train → analyze → plot); the cost gradient
    (train ≫ analyze > plot) means you only ever skip the expensive PREFIX, so
    two flags cover every case.
  • dispatch — the RunPod fan-out surface, opt-in (allow_dispatch=True) for the
    fan-out runners only. The orchestration itself lives in helpers/runpod.py.

Parsing is bare-argv (no argparse) but STRICT: an unknown --flag is a hard error,
so a typo like `--skip-trianing` fails loudly instead of silently training.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

# ── Lifecycle flags (any runner) ──────────────────────────────────────────
# Booleans, no value.
LIFECYCLE_BOOL = ("--skip-training", "--only-missing")
# Optional value: bare = all figures, `--plot-only <fig>` = just that one.
LIFECYCLE_OPTVALUE = ("--plot-only",)
LIFECYCLE_FLAGS = (*LIFECYCLE_BOOL, *LIFECYCLE_OPTVALUE)

# ── Dispatch flags (opt-in, fan-out runners only) ─────────────────────────
DISPATCH_BOOL = ("--runpod", "--live", "--collect", "--reap", "--pod-run", "--plumbing")
DISPATCH_VALUE = ("--gpu", "--cells-per-pod")   # required single value
DISPATCH_MULTIVALUE = ("--only-cells",)         # consumes tokens until the next --flag
DISPATCH_FLAGS = (*DISPATCH_BOOL, *DISPATCH_VALUE, *DISPATCH_MULTIVALUE)

# Everything the gate treats as allowed meta (plus --help).
ALL_META_FLAGS = frozenset((*LIFECYCLE_FLAGS, *DISPATCH_FLAGS, "--help"))

_GPU_CHOICES = ("4090", "5090")


@dataclass
class Meta:
    """Parsed meta flags. `plot_only` is True when `--plot-only` is present;
    `plot_fig` holds its optional value (None = redraw all figures)."""

    # lifecycle
    skip_training: bool = False
    only_missing: bool = False
    plot_only: bool = False
    plot_fig: str | None = None
    # dispatch (only populated when the runner opts in)
    runpod: bool = False
    live: bool = False
    collect: bool = False
    reap: bool = False
    pod_run: bool = False
    plumbing: bool = False
    # 5090 is the default pool: measured more reliable to provision than 4090
    # (which hit an account-wide Low-stock crunch on 2026-07-09 and stalled on
    # startup), and faster compute narrows the ~40%/hr price gap. Override per
    # run with --gpu 4090 when 4090 stock is healthy and you want the cheaper box.
    gpu: str = "5090"
    cells_per_pod: int = 9
    only_cells: list[str] = field(default_factory=list)

    @property
    def start_stage(self) -> str:
        """First pipeline stage to run, from the lifecycle flags."""
        if self.plot_only:
            return "plot"
        if self.skip_training:
            return "analyze"
        return "train"


def _usage(prog: str, *, allow_dispatch: bool) -> str:
    lines = [
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
    ]
    if allow_dispatch:
        lines += [
            "",
            "  RunPod fan-out (see helpers/runpod.py):",
            "  --runpod           dispatch the fleet (DRY-RUN unless --live)",
            "  --live             actually create pods and spend money",
            "  --collect          pull trained cells off the shared volume",
            "  --reap             terminate all pods (kill switch)",
            "  --gpu {4090,5090}  GPU to provision",
            "  --cells-per-pod N  sweep cells packed per pod",
            "  --only-cells A B   restrict to named cells",
            "  --plumbing         tiny wiring-test scale (cheap pod smoke)",
            "  --pod-run          internal pod-side entrypoint",
        ]
    return "\n".join(lines)


def parse_meta(argv: list[str], *, allow_dispatch: bool = False) -> Meta:
    """Parse the closed meta vocabulary out of argv[1:]. STRICT: any unknown
    --flag is a hard error. Dispatch flags error unless allow_dispatch is set."""
    prog = argv[0].rsplit("/", 1)[-1] if argv else "experiment"
    allowed = set(LIFECYCLE_FLAGS)
    if allow_dispatch:
        allowed |= set(DISPATCH_FLAGS)

    meta = Meta()
    i = 1
    toks = argv
    while i < len(toks):
        tok = toks[i]
        if tok == "--help":
            print(_usage(prog, allow_dispatch=allow_dispatch))
            sys.exit(0)
        if not tok.startswith("--"):
            raise SystemExit(f"{prog}: unexpected argument {tok!r} (positional args are not accepted)")
        if tok not in allowed:
            hint = ""
            if tok in DISPATCH_FLAGS:
                hint = " (dispatch flag — this runner has no fan-out backend)"
            raise SystemExit(f"{prog}: unknown flag {tok!r}{hint}. Try --help.")

        # lifecycle
        if tok == "--skip-training":
            meta.skip_training = True
        elif tok == "--only-missing":
            meta.only_missing = True
        elif tok == "--plot-only":
            meta.plot_only = True
            # optional value: consume the next token ONLY if it isn't a flag.
            if i + 1 < len(toks) and not toks[i + 1].startswith("--"):
                meta.plot_fig = toks[i + 1]
                i += 1
        # dispatch
        elif tok == "--runpod":
            meta.runpod = True
        elif tok == "--live":
            meta.live = True
        elif tok == "--collect":
            meta.collect = True
        elif tok == "--reap":
            meta.reap = True
        elif tok == "--pod-run":
            meta.pod_run = True
        elif tok == "--plumbing":
            meta.plumbing = True
        elif tok == "--gpu":
            i += 1
            if i >= len(toks):
                raise SystemExit(f"{prog}: --gpu requires a value {_GPU_CHOICES}")
            meta.gpu = toks[i]
            if meta.gpu not in _GPU_CHOICES:
                raise SystemExit(f"{prog}: --gpu must be one of {_GPU_CHOICES}, got {meta.gpu!r}")
        elif tok == "--cells-per-pod":
            i += 1
            if i >= len(toks):
                raise SystemExit(f"{prog}: --cells-per-pod requires an integer")
            try:
                meta.cells_per_pod = int(toks[i])
            except ValueError:
                raise SystemExit(f"{prog}: --cells-per-pod must be an integer, got {toks[i]!r}") from None
        elif tok == "--only-cells":
            # consume tokens until the next --flag.
            j = i + 1
            while j < len(toks) and not toks[j].startswith("--"):
                meta.only_cells.append(toks[j])
                j += 1
            if not meta.only_cells:
                raise SystemExit(f"{prog}: --only-cells requires at least one cell name")
            i = j - 1
        i += 1

    return meta


# Back-compat shim: the old figure re-render selector. `--plot-only <fig>`
# subsumes it; kept so any un-migrated caller importing replot_target still runs.
def replot_target(argv: list[str]) -> str | None:
    """Deprecated: the figure name after `--plot-only` (or the old `--replot`)."""
    for flag in ("--plot-only", "--replot"):
        if flag in argv:
            i = argv.index(flag)
            if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                return argv[i + 1]
    return None
