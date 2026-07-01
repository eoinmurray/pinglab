#!/usr/bin/env python3
"""Render manuscript figures for paper001 at print quality.

Re-runs each source notebook's *inference / plotting* path (never training)
with the shared print profile (theme.set_paper_mode) so every figure is a
vector PDF, sized to a single column/full width, with uniform print fonts and
rule weights. The notebook is still the recipe — we only override rendering:

  * theme is flipped into PAPER_MODE (smaller absolute fonts, thin rules, PDF);
  * plt.subplots / plt.figure / Figure.set_size_inches widths are forced to the
    figure's target width, keeping each figure's own aspect ratio, so on-page
    type is homogeneous across the whole set;
  * each notebook's FIGURES dir is redirected to a private scratch dir, so docs/
    is never touched and notebooks can't clobber each other's output;
  * Figure.savefig is coerced to vector PDF.

Run:  uv run python src/papers/paper001/render_figures.py [slug ...]
With no args, renders every figure in REGISTRY.
"""
from __future__ import annotations

import importlib
import shutil
import sys
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parent
REPO = PAPER_DIR.parents[2]
FIG_DIR = PAPER_DIR / "figures"
SCRATCH = PAPER_DIR / ".render_tmp"
FIG_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "src" / "notebooks"))

# Width tiers (inches) matched to how the figure is included in main.tex.
COLUMN = 5.6   # single-panel plots, \includegraphics[width=\linewidth]
FULL = 6.9     # wide multi-panel compounds (eLife fullwidth)

# slug -> (target width, [figure stems to lift into the manuscript]).
# Each stem is the notebook's own savefig basename; the manuscript file is
# <slug>_<stem>.pdf to match the \includegraphics calls in main.tex.
REGISTRY = {
    "nb023": (FULL, ["overview_compound"]),
    "nb057": (FULL, ["onset_super_compound"]),
    "nb025": (FULL, ["results_compound"]),
    "nb038": (FULL, ["loop_transfer_compound"]),
    "nb049": (COLUMN, ["training_curves"]),
    "nb041": (COLUMN, ["rate_vs_fgamma"]),
    "nb046": (FULL, ["spikes_per_cycle_distribution"]),
    "nb047": (COLUMN, ["rate_vs_w_ie"]),
    "nb037": (COLUMN, ["perturbation_curves"]),
    "nb042": (FULL, ["rhythm_compound"]),
    "nb044": (COLUMN, ["dt_sweep"]),
    "nb048": (FULL, ["varying_headline_stream", "acc_grid_tau_rate"]),
}

# argv handed to each notebook's main(). Several expose a fast "just the
# compound" path that skips video/sweep work (and its fragility); the rest
# run inference-only via --skip-training. Default covers the simple cases.
_DEFAULT_ARGS = ["--skip-training", "--no-wipe-dir"]
ARGS = {
    "nb025": ["--compound-only"],
    "nb038": ["--compound-only"],
    "nb042": ["--compound-only"],
    "nb049": ["--curves-only"],
    "nb046": ["--no-wipe-dir"],  # argparse-based; rejects --skip-training
}

# Set per notebook before its main() runs; read by the patched figure makers.
_TARGET_W = COLUMN
_PATCHED = False


def _scaled(w, h):
    return (_TARGET_W, h * _TARGET_W / float(w))


def _install_patches():
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True
    import matplotlib.figure
    import matplotlib.pyplot as plt

    _orig_subplots = plt.subplots
    _orig_figure = plt.figure
    _orig_savefig = matplotlib.figure.Figure.savefig
    _orig_setsize = matplotlib.figure.Figure.set_size_inches

    def subplots(*args, **kwargs):
        w, h = kwargs.get("figsize") or (8.0, 6.0)
        kwargs["figsize"] = _scaled(w, h)
        kwargs.pop("dpi", None)
        return _orig_subplots(*args, **kwargs)

    def figure(*args, **kwargs):
        fs = kwargs.get("figsize")
        if fs is None and args and isinstance(args[0], (tuple, list)):
            fs = args[0]
            args = args[1:]
        if fs is not None:
            kwargs["figsize"] = _scaled(*fs)
        kwargs.pop("dpi", None)
        return _orig_figure(*args, **kwargs)

    def set_size_inches(self, *args, **kwargs):
        if len(args) >= 2:
            w, h = args[0], args[1]
            rest = args[2:]
        else:
            w, h = args[0]
            rest = ()
        return _orig_setsize(self, *_scaled(w, h), *rest, **kwargs)

    def savefig(self, fname, *args, **kwargs):
        p = Path(fname)
        if p.suffix.lower() != ".pdf":
            p = p.with_suffix(".pdf")
        kwargs["format"] = "pdf"
        kwargs.pop("dpi", None)
        return _orig_savefig(self, p, *args, **kwargs)

    plt.subplots = subplots
    plt.figure = figure
    matplotlib.figure.Figure.savefig = savefig
    matplotlib.figure.Figure.set_size_inches = set_size_inches


def render(slug: str) -> None:
    global _TARGET_W
    width, stems = REGISTRY[slug]
    _TARGET_W = width

    import matplotlib.pyplot as plt
    from helpers import theme
    theme.set_paper_mode(True)
    _install_patches()

    scratch = SCRATCH / slug
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)

    # Seed the scratch dir with the committed figure outputs (numbers.json and
    # friends) so notebooks whose --compound-only path *reads back* their own
    # cached summary find it. Savefig still overwrites the wanted stems as PDF.
    docs_figs = REPO / "src" / "docs" / "public" / "figures" / "notebooks" / slug
    if docs_figs.is_dir():
        for f in docs_figs.iterdir():
            if f.is_file():
                shutil.copyfile(f, scratch / f.name)

    mod = importlib.import_module(slug)
    mod.FIGURES = scratch  # redirect every write off docs/ and out of the way

    print(f"\n[render] {slug} at width={width}in")
    sys.argv = [slug, *ARGS.get(slug, _DEFAULT_ARGS)]
    try:
        mod.main()
    except (Exception, SystemExit) as exc:  # keep the batch alive (SystemExit
        # is a BaseException — a bare sys.exit in a notebook must not abort it)
        print(f"[warn] {slug} main() raised: {type(exc).__name__}: {exc}")
    plt.close("all")

    for stem in stems:
        src = scratch / f"{stem}.pdf"
        dst = FIG_DIR / f"{slug}_{stem}.pdf"
        if src.exists():
            shutil.copyfile(src, dst)
            print(f"[ok] {dst.relative_to(PAPER_DIR)}")
        else:
            print(f"[MISS] {slug}: expected {stem}.pdf not produced")


def main() -> None:
    slugs = list(sys.argv[1:]) or list(REGISTRY)  # snapshot before main() eats argv
    for slug in slugs:
        if slug not in REGISTRY:
            print(f"[skip] unknown slug {slug}")
            continue
        render(slug)
    if SCRATCH.exists():
        shutil.rmtree(SCRATCH)


if __name__ == "__main__":
    main()
