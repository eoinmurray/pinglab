---
layout: ../layouts/MarkdownLayout.astro
title: "The Notebooks"
---

# The Notebooks

A notebook entry is a reproducible unit of work. Each entry pairs a writeup with a runner script that regenerates every figure, video, and number the writeup cites. There are no loose plots — if a figure isn't emitted by a runner, it isn't published.

## Anatomy of an entry

Every entry is a triple, keyed by a zero-padded slug *nbNNN*:

1. **Runner** — *src/pinglab/notebooks/nbNNN.py*. Single-file Python script; the promotion gate for the entry.
2. **Artifacts** — *src/artifacts/notebooks/&lt;slug&gt;/*. Raw run outputs (gitignored).
3. **Published figures** — *src/docs/public/figures/notebooks/&lt;slug&gt;/*. Figures, videos, *numbers.json*, *_run.txt*. The MDX imports from here.
4. **Entry** — *src/docs/src/pages/notebooks/nbNNN.mdx*. Writeup with Introduction / Method / Findings / Implications / Next steps.

## Runner contract

Every runner:

- Sets *SLUG = "nbNNN"* at module scope, which drives the artifact and figure paths.
- Wipes its output directories by default on start, so stale artifacts can't masquerade as current ones.
- Records a per-notebook monotonic *rNNN* run identifier via *_run_id.py* and stamps it onto published videos.
- Dispatches computation through *src/pinglab/oscilloscope.py* — the notebook script never touches models directly. All scientific knobs are hardcoded as literals in the runner's CLI args list, so reproducing a result never requires remembering flags.

## CLI

Runners accept exactly three flags:

- *--tier &lt;name&gt;* — run size. Tier names come from the [Run sizing tiers](/styleguide/#10-run-sizing-tiers) table in the styleguide. Smaller tiers are for iteration, larger tiers for published runs.
- *--modal-gpu &lt;spec&gt;* — dispatch the oscilloscope calls to Modal instead of running locally. Absent = local.
- *--skip-training* — reuse existing artifacts and regenerate only the published figures. Useful when only the plotting code changed.

An implicit *--no-wipe-dir* opt-out exists but is rarely useful; cache drift is how figures fall out of sync with code.

## Why this shape

The discipline is that an entry must be fully specified by its runner. Scientific parameters belong in the runner as literals, not on the CLI — because the notebook *is* the recipe. Hyperparameters promoted to CLI flags drift out of the writeup over time; hardcoded literals move with the code and survive git history.

New scientific knobs go onto *oscilloscope.py* as flags; the notebook just passes the recipe value inline.
