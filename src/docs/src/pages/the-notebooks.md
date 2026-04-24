---
layout: ../layouts/MarkdownLayout.astro
title: "The Notebooks"
---

# The Notebooks

**Last updated:** 2026-04-24 (sources at SHA *dd4d9db*).

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

Runners accept this small, fixed flag set:

- *--tier &lt;name&gt;* — run size. Tier names come from the [Run sizing tiers](/styleguide/#10-run-sizing-tiers) table in the styleguide. Smaller tiers are for iteration, larger tiers for published runs.
- *--modal-gpu &lt;spec&gt;* — dispatch the oscilloscope calls to Modal instead of running locally. Absent = local.
- *--skip-training* — reuse existing artifacts and regenerate only the published figures. Useful when only the plotting code changed.
- *--evaluate-success-only* — re-run the success-criteria gate against the existing *numbers.json* and published figures, without dispatching any training or wiping anything. Useful when only the criteria themselves have changed.
- *--no-wipe-dir* — opt out of the default wipe of artifacts + figures. Rarely useful; cache drift is how figures fall out of sync with code.

Scientific hyperparameters — learning rate, epochs, readout, surrogate slope, regulariser strengths, etc. — are never CLI flags on the runner. They are hardcoded as literals in the runner itself.

## Why this shape

The discipline is that an entry must be fully specified by its runner. Scientific parameters belong in the runner as literals, not on the CLI — because the notebook *is* the recipe. Hyperparameters promoted to CLI flags drift out of the writeup over time; hardcoded literals move with the code and survive git history.

New scientific knobs go onto *oscilloscope.py* as flags; the notebook just passes the recipe value inline.

## Appendix: keeping this page in sync with the code

This page is hand-written — nothing regenerates it — so it drifts whenever the runner contract or the entry layout changes. The refresh is done by Claude Code on demand; the procedure below is the checklist it follows.

### When to refresh

Trigger a refresh whenever any of these change:

- *src/pinglab/notebooks/_run_id.py* or the *SLUG* / wipe-dir / *rNNN* conventions any runner relies on.
- The set of CLI flags a runner accepts (currently *--tier*, *--modal-gpu*, *--skip-training*, *--evaluate-success-only*, *--no-wipe-dir*).
- The artifact / published-figure directory layout under *src/artifacts/notebooks/* or *src/docs/public/figures/notebooks/*.
- The MDX entry skeleton (Introduction / Method / Findings / Implications / Next steps) — see the *feedback_notebook_entry_headings* convention.
- Run-sizing tiers referenced from the [styleguide](/styleguide/#10-run-sizing-tiers).

A quick signal: *git log --since="&lt;last-updated date&gt;" -- src/pinglab/notebooks/ src/docs/src/pages/styleguide.md src/docs/src/pages/notebooks/* returning anything means this page is potentially stale.

### Refresh procedure (for Claude Code)

1. Read a representative sample of *src/pinglab/notebooks/nbNNN.py* runners end-to-end (including the most recent one) plus *_run_id.py*. The runner contract section must match what the scripts actually do.
2. Cross-check the MDX skeleton against *src/docs/src/pages/notebooks/nbNNN.mdx* and the *feedback_notebook_entry_headings* convention.
3. Cross-check the CLI flag list against the argparse block each runner uses.
4. Respect the conventions in *src/docs/src/pages/styleguide.md*: no backtick inline code, italics for identifiers/flags, math in *$…$*.
5. Update the **Last updated** line above with today's date and the short SHA of *HEAD* at the time of the refresh (*git rev-parse --short HEAD*).
6. Commit on the current branch with a message like *docs: refresh the-notebooks reference against &lt;sha&gt;* and stop — do not open a PR unless asked.
