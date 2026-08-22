---
name: experiment
description: Use only when the user explicitly invokes $experiment to design, draft, review, or locally run one Pinglab experiment.
---

# Experiment

Activate this skill only for an explicit `$experiment` command. Do not activate
it from semantic similarity, automatic selection, or an ordinary request about
an experiment.

Use exactly one subcommand:

- `$experiment design` — inspect relevant prior work and develop a falsifiable,
  proportionate experiment design conversationally. Read-only.
- `$experiment draft` — create or revise only an unrun experiment writing and
  hand-authored design assets. Read [references/drafts.md](references/drafts.md).
- `$experiment review ID` — review the named design, runner, artifacts,
  provenance, and interpretation. Read-only; report evidence separately from
  inference.
- `$experiment run ID` — run the existing named runner locally. Resolve the
  exact runner and output targets first; do not change source, use remote
  compute, promote artifacts, or mutate Git.

Bare `$experiment` explains the subcommands. An experiment command never
authorizes edits to `tools/snn`, Git operations, paid compute, or publication.
Stop when the named experiment or its declared output boundary is ambiguous.
