---
name: writing
description: Use only when the user explicitly invokes $writing to draft, review, or build Pinglab scientific writings.
---

# Writing

Activate this skill only for an explicit `$writing` command. Do not activate it
from semantic similarity, automatic selection, or an ordinary writing request.

Use exactly one subcommand:

- `$writing draft` — create or revise `writings/*.typ` and its hand-authored
  writing assets only. Follow the scientific-record invariants in `AGENTS.md`;
  for an experiment design also read
  [the experiment draft guidance](../experiment/references/drafts.md).
- `$writing review ID` — review one writing, its cited evidence, figures,
  captions, accessibility, and interpretation; read-only.
- `$writing build ID` — build one existing entry through Demolab and update only
  its declared generated publication outputs. Do not edit prose, run software
  tests, build the full publication, or mutate Git.

Bare `$writing` explains the subcommands. Write directly for a scientific
reader. Define technical terms when introduced and define every symbol used in
an equation. Captions must identify axes, units, marks, conditions, and the
supported takeaway. Computed values must come from the run, including values in
captions.

Do not open, embed, or display generated PDFs in the Codex app. On handoff,
provide clickable links to both the generated PDF and HTML entry.
