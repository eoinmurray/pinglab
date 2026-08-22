---
name: lab
description: Use only when the user explicitly invokes $lab to explain the Pinglab lexicon or report project state.
---

# Lab

Activate this skill only for an explicit `$lab` command. Do not activate it from
semantic similarity, automatic selection, or an ordinary request about project
state.

Use exactly one read-only subcommand:

- `$lab help` — explain the Pinglab lexicon from the repository `AGENTS.md`,
  including authorization boundaries and one or two grounded examples.
- `$lab status` — inspect Git state, current experiments, local campaigns,
  publication inputs, and obvious blockers; distinguish observed state from
  inference and stale remote state.

Bare `$lab` lists the two subcommands. Do not edit files, run experiments,
submit jobs, build publications, or mutate Git or external systems.
