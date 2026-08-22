---
name: publish
description: Use only when the user explicitly invokes $publish to check or build Pinglab's complete local publication.
---

# Publish

Activate this skill only for an explicit `$publish` command. Do not activate it
from semantic similarity, automatic selection, or an ordinary publication
request.

Use exactly one subcommand:

- `$publish check` — inspect collection registration, writing metadata,
  referenced artifacts, provenance, generated-output drift, and publication
  readiness. Read-only.
- `$publish build` — from the dedicated publication branch and worktree, run the
  complete supported Demolab build and inspect its outputs. Do not stage,
  commit, push, open or update a PR, merge, or deploy.

Bare `$publish` explains the subcommands. Do not run experiments or repair
missing evidence during a publication operation. Report the missing upstream
work instead.
