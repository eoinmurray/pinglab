---
name: campaign
description: Use only when the user explicitly invokes $campaign to plan, inspect, execute, resume, review, or promote a Pinglab campaign.
---

# Campaign

Activate this skill only for an explicit `$campaign` command. Do not activate it
from semantic similarity, automatic selection, or an ordinary request about a
campaign.

Read [references/lifecycle.md](references/lifecycle.md), then use exactly one
subcommand:

- `$campaign plan` — formulate a campaign and its acceptance gates; read-only.
- `$campaign status ID` — inspect declared stages, jobs, outputs, provenance,
  archive state, and blockers; read-only.
- `$campaign run ID` — execute the already-approved campaign on its declared
  target. Pod-creating, Modal, or other paid targets require explicit permission
  naming that target in the same request.
- `$campaign resume ID` — run only incomplete approved stages after verifying
  existing outputs and the same compute authorization.
- `$campaign review ID` — assess completeness, provenance, scientific validity,
  and readiness for archive or promotion; read-only.
- `$campaign promote ID` — promote a reviewed, complete, durably archived
  campaign into its dedicated publication worktree. Do not commit or publish.

Bare `$campaign` explains the subcommands. Never silently cross from planning
to execution, from execution to promotion, or from promotion to publication.
