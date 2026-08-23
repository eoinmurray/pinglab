---
name: exp
description: Use only when the user explicitly invokes exp scout-plan or $exp scout-plan to formalize one unrun Pinglab scouting plan.
---

# Exp

## Signature

| Verb | Input noun | Output noun |
| --- | --- | --- |
| `exp scout-plan` | `HypoPacket` | `ExpScoutPlan` |

Noun definitions: [../../NOUNS.md](../../NOUNS.md).

Command grammar: `exp scout-plan`. The project-wide optional `$` alias and
exact-invocation rule apply.

Read [references/drafts.md](references/drafts.md), then convert the frozen
hypothesis into a formal, self-contained unrun experiment plan. Return the plan
in the response. Do not claim that it has been persisted,
implemented, or run.

Bare `exp` explains that `scout-plan` is its only subcommand.
