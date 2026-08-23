---
name: exp
description: Use only when the user explicitly invokes exp scope or $exp scope to formalize one unrun Pinglab scouting plan.
---

# Exp

## Signature

| Operator | Input artifact | Output artifact |
| --- | --- | --- |
| `exp scope` | `FrozenHypothesisPacket` | `ExpScoutPlan` |

Artifact definitions: [../../ARTIFACTS.md](../../ARTIFACTS.md).

Command grammar: `exp scope`. The project-wide optional `$` alias and
exact-invocation rule apply.

Read [references/drafts.md](references/drafts.md), then convert the frozen
hypothesis into a formal, self-contained unrun experiment plan. Return the plan
in the response. Do not claim that it has been persisted,
implemented, or run.

Bare `exp` explains that `scope` is its only subcommand.
