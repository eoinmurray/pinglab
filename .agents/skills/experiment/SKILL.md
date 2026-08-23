---
name: experiment
description: Use only when the user explicitly invokes experiment draft or $experiment draft to formalize one unrun Pinglab experiment.
---

# Experiment

## Signature

| Operator | Input artifact | Output artifact |
| --- | --- | --- |
| `experiment draft` | `FrozenHypothesisPacket` | `UnrunExperimentSpecification` |

Artifact definitions: [../../ARTIFACTS.md](../../ARTIFACTS.md).

Activate this skill only for the exact `experiment draft` command, with or
without a leading `$`. Do not activate
it from semantic similarity, automatic selection, or an ordinary request about
an experiment.

Read [references/drafts.md](references/drafts.md), then convert the frozen
hypothesis into a formal, self-contained unrun experiment specification. Return
the specification in the response. Do not claim that it has been persisted,
implemented, or run.

Bare `experiment` explains that `draft` is its only subcommand.
