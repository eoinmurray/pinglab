---
name: hypo
description: Use only when the user explicitly invokes hypo or $hypo to request branches, canon, literature, repository evidence, a checkpoint, or a packet for Pinglab's scientific search.
---

# Hypo

## Signature

| Verb | Input noun | Output noun |
| --- | --- | --- |
| `hypo branches[X]` | `Seed`, `Formulation`, or `HypoCheckpoint` | `HypoBranches` |
| `hypo canon` | `Formulation` | `HypoCanon` |
| `hypo literature` | `Formulation` | `HypoLiterature` |
| `hypo repository` | `Formulation` | `HypoRepository` |
| `hypo checkpoint` | `OpenSearchTrajectory` | `HypoCheckpoint` |
| `hypo packet` | `GroundedSearchTrajectory` or `HypoCheckpoint` | `HypoPacket` |

Noun definitions: [../../NOUNS.md](../../NOUNS.md).

Command grammar: `hypo SUBCOMMAND`. The project-wide optional `$` alias and
exact-invocation rule apply.

Use exactly one subcommand:

- `hypo branches` or `hypo branchesX` — read
  [references/branches.md](references/branches.md).
- `hypo canon` — read [references/canon.md](references/canon.md).
- `hypo literature` — read [references/literature.md](references/literature.md).
- `hypo repository` — read
  [references/repository.md](references/repository.md).
- `hypo checkpoint` — read
  [references/checkpoint.md](references/checkpoint.md).
- `hypo packet` — read [references/packet.md](references/packet.md).

Bare `hypo` lists the subcommands without choosing one.

## Shared search contract

Maintain the current question and search coordinates, leading formulation and
serious rivals, user decisions versus model proposals, observed evidence versus
inference, branch status and rejection reasons, and material uncertainty.

Use stable branch identifiers such as `B1`, `B2`, and `B3`. Represent grounding
results as evidence capsules:

```text
Claim:
Evidence:
Provenance:
Verification: canon-comparison / verified-literature / observed-repository
Limitations:
Consequence:
```

When user judgment is required, end with:

```text
Keep:
Reject:
Add:
Uncertain:
Suggested next operation:
```

Ordinary conversation supplies review, selection, and single-branch refinement.
Do not create a synthetic consensus merely to advance the workflow.
