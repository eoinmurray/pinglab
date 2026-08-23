---
name: hypo
description: Use only when the user explicitly invokes hypo or $hypo to branch, compare, ground, checkpoint, or freeze Pinglab's scientific search.
---

# Hypo

## Signature

| Verb | Input noun | Output noun |
| --- | --- | --- |
| `hypo beam[X]` | `Seed`, `Formulation`, or `ResumableCheckpoint` | `BranchSet` |
| `hypo compare` | `Formulation` | `CanonComparisonCapsules` |
| `hypo ground web` | `Formulation` | `LiteratureEvidenceCapsules` |
| `hypo ground local` | `Formulation` | `RepositoryEvidenceCapsules` |
| `hypo checkpoint` | `OpenSearchTrajectory` | `ResumableCheckpoint` |
| `hypo freeze` | `GroundedSearchTrajectory` or `ResumableCheckpoint` | `FrozenHypothesisPacket` |

Noun definitions: [../../NOUNS.md](../../NOUNS.md).

Command grammar: `hypo SUBCOMMAND`. The project-wide optional `$` alias and
exact-invocation rule apply.

Use exactly one subcommand:

- `hypo beam` or `hypo beamX` — read [references/beam.md](references/beam.md).
- `hypo compare` — read [references/compare.md](references/compare.md).
- `hypo ground web` — read [references/ground-web.md](references/ground-web.md).
- `hypo ground local` — read
  [references/ground-local.md](references/ground-local.md).
- `hypo checkpoint` — read
  [references/checkpoint.md](references/checkpoint.md).
- `hypo freeze` — read [references/freeze.md](references/freeze.md).

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
