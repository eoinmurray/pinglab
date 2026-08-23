---
name: hypo
description: Use only when the user explicitly invokes hypo or $hypo to branch, compare, ground, checkpoint, or freeze Pinglab's scientific search.
---

# Hypo

## Signature

| Operator | Input artifact | Output artifact |
| --- | --- | --- |
| `hypo beam[X]` | `Seed`, `Formulation`, or `ResumableCheckpoint` | `BranchSet` |
| `hypo compare` | `Formulation` | `CanonComparisonCapsules` |
| `hypo ground web` | `Formulation` | `LiteratureEvidenceCapsules` |
| `hypo ground local` | `Formulation` | `RepositoryEvidenceCapsules` |
| `hypo checkpoint` | `OpenSearchTrajectory` | `ResumableCheckpoint` |
| `hypo freeze` | `GroundedSearchTrajectory` or `ResumableCheckpoint` | `FrozenHypothesisPacket` |

Artifact definitions: [../../ARTIFACTS.md](../../ARTIFACTS.md).

Activate this skill only for an exact documented `hypo` command, with or
without a leading `$`. Do not activate it from semantic similarity or ordinary
discussion of hypotheses.

Use exactly one subcommand:

- `hypo beam` or `hypo beamX` — read [references/beam.md](references/beam.md).
- `hypo compare` — read [references/compare.md](references/compare.md).
- `hypo ground web` — read [references/ground-web.md](references/ground-web.md).
- `hypo ground local` — read
  [references/ground-local.md](references/ground-local.md).
- `hypo checkpoint` — read
  [references/checkpoint.md](references/checkpoint.md).
- `hypo freeze` — read [references/freeze.md](references/freeze.md).

Treat `$hypo ...` as an exact alias. Bare `hypo` lists the subcommands without
choosing one.

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
