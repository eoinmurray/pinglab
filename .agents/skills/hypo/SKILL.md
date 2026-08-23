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

- `hypo branches` uses three branches. In `hypo branchesX`, require a practical
  positive integer and reject zero, negative, ambiguous, or unreasonably large
  values. Beam only at the current consequential uncertainty. After selection,
  return to ordinary single-branch work unless another branch set is requested.
- `hypo canon` compares the formulation with internal understanding of the
  established scientific canon. Include two or three remembered academic
  references when useful, but do not browse or fabricate bibliographic
  precision.
- `hypo literature` identifies claims that could materially change the
  scientific direction and verifies them using current web literature. Build
  searches independently from remembered references and actively seek
  conflicting or limiting evidence. Prefer primary research, authoritative
  datasets, and first-party technical documentation; use reviews for
  orientation or field-level synthesis.
- `hypo repository` inspects relevant code, model definitions, writings,
  compact artifacts, recorded runs, provenance, and configuration. Do not run
  or rebuild scientific work.
- `hypo checkpoint` serializes the current `OpenSearchTrajectory` without
  replaying the conversation.
- `hypo packet` freezes only the best currently grounded formulation without
  reopening broad ideation. If an essential claim remains ungrounded, identify
  the gap and decline to label the packet definitive.

Bare `hypo` lists the subcommands without choosing one.

Ordinary conversation supplies review, selection, and single-branch refinement.
