---
name: pinglab-lexicon
description: Interpret, construct, transform, review, or serialize Pinglab noun types whenever a prompt invokes ScientificRecord, Abstract, Seed, Formulation, HypoBranches, HypoCanon, HypoLiterature, HypoRepository, OpenSearchTrajectory, HypoCheckpoint, GroundedSearchTrajectory, HypoPacket, ExpScoutPlan, ExpSharedPlan, ExpInvestigationPlan, ExpInvestigationIdentity, ExpInvestigationIntroduction, ExpExpectedPatterns, ExpVisualSet, ExpDesignSchematic, ExpMeasuredResultSlot, ExpMethodsPlan, ExpConclusionSlot, ExpReferences, ExpAppendices, ExpScout, ScoutExecution, ExpScoutSummary, ExpSharedExecution, ExpInvestigationExecution, ExpMeasuredResult, ExpObservedPatterns, ExpMethodsExecuted, ExpConclusion, ExpStudyPlan, StudyExecution, ExpStudy, ExpImplementation, CollectionDataset, ExperimentRun, CampaignPlan, CampaignExecution, RunRecord, PublicationView, or ScientificCollectionState in any case, spacing, joining, or optional dollar-prefixed form, or asks to scope encode a rule into the Pinglab writing system.
---

# Pinglab lexicon

Pinglab's scientific vocabulary is organized as global rules, verbs, and nouns.
Use this file to recognize invocations and route to the authoritative reference.

## Recognition

Recognize every Pinglab noun regardless of letter case. Treat joined and
whitespace-separated noun words as equivalent where the canonical form is
unambiguous: for example, `ScientificRecord` and `scientific record`, or
`ExpScoutPlan` and `exp scout plan`. Allow an optional `$` before the complete
noun or before any noun word, so `$ExpScoutPlan`, `$exp scout plan`, and
`$exp $scout $plan` are aliases of `ExpScoutPlan`.

Apply this normalization only when identifying a Pinglab noun, not to an
ordinary phrase that happens to use the same words. Use the canonical names in
[NOUNS.md](references/NOUNS.md) when constructing or reporting artifacts.

Recognize `scope encode` regardless of case and allow `$scope`, `$encode`, or
both. Preserve its rule argument without normalization.

## Routing

| Request contains | Read completely |
|---|---|
| Any Pinglab invocation | [GLOBAL.md](references/GLOBAL.md) |
| A registered noun | [NOUNS.md](references/NOUNS.md) |
| `ScopeEncode` or `scope encode` | [VERBS.md](references/VERBS.md), then the reference owning the affected rule |
| A noun plus `ScopeEncode` | All three references |

Read each selected reference completely before acting. `GLOBAL.md` always
applies. When rules appear to conflict, preserve higher-priority instructions,
then prefer the narrower noun or verb contract over general guidance.

## Reference map

- [GLOBAL.md](references/GLOBAL.md) — canvas ownership, lifecycle, authority,
  progressive interaction, and human-facing writing.
- [VERBS.md](references/VERBS.md) — Pinglab-specific verb behaviour and
  self-update rules. The separate general Lexicon remains authoritative for
  generic `Scope` and `go` Git behaviour.
- [NOUNS.md](references/NOUNS.md) — canonical nouns, families, composition,
  execution, datasets, campaigns, publication, and collection state.
