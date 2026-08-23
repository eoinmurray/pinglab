# Pinglab artifact registry

Pinglab commands are scientific operators over primitive artifacts. The
artifacts are prose-defined and serialized as Markdown rather than formally
schema-validated. Repository evidence and publication outputs retain their
native files alongside the textual artifact describing them.

## `ScientificRecord`

The evidenced project history: aims, writings, compact artifacts, recorded
runs, demonstrated findings, negative results, and current direction.

## `ScientificAbstract`

A connected two-, four-, or six-paragraph narrative covering the central
question, approach, supported findings, and present direction.

## `Seed`

A short scientific intuition, question, anomaly, or proposed mechanism.

## `Formulation`

A current candidate framing with its mechanism, scope, claims, and uncertainty.

## `BranchSet`

A numbered collection of distinct candidate continuations with stable IDs,
potential value, liabilities, and distinguishing observations.

## `CanonComparisonCapsules`

Evidence capsules locating a formulation relative to remembered scientific
canon, with every reference-dependent claim marked unverified.

## `LiteratureEvidenceCapsules`

Evidence capsules containing verified current literature, provenance,
conflicting evidence, limitations, and consequences for the formulation.

## `RepositoryEvidenceCapsules`

Evidence capsules derived from existing Pinglab code, writings, artifacts, and
recorded runs without executing new scientific work.

## `OpenSearchTrajectory`

The live sequence of seeds, branches, reviews, decisions, evidence, rejected
paths, and unresolved uncertainty.

## `ResumableCheckpoint`

A standalone snapshot preserving enough of an open search trajectory for a new
agent or later conversation to resume it faithfully.

## `GroundedSearchTrajectory`

An open search trajectory whose load-bearing formulation has been compared or
grounded sufficiently for a commitment decision.

## `FrozenHypothesisPacket`

A context-free execution contract containing objective, mechanism, rivals,
evidence, predictions, experiment, estimand, controls, falsifiers, limits,
provenance, and completion criteria.

## `PinglabLexiconContext`

The live Pinglab command vocabulary and its relationship to the global Lexicon.

## `PinglabLexiconReference`

A concise Markdown reference mapping Pinglab operators to their input and output
artifact types.

## `ExperimentPlan`

A self-contained, unrun experiment contract containing identity, abstract,
scientific frame, locally aligned investigation units, cross-result synthesis,
controls and validity, and a detailed protocol. Each investigation unit pairs a
method summary with its planned output, expected patterns, decision rule, and
local caveat; lengthy shared implementation detail lives once in the protocol.
Nothing in the plan is represented as an observed result.

## `ExperimentRecord`

An executed experiment record that extends and preserves its frozen
`ExperimentPlan`, then adds run provenance, actual configuration, deviations,
observations, uncertainty, interpretation, and completion status. Planned
expectations remain distinguishable from observed results.

## `ScientificCollectionState`

The current collection registration, writing metadata, referenced artifacts,
provenance, generated outputs, and publication blockers.

## `PublicationReadinessReport`

A Markdown assessment of collection completeness, missing evidence, provenance,
artifact drift, and required upstream work.

## `PublicationReadyCollection`

A scientific collection whose required writings, artifacts, provenance, and
configuration satisfy the supported build contract.

## `PublicationBundle`

The complete rendered local PDFs and ignored site outputs, accompanied by a
concise build-verification report.
