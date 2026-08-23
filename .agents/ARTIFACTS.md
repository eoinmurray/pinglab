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

## `ExpScoutPlan`

A prospective, budgeted reconnaissance contract containing identity, abstract,
scientific frame, locally aligned investigation units, decision gates, controls
and validity, and a detailed protocol. It defines cheap tests for feasibility,
pattern discovery, and deciding whether deeper study is warranted.

## `ExpScout`

An executed scouting mission containing its frozen `ExpScoutPlan`,
implementation and provenance, actual configuration and deviations,
provisional observations, uncertainty, and a stop, revise, or escalate decision.
Its evidence is explicitly exploratory rather than durable.

## `ExpStudyPlan`

A new prospective contract informed by one or more `ExpScout` artifacts. It
uses the shared experiment-plan structure while strengthening estimands,
sampling, controls, uncertainty treatment, falsifiers, and robustness
requirements for a durable scientific test.

## `ExpStudy`

A durable executed scientific record containing its frozen `ExpStudyPlan`,
exact implementation and provenance, observations, uncertainty, deviations,
rival discrimination, conclusions, limitations, and completion status. An
`ExpScout` cannot be relabelled as an `ExpStudy`; a new `ExpStudyPlan` must
separate exploratory choices from the study's prospective commitments.

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
