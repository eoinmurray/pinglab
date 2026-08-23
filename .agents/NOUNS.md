# Pinglab noun registry

Pinglab commands are scientific verbs over primitive nouns. The nouns are
prose-defined and serialized as Markdown rather than formally schema-validated.
Repository evidence and publication outputs retain their native files alongside
the textual noun describing them.

## `ScientificRecord`

The evidenced project history: aims, writings, compact artifacts, recorded
runs, demonstrated findings, negative results, and current direction.

## `PinglabAbstract`

A connected two-, four-, or six-paragraph narrative covering the central
question, approach, supported findings, and present direction.

The short, medium, and long forms contain exactly two, four, and six non-empty
prose paragraphs respectively, separated by blank lines. Medium is the default
when no length is specified. The body has no title, headings, bullets, numbered
lists, source inventory, process narration, or follow-up question. Required
higher-level response footers sit outside the body and do not count toward its
paragraph total.

Construct the narrative from the current `ScientificRecord`, following the
scientific-record invariants in `AGENTS.md`. Summarize what Pinglab has been
trying to understand and build rather than merely listing files. Inspect only
enough current repository evidence to support the account, prioritizing:

1. the project purpose and scientific-record rules in `AGENTS.md` and
   `README.md`;
2. collection aims in `demolab.yaml`;
3. the manuscript, collection introductions, and relevant current writing
   metadata under `writings/`;
4. compact published artifacts or run evidence behind any reported result;
5. recent Git history when needed to identify the active direction.

Do not use conversation history as evidence for the project's scientific aims
or imply that every collection or recent commit has equal scientific
importance. Each paragraph must advance the account. Define unfamiliar
abbreviations on first use. Avoid exact numbers unless they materially improve
the summary and satisfy the evidence rule.

## `Seed`

A short scientific intuition, question, anomaly, or proposed mechanism.

## `Formulation`

A current candidate framing with its mechanism, scope, claims, and uncertainty.

## `HypoBranches`

A numbered collection of distinct candidate continuations with stable IDs,
potential value, liabilities, and distinguishing observations.

## `HypoCanon`

Evidence capsules locating a formulation relative to remembered scientific
canon, with every reference-dependent claim marked unverified.

## `HypoLiterature`

Evidence capsules containing verified current literature, provenance,
conflicting evidence, limitations, and consequences for the formulation.

## `HypoRepository`

Evidence capsules derived from existing Pinglab code, writings, artifacts, and
recorded runs without executing new scientific work.

## `OpenSearchTrajectory`

The live sequence of seeds, branches, reviews, decisions, evidence, rejected
paths, and unresolved uncertainty.

## `HypoCheckpoint`

A standalone snapshot preserving enough of an open search trajectory for a new
agent or later conversation to resume it faithfully.

## `GroundedSearchTrajectory`

An open search trajectory whose load-bearing formulation has been compared or
grounded sufficiently for a commitment decision.

## `HypoPacket`

A context-free execution contract containing objective, mechanism, rivals,
evidence, predictions, experiment, estimand, controls, falsifiers, limits,
provenance, and completion criteria.

## `PinglabLexiconContext`

The live Pinglab command vocabulary and its relationship to the global Lexicon.

## `PinglabLexicon`

A concise Markdown reference mapping Pinglab verbs to their input and output
noun types.

## Experiment family

Use one shared experiment contract across the four experiment noun types. Do
not duplicate these fields between plans and their executed records:

- identity and scope;
- question and hypothesis;
- methods, variables, and controls;
- measurements and decision rules.

The lifecycle is:

`ExpScoutPlan` -> `ExpScout` -> `ExpStudyPlan` -> `ExpStudy`.

`Plan` always means prospective. A paired non-`Plan` noun always means an
executed record.

## `ExpScoutPlan`

A prospective, budgeted reconnaissance contract containing identity, abstract,
scientific frame, locally aligned investigation units, decision gates, controls
and validity, and a detailed protocol. It defines cheap tests for feasibility,
pattern discovery, and deciding whether deeper study is warranted.

Use `status: "draft"` while a design is being refined and has not been run.

Use this canonical reading order:

1. **Identity.** Give the title, status, collection, dependencies, and available
   provenance. State near the start what, if anything, has already run.
2. **Abstract.** State the question, hypothesis, intervention, primary estimand,
   and what outcome would matter.
3. **Scientific frame.** Explain the proposed mechanism, established context,
   competing explanations, assumptions, and scope.
4. **Investigation units.** For each numbered unit, keep the evidence logic
   together in this order:
   - **Question:** the uncertainty this unit resolves.
   - **Method summary:** only enough method to understand the planned result,
     with references to the detailed protocol where needed.
   - **Planned output:** the figure, table, or statistic, including axes, traces,
     marks, or displayed elements and its purpose.
   - **Expected patterns:** predictions under the hypothesis and its rivals.
   - **Decision rule:** how each material outcome updates the hypothesis.
   - **Local caveat:** what this result cannot establish.
5. **Cross-result synthesis.** Explain how the units jointly distinguish the
   mechanism from rivals, including informative contradictions.
6. **Controls and validity.** Define positive and negative controls, confounds,
   uncertainty treatment, falsifiers, stopping rules, and completion criteria.
7. **Detailed protocol.** Put the complete shared configuration, datasets,
   parameter values, sampling, seeds, execution sequence, analysis definitions,
   provenance, and reproducibility requirements here. Number its sections so
   investigation units can reference them without duplication.

Keep a network or experimental-design diagram beside the investigation unit it
explains. Clearly label conceptual curves as design schematics rather than data.
When a mechanism benefits from visual mirroring, pair a precise hand-authored
design schematic with a planned reproducible result using the same panel layout
and visual language.

Write expected patterns conditionally. The plan contains no observed results,
and neither existing context nor a schematic may be presented as execution
evidence.

Keep scout execution cheap and explicitly budgeted. Its investigation units
should establish feasibility, search broadly for useful structure, reject dead
branches, and define gates for stopping, revision, or escalation. A scout plan
does not promise durable inference.

## `ExpVisualSet`

An optional visual evidence scaffold pairing design schematics with
structurally matched measured plots. Each pair preserves a shared panel layout,
variables, colours, and visual grammar, and links to its investigation unit,
protocol section, generating code, source data, and interpretive caption. It is
specified by an `ExpScoutPlan`, begins during implementation, and becomes
evidence only when its measured mirrors are completed from executed results.
Its default epistemic colour grammar uses blue-grey for conceptual or
prospective schematics and red-black for measured evidence; documented
overrides must preserve that distinction.

When an `ExpVisualSet` is useful, the plan specifies each visual pair's design
schematic, measured mirror, shared panel structure, variables, colours, visual
grammar, and interpretive purpose. It also reserves links to the investigation
unit, protocol section, generating code, and source data. The schematic and
empty result slot form an implementation scaffold; the measured mirror does not
exist until it is generated from executed evidence.

The colour grammar is epistemic rather than a fixed palette. A scientific
semantic, accessibility need, or established figure convention may override
the default, but the plan must document the new mapping and keep schematic and
measured roles visibly distinct. Never style an unexecuted or conceptual curve
as red-black measured evidence.

## `ExpScout`

An executed scouting mission containing its frozen `ExpScoutPlan`,
implementation and provenance, actual configuration and deviations,
provisional observations, uncertainty, and a stop, revise, or escalate decision.
It may include a completed `ExpVisualSet`; its evidence is explicitly
exploratory rather than durable.

Freeze and preserve the `ExpScoutPlan`, then add:

- run provenance and completion status;
- actual configuration and deviations from plan;
- provisional observations and uncertainty;
- interpretation against the plan's decision gates;
- a stop, revise, or escalate decision.

Use this canonical publication anatomy:

1. **Abstract.** State the question, scout design, highest-level observation,
   and disposition without procedural detail.
2. **Shared.** Record the common scientific frame, activity or inputs,
   controls, decision gate, budget, provenance, deviations, and limitations
   once rather than repeating them inside investigations.
3. **Investigations.** List investigations progressively. Each investigation
   has a numbered descriptive header, a relevance paragraph, the expected
   pattern or discriminating outcome, its `ExpVisualSet`, and a high-level
   discussion of what was observed. Keep numerical and procedural ramble out
   of the discussion. Use one investigation per plotted rule or aggregate
   diagnostic: an investigation contains exactly one `ExpVisualSet` plot, and
   distinct rules must not be bundled merely because they belong to the same
   conceptual family. Shared setup schematics may remain in **Shared** because
   they are not evidence plots.
4. **Methods.** List methods progressively. Each method has a numbered
   descriptive header, the mathematical or algorithmic steps involved, its
   concrete output, and an explicit link to the investigation that consumes
   that output.
5. **Conclusion** when the scout benefits from a consolidated interpretation
   or disposition.
6. **References** when external sources are used.
7. **Appendices** when supporting detail would interrupt the main sequence.

Number every body heading hierarchically. The publication title, figure
captions, equations, and inline structural labels such as `Relevance` and
`ExpVisualSet` are not body headings and remain unnumbered. Optional trailing
sections are omitted rather than emitted empty; retain the canonical section
number when one is present.

If the plan requested an `ExpVisualSet`, attach its completed measured mirrors
with code and data provenance. Preserve incomplete and failed result slots
rather than silently omitting them. A scout without an `ExpVisualSet` remains
valid.

Label its evidence exploratory. It may motivate a new study plan but may not be
promoted or relabelled as durable evidence.

The optional composition is `ExpScout = frozen ExpScoutPlan + completed
ExpVisualSet + other execution evidence + disposition`.

## `ExpStudyPlan`

A new prospective contract informed by one or more `ExpScout` artifacts. It
uses the shared experiment-plan structure while strengthening estimands,
sampling, controls, uncertainty treatment, falsifiers, and robustness
requirements for a durable scientific test.

Create a new prospective contract informed by the scout rather than expanding
the scout retrospectively. Use the same canonical plan structure, with stronger
requirements for estimands, sampling and seeds, controls, uncertainty,
falsifiers, rival discrimination, stopping rules, and robustness. Freeze which
scout-informed choices are carried forward before study execution begins.

## `ExpStudy`

A durable executed scientific record containing its frozen `ExpStudyPlan`,
exact implementation and provenance, observations, uncertainty, deviations,
rival discrimination, conclusions, limitations, and completion status. An
`ExpScout` cannot be relabelled as an `ExpStudy`; a new `ExpStudyPlan` must
separate exploratory choices from the study's prospective commitments.

Freeze and preserve the `ExpStudyPlan`, then add:

- exact implementation, run provenance, and completion status;
- actual configuration and deviations from plan;
- observations and uncertainty;
- interpretation against decision rules, falsifiers, and rivals;
- durable conclusions and limitations.

For each completed result, show its title, figure or output, and concise
interpretation. Never replace or rewrite planned expectations as observations.
The invariant is `ExpScout` -> new `ExpStudyPlan` -> `ExpStudy`, never
`ExpScout` -> relabelled `ExpStudy`.

## `ScientificCollectionState`

The current collection registration, writing metadata, referenced artifacts,
provenance, generated outputs, and publication blockers.
