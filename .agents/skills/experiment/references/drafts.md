# Experiment formats

Use one shared experiment contract across the four artifact types. Do not
duplicate these fields between plans and their executed records:

- identity and scope;
- question and hypothesis;
- methods, variables, and controls;
- measurements and decision rules.

The lifecycle is:

`ExpScoutPlan` -> `ExpScout` -> `ExpStudyPlan` -> `ExpStudy`.

`Plan` always means prospective. A paired non-`Plan` artifact always means an
executed record.

## `ExpScoutPlan`

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

An `ExpVisualSet` is optional. When one is useful, the plan specifies each
visual pair's design schematic, measured mirror, shared panel structure,
variables, colours, visual grammar, and interpretive purpose. It also reserves
links to the investigation unit, protocol section, generating code, and source
data. The schematic and empty result slot form an implementation scaffold; the
measured mirror does not exist until it is generated from executed evidence.

Write expected patterns conditionally. The plan contains no observed results,
and neither existing context nor a schematic may be presented as execution
evidence.

Keep scout execution cheap and explicitly budgeted. Its investigation units
should establish feasibility, search broadly for useful structure, reject dead
branches, and define gates for stopping, revision, or escalation. A scout plan
does not promise durable inference.

## `ExpScout`

Freeze and preserve the `ExpScoutPlan`, then add:

- run provenance and completion status;
- actual configuration and deviations from plan;
- provisional observations and uncertainty;
- interpretation against the plan's decision gates;
- a stop, revise, or escalate decision.

If the plan requested an `ExpVisualSet`, attach its completed measured mirrors
with code and data provenance. Preserve incomplete and failed result slots
rather than silently omitting them. A scout without an `ExpVisualSet` remains
valid.

Label its evidence exploratory. It may motivate a new study plan but may not be
promoted or relabelled as durable evidence.

The optional composition is `ExpScout = frozen ExpScoutPlan + completed
ExpVisualSet + other execution evidence + disposition`.

## `ExpStudyPlan`

Create a new prospective contract informed by the scout rather than expanding
the scout retrospectively. Use the same canonical plan structure, with stronger
requirements for estimands, sampling and seeds, controls, uncertainty,
falsifiers, rival discrimination, stopping rules, and robustness. Freeze which
scout-informed choices are carried forward before study execution begins.

## `ExpStudy`

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
