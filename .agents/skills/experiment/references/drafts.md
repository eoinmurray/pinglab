# Experiment formats

Use one shared experiment contract for both artifact types. Do not duplicate
these fields between the plan and record:

- identity and scope;
- question and hypothesis;
- methods, variables, and controls;
- measurements and decision rules.

## `ExperimentPlan`

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

## `ExperimentRecord extends ExperimentPlan`

Freeze and preserve the original plan, then add execution evidence:

- run provenance and completion status;
- actual configuration and deviations from plan;
- observations and uncertainty;
- interpretation against the plan's decision rules and falsifiers.

For each completed result, show its title, figure or output, and concise
interpretation. Never replace or rewrite planned expectations as observations.
The relationship is `ExperimentRecord = frozen ExperimentPlan + execution
evidence`.
