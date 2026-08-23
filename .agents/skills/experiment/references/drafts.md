# Experiment formats

Use one shared experiment contract for both artifact types. Do not duplicate
these fields between the plan and record:

- identity and scope;
- question and hypothesis;
- methods, variables, and controls;
- measurements and decision rules.

## `ExperimentPlan`

Use `status: "draft"` while a design is being refined and has not been run.

1. Write a short Abstract stating the question and intended measurement.
2. Write enumerated Methods and state near the start what, if anything, has
   already run.
3. Write enumerated planned Results in the same order as Methods. For each
   result define axes, traces or marks, purpose, and expected observation. For
   non-plots, state that axes and traces do not apply and define the displayed
   elements, purpose, and expected output.

Keep a network or experimental-design diagram beside the method it explains.
Clearly label conceptual curves as design schematics rather than data. When a
mechanism benefits from visual mirroring, use the exp086 pattern: a precise
hand-authored SVG in Methods and a reproducibly generated Results figure with
the same panel layout and visual language.

The plan additionally defines expected patterns, falsifiers, planned outputs,
and execution requirements. It contains no observed results.

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
