# Nouns

## Visual index

- [Scientific record](#scientificrecord)
- [Abstract and writing objects](#abstract)
- [Hypothesis search](#hypothesis-search-family)
- [Execution and datasets](#execution-artifact-family)
- [Experiment composition](#experiment-family)
- [Experiment plans](#expscoutplan)
- [Experiment execution](#expscoutsummary)
- [Campaigns and publication](#campaignplan)
- [Collection state](#scientificcollectionstate)

## `ScientificRecord`

The evidenced project history: aims, writings, compact artifacts, recorded
runs, demonstrated findings, negative results, and current direction.

## `Abstract`

A concise, connected account of a scientific work's question, approach,
supported content, and consequence.

Write only from the work's current epistemic state. Present intended work as
intended, completed observations as observed, and interpretations as
interpretations. Do not invent findings. State limitations or unresolved issues
when they materially affect the meaning.

Explain the principal pattern qualitatively rather than giving a numerical
summary; keep measurements in the evidence-bearing body. Omit headings,
bullets, field labels, procedural detail, repository plumbing, and ornamental
formality.

End with one plain-language sentence in its own paragraph. It must explain the
main meaning to a non-specialist without introducing claims, deleting important
uncertainty, or relying on specialist terminology.

Other nouns may contain or link to an `Abstract`; they do not redefine it.

## Hypothesis-search family

Hypothesis-search nouns preserve the current question and search coordinates,
the leading formulation and serious rivals, user decisions versus model
proposals, observed evidence versus inference, branch status and rejection
reasons, and material uncertainty. They do not create synthetic consensus merely
to advance the workflow.

Ordinary conversation supplies review, selection, and single-branch refinement.

Grounding results use evidence capsules with this structure:

```text
Claim:
Evidence:
Provenance:
Verification: canon-comparison / verified-literature / observed-repository
Limitations:
Consequence:
```

When user judgment is required, the artifact ends with this review hook:

```text
Keep:
Reject:
Add:
Uncertain:
Suggested next operation:
```

## `Seed`

A short scientific intuition, question, anomaly, or proposed mechanism.

## `Formulation`

A current candidate framing with its mechanism, scope, claims, and uncertainty.

## `HypoBranches`

A numbered collection of distinct candidate continuations with stable IDs,
potential value, liabilities, and distinguishing observations.

It contains exactly the requested number of genuinely distinct continuations.
Each has a stable identifier such as `B1`, `B2`, or `B3`, its central move,
potential value, main liability, and the observation that would most efficiently
distinguish it from the others. Preserve viable minority hypotheses rather than
forcing consensus. End with a stable numbered set and the shared review hook.

When no cardinality is specified, use three branches. Any requested cardinality
must be a practical positive integer; reject zero, negative, ambiguous, or
unreasonably large values. Branch only at the current consequential uncertainty.
After selection, return to ordinary single-branch work unless another branch
set is requested.

## `HypoCanon`

Evidence capsules locating a formulation relative to remembered scientific
canon, with every reference-dependent claim marked unverified.

Separate canonical agreement, canonical tension, novel synthesis, and the
claims most worth live verification. Mark every remembered reference and
reference-dependent claim as **remembered and unverified**. This artifact
locates a formulation relative to established thinking; it does not establish
that the remembered canon is current or correct. End with the shared review
hook.

Construct the comparison from internal understanding of the established
scientific canon. Include two or three remembered academic references when
useful, but do not browse or fabricate bibliographic precision.

## `HypoLiterature`

Evidence capsules containing verified current literature, provenance,
conflicting evidence, limitations, and consequences for the formulation.

Separate verified evidence, conflicting evidence, inference beyond the sources,
and unresolved claims. Cite sources close to the claims they support. Do not
substitute citation count for relevance or convert absence of evidence into
evidence of absence. End with the shared review hook.

Identify claims whose truth could materially change the scientific direction
and verify them using current web literature. Construct searches independently
from remembered references and actively seek conflicting or limiting evidence.
Prefer primary research, authoritative datasets, and first-party technical
documentation; use reviews for orientation or field-level synthesis.

## `HypoRepository`

Evidence capsules derived from existing Pinglab code, writings, artifacts, and
recorded runs without executing new scientific work.

State what repository evidence establishes, what is merely planned or inferred,
what lies outside the model's validity, and what observation would distinguish
the leading formulation from its strongest rival. Give exact file or run
provenance, verification state, limitations, and consequence. Read computed
values only from the run or artifact that produced them. End with the shared
review hook.

Construct it by inspecting relevant code, model definitions, writings, compact
artifacts, recorded runs, provenance, and configuration. Do not run or rebuild
scientific work.

## `OpenSearchTrajectory`

The live sequence of seeds, branches, reviews, decisions, evidence, rejected
paths, and unresolved uncertainty.

## `HypoCheckpoint`

A standalone snapshot preserving enough of an open search trajectory for a new
agent or later conversation to resume it faithfully.

Preserve the current question and search coordinates, stable branch IDs and
status, decisions and reasons, evidence capsules, assumptions, contradictions,
uncertainty, the leading formulation, the strongest rival, and the next
consequential choice or grounding action. Compress repetition and discarded
wording, not scientific dissent or negative results. Distinguish user decisions
from model proposals and evidence from inference. Do not manufacture closure.
Finish with the shared review hook.

Serialize the current trajectory without requiring the preceding conversation
to be replayed.

## `GroundedSearchTrajectory`

An open search trajectory whose load-bearing formulation has been compared or
grounded sufficiently for a commitment decision.

## `HypoPacket`

A context-free execution contract containing objective, mechanism, rivals,
evidence, predictions, experiment, estimand, controls, falsifiers, limits,
provenance, and completion criteria.

It contains:

1. objective, significance, precise hypothesis, and mechanism;
2. strongest plausible rivals;
3. an evidence ledger using accumulated evidence capsules;
4. discriminating qualitative and quantitative predictions;
5. a definitive experiment, estimand, and decision rule;
6. positive, negative, procedural, and numerical controls;
7. falsifiers and inconclusive outcomes;
8. operational definitions, assumptions, scope limits, and uncertainty;
9. grounded repository entry points;
10. resource limits and completion criteria;
11. the exact recommended next command;
12. provenance distinguishing user decisions, model proposals, literature, and
    repository evidence.

The experiment must be capable of changing the conclusion and state what result
would force a return to branching. A context-free agent must be able to identify
the claim, strongest rival, next action, decision rule, and completion
condition. Preserve unresolved uncertainty rather than presenting an
insufficiently grounded packet as definitive.

Freeze only the best currently grounded formulation without reopening broad
ideation. If an essential claim remains ungrounded, identify the gap and decline
to label the packet definitive.

## Execution-artifact family

Execution-artifact nouns connect prospective experiment plans to native code,
runs, durable evidence, and the continuously rendered presentation. They name
ownership boundaries rather than requiring every native file to be rewritten as
Markdown.

## `ExpImplementation`

The SNNLANG bundle or other tool contract, experiment runner, configuration,
tests, and implementation provenance that realize a persistent experiment plan.
Reusable computation belongs to the tool; hypothesis-specific conditions,
analysis, and figures belong to the experiment. An implementation does not
replace its scientific plan or establish that the experiment was executed.

## `CollectionDataset`

The collection-scoped scientific data object. Its working form retains useful
`ExperimentRun` identities, collection-level assets and upstream datasets, and
maps each experiment to one official run with optional local preview overrides.
The newest successfully finalized run automatically becomes official. A failed,
interrupted, or incomplete run leaves the previous official pointer unchanged.
Snapshotting creates an immutable, digest-bearing collection revision; later
work continues in a successor working revision. Verification, publication and
pruning remain separate operations.

## `ExperimentRun`

One experiment-scoped local or remote execution. It records execution state,
the exact prospective writing, implementation and configuration used, source
and dirty-patch provenance, command and host, upstream runs and datasets,
payload location and inventory digest, archive identity and legacy lineage. A
finalized run is immutable. Successful finalization advances its experiment's
`CollectionDataset` official pointer. Unsuccessful execution does not.

## `CampaignPlan`

A dry, cold-readable executable snapshot of a collection. It identifies the
captured source, included experiment versions, hard dependencies and stages,
commands, resource requests, shards, expected outputs, explicit exclusions,
acceptance conditions, and blocking decisions. Constructing it does not
authorize live submission or paid compute.

## `CampaignExecution`

The mutable execution of one `CampaignPlan`: jobs, shards, logs, experiment
state, derived artifact candidates, failures, repairs, resume state, aggregation,
and completion status. Independent stages may run concurrently after their hard
dependencies complete. A repair or composed campaign retains explicit lineage
to its sources rather than inheriting their identity.

## `RunRecord`

The legacy provenance-bearing runstore record for an ad-hoc execution or campaign. It
contains execution identity and status, source and upstream provenance, payload
inventory and digest, and archive identity. Finalization freezes its complete
payload identity; archive, verification, and restore establish durable
recoverability but do not accept scientific claims or activate publication data.

## `PublicationView`

The explicitly selected, provenance-linked evidence set from which Demolab
renders the current scientific presentation. It may expose draft or candidate
evidence during local review and accepted campaign evidence after activation.
Changing the view is distinct from rendering it: `CollectionDataset` selection
changes evidence ownership, Pingstore materializes that selection, and Demolab
reacts to the selected data and writing.

## Experiment family

Experiment noun definitions form a directed composition map. A composite noun
names only its immediate children. A leaf noun names no other experiment noun.
Children never name their parent, and definitions do not encode sibling,
provenance, lifecycle, or downstream relationships.

The composition map is:

```text
ExpScout
├── ExpScoutPlan
│   ├── Abstract
│   ├── ExpSharedPlan
│   ├── ExpInvestigationPlan[]
│   ├── ExpMethodsPlan
│   ├── ExpConclusionSlot
│   ├── ExpReferences?
│   └── ExpAppendices?
└── ScoutExecution
    ├── ExpScoutSummary
    ├── ExpSharedExecution
    ├── ExpInvestigationExecution[]
    ├── ExpMethodsExecuted
    └── ExpConclusion

ExpSharedPlan
└── ExpDesignSchematic?

ExpInvestigationPlan
├── ExpInvestigationIdentity
├── ExpInvestigationIntroduction
├── ExpExpectedPatterns
└── ExpVisualSet?

ExpVisualSet
├── ExpDesignSchematic
└── ExpMeasuredResultSlot

ExpInvestigationExecution
├── ExpMeasuredResult
├── ExpVisualSet?
└── ExpObservedPatterns

ExpStudy
├── ExpStudyPlan
│   ├── Abstract
│   ├── ExpSharedPlan
│   ├── ExpInvestigationPlan[]
│   ├── ExpMethodsPlan
│   ├── ExpConclusionSlot
│   ├── ExpReferences?
│   └── ExpAppendices?
└── StudyExecution
```

An experiment identity owns one continuing central scientific question.
Parameter refinement, an additional investigation, a new visualization, a
control, a failed run, or follow-up execution defaults to the same experiment.
After evidence-bearing execution, revise prospective material in the same
writing canvas before the next affected execution. Each finalized
`ExperimentRun` preserves the exact writing, implementation, configuration,
source revision, and dirty-patch provenance it executed, so later changes do
not rewrite the earlier plan. Do not create a new experiment merely because a
run already exists.
Create one only when the central scientific question materially changes or the
user explicitly requests it; when that distinction is consequential, return it
to the user rather than creating the experiment automatically.

`Plan` always means prospective. Its paired executed noun preserves the
prospective state recorded by its run and adds what actually happened without
rewriting expectations as observations. A new study plan is derived from the
scout rather than extending or relabelling it; record which exploratory choices
were carried forward, changed, rejected, or added.

The rendered executed artifact follows its prospective publication scaffold
and adds execution evidence beside the corresponding prospective material. It
may replace planned methods with one complete account of executed methods for
readability, while retaining the executed protocol through run provenance.

In every rendered `ExpScoutPlan`, `ExpScout`, `ExpStudyPlan`, and `ExpStudy`,
group the ordered investigation collection beneath one top-level section titled
`Results`. Title each investigation from its descriptive
`ExpInvestigationIdentity`; do not expose the noun name as its heading. In a
prospective plan, keep expected patterns and result slots explicitly planned or
pending so the `Results` scaffold cannot imply that observations exist. After
execution, place each result directly after its recorded expected patterns and
planned visual evidence within the same custom-titled investigation. Mark
simulated evidence plainly as **Simulation result** in its lead text or caption;
“measured” alone is not sufficient when readers could mistake model output for
biological measurement. Do not duplicate local evidence in a second aggregate
results overview. A cross-investigation figure may appear near the conclusion
only when it adds genuine synthesis, and it never substitutes for the local
results. Keep `ExpConclusion` outside the `Results` section.

Composition is not publication anatomy. Child nouns are semantic inputs to a
connected scientific narrative, not mandatory paragraphs, labels, cards, or
subsections. In rendered experiment documents, synthesize adjacent children
into prose that develops an argument. Do not print noun names, field names, or
repeated inline labels. Use headings only for substantive document sections and
descriptively named investigations or methods.

Give the most narrative and visual space to the evidence that bears most
strongly on the central question. Keep setup sufficient to interpret that
evidence, and keep illustrative mechanisms subordinate to aggregate results.
Combine short related material, remove repeated framing and limitations, and
use transitions to explain why the next evidence is needed. Review the rendered
document as a whole: avoid stranded headings, broken figure sequences, choppy
runs of small sections, and nearly empty final pages. Let scientific importance
determine length; do not target uniform sections or a fixed page count.

Investigations remain one ordered flat collection within `Results`. A rendering
may place consecutive investigations beneath custom descriptive subgroup
headings when that clarifies the experiment's scientific logic. These headings
are optional presentation, not nouns or additional semantic hierarchy: they do
not own, renumber, or change the identity of an investigation, and no fixed
subgroup taxonomy is implied. Introduce each subgroup with prose that explains
the shared question and its relationship to neighbouring groups.

Rendered experiment documents contain scientific content, not repository
plumbing. Omit opaque run and campaign identifiers, commit hashes, checkpoint
keys, filenames, paths, manifests, commands, and implementation module names.
Retain those exact details in native artifacts and `ScientificCollectionState`.
Include a parameter in the document only when it is scientifically meaningful
to interpretation or reproduction; express model and data sources in
human-readable scientific language.

The experiment family uses the registered composition nouns defined below.
Each also independently triggers this skill.

## `ExpSharedPlan`

A compact common prospective orientation containing an optional
`ExpDesignSchematic`. It states the identity, status, dependencies, scientific
frame, shared inputs and controls, decision gates, budget, assumptions, and
scope needed to understand the investigations. Keep detailed protocol and local
rationale outside it. State shared material once.

Use the schematic when the common mechanism or experimental design would
otherwise require extended prose. The schematic replaces explanation rather
than decorating or enlarging the section.

## `ExpInvestigationIdentity`

The stable number and descriptive name that identify one investigation and
join its planned method, output, and later execution.

## `ExpInvestigationIntroduction`

The prospective rationale for one investigation: the uncertainty it resolves,
why that uncertainty matters, and how the investigation relates to the shared
scientific frame. It includes only enough method to understand the planned
evidence.

## `ExpExpectedPatterns`

Conditional predictions for one investigation under the leading hypothesis
and serious rivals, including the observations that distinguish them. It is
prospective and must not be rewritten after execution to match the result.

## `ExpInvestigationPlan`

A locally complete prospective investigation containing an
`ExpInvestigationIdentity`, `ExpInvestigationIntroduction`,
`ExpExpectedPatterns`, optional `ExpVisualSet`, planned output, protocol links,
and the limits of what its result could establish.

## `ExpMethodsPlan`

The complete prospective protocol: configuration, datasets, parameter values,
sampling, execution sequence, analysis definitions, and scientifically
meaningful reproducibility requirements. Number its subsections so
investigations can reference them without duplicating procedural detail.

## `ExpConclusionSlot`

A reserved publication position for the executed conclusion. It contains no
predicted conclusion and is omitted from a prospective rendering.

## `ExpReferences`

The external sources that materially inform an experiment plan or its executed
record. Keep references attached to the claims or choices they support.

## `ExpAppendices`

Supporting protocol, provenance, calculations, or diagnostics whose inclusion
in the main sequence would interrupt the experiment's evidence logic.

## `ExpScoutPlan`

A prospective, budgeted reconnaissance contract and publication scaffold. It
defines cheap tests for feasibility, pattern discovery, and deciding whether
deeper study is warranted.

Use `status: "draft"` while a design is being refined and has not been run.

Use this canonical reading order:

1. `Abstract`
2. `ExpSharedPlan`
3. `ExpInvestigationPlan[]`
4. `ExpMethodsPlan`
5. `ExpConclusionSlot`
6. optional `ExpReferences`
7. optional `ExpAppendices`

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

Every investigation must resolve a scientific uncertainty through a
scientifically interpretable measurement. Treat implementation capabilities,
data extraction, artifact generation, and publication delivery as declared
dependencies rather than experimental uncertainties. Move their operational
handoff to `ScientificCollectionState`, an implementation specification, or
native protocol artifacts. Do not promote a prerequisite already established
by repository evidence into an investigation merely to verify that execution
is possible. Scientific datasets, measurement definitions, sampling, controls,
and reproducibility parameters remain part of the plan when they affect the
meaning or validity of its evidence.

## `ExpVisualSet`

An optional visual evidence scaffold containing an `ExpDesignSchematic` and an
`ExpMeasuredResultSlot`. The pair preserves a shared panel layout, variables,
colours, visual grammar, interpretive purpose, and caption.

The default epistemic colour grammar uses blue-grey for conceptual or
prospective schematics and red-black for measured evidence. A scientific
semantic, accessibility need, or established figure convention may override
it, but the plan must document the new mapping and keep schematic and measured
roles visibly distinct. Never style an unexecuted or conceptual curve as
red-black measured evidence.

## `ExpDesignSchematic`

A prospective, hand-authored explanation of a mechanism, intervention, or
expected visual structure. It specifies the panel layout, variables, colours,
visual grammar, interpretive purpose, caption, and links to its investigation
or the shared scientific frame and relevant protocol. It is conceptual evidence
design, never an observed result.

## `ExpMeasuredResultSlot`

The prospective placeholder for a structurally matched measured output and its
completion status. It contains no observations before execution.

## `ExpScoutSummary`

The highest-level execution summary of a scout: its principal observation and
stop, revise, or escalate disposition without procedural detail.

## `ExpSharedExecution`

The shared execution record containing completion status, scientifically
meaningful actual configuration, deviations, and limitations. Record common
execution facts once rather than repeating them locally.

## `ExpMeasuredResult`

An executed output with its value, figure, table, or other result, completion
status, and measurement-specific limitations. A failed, incomplete, or not-run
status is preserved explicitly rather than silently omitted.

## `ExpObservedPatterns`

A concise, exploratory account of what a measured result shows and does not
show relative to the expected patterns recorded by that execution. Keep
observations separate from procedural detail and do not add a local disposition
gate.

## `ExpInvestigationExecution`

An execution unit containing an `ExpMeasuredResult` or explicit non-completion
status, an optional completed `ExpVisualSet`, and `ExpObservedPatterns`.

## `ExpMethodsExecuted`

The complete scientific account of methods actually executed, concrete
outputs, and scientifically meaningful deviations.

## `ExpConclusion`

The cross-investigation interpretation against shared decision gates and
rivals, including limitations and the stop, revise, or escalate disposition.

## `ScoutExecution`

The execution overlay comprising an `ExpScoutSummary`, `ExpSharedExecution`,
one `ExpInvestigationExecution` per planned investigation,
`ExpMethodsExecuted`, and `ExpConclusion`.

## `ExpScout`

An executed scouting mission containing the prospective `ExpScoutPlan` recorded
by its `ExperimentRun` and `ScoutExecution`.

Preserve the prospective publication structure and attach execution evidence
locally within the shared `Results` section. Replace the planned methods only in
the readable rendering; retain the executed protocol through run provenance. Do
not rewrite expectations as observations or create a second results section. All
evidence remains exploratory rather than durable.

Keep rendered scouts proportionate to reconnaissance. Write the abstract as one
short qualitative paragraph followed by the required plain-language sentence.
Keep design and scope to the minimum needed to understand the question, shared
setup, boundaries, and decision gates; prefer direct human-readable prose over
contract-like enumeration. For each investigation, use one compact prospective
passage before its output and one compact observed interpretation after it. Let
figures, captions, and methods carry detail instead of repeating it in the
surrounding prose. Keep `ExpMethodsExecuted` complete. End with a short
disposition-led conclusion containing only the decisive cross-investigation
interpretation, material limitation, and warranted next step. Brevity never
permits removing epistemic status, serious rivals, distinguishing expectations,
failed or incomplete results, material deviations, or limitations needed to
interpret the evidence. This rendering rule does not apply to `ExpStudy`.

Number every body heading hierarchically. The publication title, figure
captions, equations, and inline structural labels such as `Relevance` and
visual-set labels are not body headings and remain unnumbered. Optional
trailing sections are omitted rather than emitted empty; retain the canonical
section number when one is present.

Attach completed measured mirrors with code and data provenance when planned.
Preserve failed or incomplete result slots. A scout without visual scaffolding
remains valid and may not be promoted or relabelled as durable evidence.

## `ExpStudyPlan`

A prospective durable-study contract containing an `Abstract`,
`ExpSharedPlan`, one or more `ExpInvestigationPlan` entries, an
`ExpMethodsPlan`, `ExpConclusionSlot`, and optional `ExpReferences` and
`ExpAppendices`.

It requires stronger estimands, sampling and seeds, controls, uncertainty,
falsifiers, rival discrimination, stopping rules, and robustness than a scout
plan. The resulting `ExperimentRun` records the exact prospective choices used.

## `StudyExecution`

The durable execution overlay containing completion status, scientifically
meaningful actual configuration and deviations, observations and uncertainty,
rival discrimination, conclusions, limitations, and robustness results.

## `ExpStudy`

A durable executed scientific record containing the prospective `ExpStudyPlan`
recorded by its `ExperimentRun` and `StudyExecution`.

For each completed result, show its title, figure or output, and concise
interpretation. Never replace or rewrite planned expectations as observations.

## `ScientificCollectionState`

The current collection registration, experiment membership and scientific
roles, hard dependencies, lifecycle status, campaign readiness, writing
metadata, referenced artifacts, `PublicationView`, generated outputs, and
publication blockers. An experiment enters this state when its plan and
implementation are created. Constructing a `CampaignPlan` captures an explicit
collection snapshot without introducing a separate experiment-plan approval
gate.

Technical provenance includes exact run and campaign identifiers, commit
hashes, checkpoint keys, filenames, paths, manifests, commands, and
implementation module names; it remains outside rendered experiment prose.
