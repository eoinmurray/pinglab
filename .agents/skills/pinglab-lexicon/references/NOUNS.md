# Nouns

## Visual index

- [Canvas](#canvas)
- [Scientific record](#scientificrecord)
- [Abstract and writing objects](#abstract)
- [Hypothesis search](#hypothesis-search-family)
- [ExperimentFlow](#experimentflow)
- [Execution and datasets](#execution-artifact-family)
- [Experiment](#experiment)
- [Campaigns and publication](#campaignplan)
- [Collection state](#scientificcollectionstate)

## `Canvas`

A lightweight interactive scientific surface for exploring a mechanism,
parameter, model, dataset, or proposed experiment through controls and immediate
visual feedback.

Use the smallest scientifically meaningful model and expose the variables that
answer the user's question. Distinguish conceptual or simulated behaviour from
biological measurement, state important simplifications, and avoid presenting
illustrative output as evidence.

After a `Canvas` interaction changes executable code, validate the affected code
and run the bounded local simulation so the displayed output reflects the change.
Do not run a simulation for prose, styling, metadata, or replay-only changes. If
validation fails, execution is unsafe, or the run requires paid compute, stop and
report the blocker rather than bypassing the applicable authority gate.

After a `Canvas` runs a simulation, report its elapsed simulation runtime in the
conversational update. If an interrupted or failed simulation has no available
runtime, state that it is unavailable rather than estimating it. Presentation-only
changes and replays of retained output without a new simulation do not require a
simulation runtime.

A `Canvas` is conversational and exploratory. Creating or changing one does not
create an `Experiment`, implementation, `ExperimentRun`, official evidence,
or publication state. If the user later asks to retain or execute its scientific
question, recommend the appropriate scoped experiment operation rather than
silently promoting the canvas.

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

## `ExperimentFlow`

Pinglab's turn-based experiment-authoring system. For each persistent experiment,
it coordinates mutable `.typ` writing, mutable `.py` implementation and tests,
and an immutable `ExperimentRun` evidence history. It infers the smallest object
changes required by each turn while preserving experiment identity, evidence
provenance, and human authority gates.

## Execution-artifact family

Execution-artifact nouns connect prospective experiment plans to native code,
runs, durable evidence, and the continuously rendered presentation. They name
ownership boundaries rather than requiring every native file to be rewritten as
Markdown.

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

Create a new run when execution changes observations, measurements, source
data, simulation outputs, or derived evidence. Styling, layout, captions, and
other presentation-only changes reuse the existing run when its evidence is
unchanged. Never rerun a simulation solely to change its presentation when the
retained evidence can be rendered directly.

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

## `Experiment`

A mutable scientific composition that develops around one continuing central
question. It has one persistent writing surface and combines these components
in canonical reading order:

```text
Experiment
├── Abstract
├── MediaArea
├── Results
├── Methods
├── Conclusion
├── References?
└── Appendices?
```

Only `Abstract` is also a registered Pinglab noun. `MediaArea`, `Results`,
`Methods`, `Conclusion`, `References`, and `Appendices` are components
of `Experiment`, not independently invokable nouns or separate artifacts.
Components may be incomplete, reordered during authoring, or absent when
optional. The canonical order governs the coherent human-facing rendering.

An experiment may begin with any useful fragment and gradually acquire
structure. Do not require a complete plan, a scout/study classification, or
separate planning and execution objects before material can be retained.
Parameter refinement, an additional result, a new visualization, a control, a
failed run, or follow-up execution defaults to the same experiment. Create a
new experiment only when the central scientific question materially changes or
the user explicitly requests one.

Each element preserves its epistemic status where that status matters:
`idea`, `planned`, `expected`, `observed`, or `interpreted`. These
statuses describe content within the experiment; they are not lifecycle nouns.
Never rewrite an expectation to match a later observation. Each finalized
`ExperimentRun` preserves the exact writing, implementation, configuration,
source revision, and dirty-patch provenance it executed, so later composition
does not rewrite earlier evidence.

### MediaArea

The loose working surface for images, videos, diagrams, plots, animations, and
other media that may help the experiment take shape. Material may be placed
here before its scientific role or final position is known.

Media in `MediaArea` is not evidence merely because it appears in the
experiment. Mark conceptual, simulated, replayed, and empirically measured
media accurately. When media becomes a major scientific output, place or link
it in the relevant `Results` entry while retaining any useful working context.

When an experiment uses SNNLANG and its authored graph can be rendered, include
a compiled network diagram in `MediaArea` or beside the result it explains.
Treat it as a structural schematic, not measured evidence. Use the deepest view
that remains scientifically legible: expose relevant populations, sizes,
projections, direction, polarity, and external inputs without descending to
individual neurons or matrices unless scientifically necessary.

### Results

The structured collection of major scientific outputs. Give each result a
stable descriptive title and enough local structure to preserve:

1. the question or uncertainty it addresses;
2. planned or expected patterns when they exist;
3. the output, including explicit failed, incomplete, or not-run status;
4. what the output shows and does not show;
5. material limitations and deviations; and
6. its relationship to the central question or serious rivals.

Keep results as one ordered flat collection. Optional descriptive subgroup
headings may clarify the scientific logic, but they do not create additional
semantic hierarchy. Place each output beside its local expectations and
interpretation. Do not duplicate the same evidence in a second aggregate
results overview.

Mark simulated evidence plainly as **Simulation result** in its lead text or
caption when readers could mistake it for biological or empirical measurement.
Conceptual diagrams and unexecuted curves must remain visibly distinct from
observed evidence.

### Methods

The reproducible steps that create or analyse the experiment's evidence. Record
configuration, datasets, parameters, sampling, execution sequence, analysis
definitions, controls, and scientifically meaningful reproducibility details
in proportion to their importance.

Mark steps as planned or executed when the distinction matters. After
execution, record what actually happened, including material deviations,
without erasing the prospective method preserved by its `ExperimentRun`.
Prefer one action per procedural step. Reusable computation belongs to the
tool; hypothesis-specific conditions, analysis, and figures belong to the
experiment.

### Conclusion

The cross-result interpretation against the central question, expectations,
serious rivals, and decision gates. State limitations and the warranted next
disposition without procedural repetition. A conclusion interprets evidence;
it does not modify the recorded results or turn missing evidence into support.

### References

Optional external sources that materially inform the experiment. Keep
references attached to the claims or choices they support.

### Appendices

Optional supporting protocol, provenance, calculations, or diagnostics whose
inclusion in the main sequence would interrupt the experiment's evidence logic.

### Rendering

Synthesize the components into connected scientific notebook prose rather than
printing component names, field labels, cards, or contract-like enumeration.
Use headings only for substantive document sections and descriptively named
results. Give the most visual and narrative space to the evidence that bears
most strongly on the central question.

Rendered experiments contain scientific content, not repository plumbing. Omit
opaque run and campaign identifiers, commit hashes, checkpoint keys, filenames,
paths, manifests, commands, and implementation module names. Retain those exact
details in native artifacts and `ScientificCollectionState`. Include a
parameter only when it is scientifically meaningful to interpretation or
reproduction.

Review the rendered document as a whole. Avoid stranded headings, broken figure
sequences, choppy runs of small sections, and nearly empty final pages.
Optional trailing sections are omitted rather than emitted empty.

## `ScientificCollectionState`

The current collection registration, experiment membership and scientific
roles, hard dependencies, lifecycle status, campaign readiness, writing
metadata, referenced artifacts, `PublicationView`, generated outputs, and
publication blockers. An experiment enters this state when its writing and
implementation are created. Constructing a `CampaignPlan` captures an explicit
collection snapshot without introducing a separate experiment approval gate.

Technical provenance includes exact run and campaign identifiers, commit
hashes, checkpoint keys, filenames, paths, manifests, commands, and
implementation module names; it remains outside rendered experiment prose.
