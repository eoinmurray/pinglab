---
name: pinglab-lexicon
description: Interpret, construct, transform, review, or serialize Pinglab noun types whenever a prompt invokes ScientificRecord, PinglabAbstract, Seed, Formulation, HypoBranches, HypoCanon, HypoLiterature, HypoRepository, OpenSearchTrajectory, HypoCheckpoint, GroundedSearchTrajectory, HypoPacket, ExpScoutPlan, ExpPlanAbstract, ExpSharedPlan, ExpInvestigationPlan, ExpInvestigationIdentity, ExpInvestigationIntroduction, ExpExpectedPatterns, ExpVisualSet, ExpDesignSchematic, ExpMeasuredResultSlot, ExpMethodsPlan, ExpConclusionSlot, ExpReferences, ExpAppendices, ExpScout, ScoutExecution, ExpScoutSummary, ExpSharedExecution, ExpInvestigationExecution, ExpMeasuredResult, ExpObservedPatterns, ExpMethodsExecuted, ExpConclusion, ExpStudyPlan, StudyExecution, ExpStudy, or ScientificCollectionState in any case, spacing, joining, or optional dollar-prefixed form, or asks to scope encode a rule into the Pinglab writing system.
---

# Pinglab noun registry

Pinglab's scientific vocabulary consists of primitive nouns. The nouns are
prose-defined and serialized as Markdown rather than formally schema-validated.
Ordinary conversation constructs and transforms them. Repository evidence and
publication outputs retain their native files alongside the textual noun
describing them.

## Recognition

Recognize every Pinglab noun regardless of letter case. Treat joined and
whitespace-separated noun words as equivalent where the canonical form is
unambiguous: for example, `ScientificRecord` and `scientific record`, or
`ExpScoutPlan` and `exp scout plan`. Allow an optional `$` before the complete
noun or before any noun word, so `$ExpScoutPlan`, `$exp scout plan`, and
`$exp $scout $plan` are aliases of `ExpScoutPlan`.

Apply this normalization only when identifying a Pinglab noun, not to an
ordinary phrase that happens to use the same words. Use the canonical noun
names below when constructing or reporting artifacts.

## Encode and self-update

`scope encode RULE` requests a read-only specification for the smallest
coherent change that brings `RULE` into the Pinglab writing system. Recognize
`scope encode` regardless of case and allow `$scope`, `$encode`, or both. The
command does not authorize editing; a later Lexicon `go` mode supplies that
authority.

To encode a rule:

1. State the observable writing decision that the rule must change.
2. Check the live skill for an existing rule with the same effect. Return a
   no-op when it already exists, or refine the existing rule instead of adding
   a duplicate.
3. Reject a one-off edit, vague preference, anecdote, or rule that would weaken
   scientific accuracy, epistemic boundaries, user intent, or a noun contract.
4. Select the narrowest owner: one noun, one noun family, the shared
   human-facing writing contract, or recognition metadata when invocation must
   change.
5. Integrate the rule into the owner's existing instructions. State its scope,
   precedence, and material exclusions without growing an exception list.
6. Specify one positive case, one boundary case, and one conflicting case that
   the implementation must handle correctly.
7. Validate the changed skill with the skill validator and inspect the final
   diff for duplicated guidance, scope drift, and accidental new authority.

This skill updates itself through the same process. An Encode specification can
target this section, recognition, a shared rule, or a noun definition. It must
still propose the minimum change, preserve higher-priority constraints, and
wait for an authorized `go` mode before mutation. Encoding a rule never grants
the rule, the skill, or the agent permission to publish or perform unrelated
work.

## Human-facing writing

Apply this writing contract to abstracts, evidence capsules, experiment prose,
captions, conclusions, and other text intended for a reader. It adapts useful
parts of ASD-STE100 Simplified Technical English and William Zinsser's clarity,
simplicity, brevity, and humanity. It does not make the text ASD-STE100
compliant and does not import that standard's controlled dictionary.

Revise in this order:

1. Preserve scientific accuracy, epistemic status, uncertainty, defined
   terminology, and the noun's structural contract. These take priority over
   brevity or style.
2. Give each paragraph one topic and open with the sentence that orients the
   reader. Develop information gradually from question or context, through
   mechanism and evidence, to consequence.
3. Give each sentence one main claim, action, or contrast. Prefer a concrete
   subject and an active verb when the real agent is known. Retain passive
   voice when the agent is unknown, irrelevant, or less important than the
   scientific object.
4. Use one stable term for each concept. Define unfamiliar abbreviations and
   specialist terms on first use. Do not replace precise scientific terms with
   simpler words that change their meaning.
5. Prefer concrete nouns and strong verbs. Remove empty qualifiers, repeated
   framing, avoidable nominalizations, stacked noun phrases, and implementation
   jargon that does not help interpretation.
6. In procedures, use one action per step and the imperative form. Where the
   noun permits lists, use them for genuinely parallel items or complex
   sequences rather than embedding the sequence in one sentence.
7. Keep sentences short enough to understand in one reading. Treat 20 words
   for procedural sentences and 25 words for descriptive sentences as prompts
   to inspect the sentence, not hard limits. Equations, citations, necessary
   terms, and accurate qualifications can justify a longer sentence.
8. Preserve humanity through intelligible stakes, natural rhythm, and an
   identifiable reasoning voice. Do not manufacture intimacy, anthropomorphize
   models, dramatize evidence, or trade precision for personality.
9. After the epistemic review, do one final pass for compression, continuity,
   and rhythm. Remove words only when the meaning and evidentiary boundaries
   remain intact.

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

## Experiment family

Experiment noun definitions form a directed composition map. A composite noun
names only its immediate children. A leaf noun names no other experiment noun.
Children never name their parent, and definitions do not encode sibling,
provenance, lifecycle, or downstream relationships.

The composition map is:

```text
ExpScout
├── ExpScoutPlan
│   ├── ExpPlanAbstract
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
│   ├── ExpPlanAbstract
│   ├── ExpSharedPlan
│   ├── ExpInvestigationPlan[]
│   ├── ExpMethodsPlan
│   ├── ExpConclusionSlot
│   ├── ExpReferences?
│   └── ExpAppendices?
└── StudyExecution
```

`Plan` always means prospective. Its paired executed noun preserves the frozen
plan and adds what actually happened without rewriting expectations as
observations. A new study plan is derived from the scout rather than extending
or relabelling it; record which exploratory choices were carried forward,
changed, rejected, or added.

The rendered executed artifact follows its prospective publication scaffold
and adds execution evidence beside the corresponding prospective material. It
may replace planned methods with one complete account of executed methods for
readability, while retaining the frozen protocol for provenance.

Composition is not publication anatomy. Child nouns are semantic inputs to a
connected scientific narrative, not mandatory paragraphs, labels, cards, or
subsections. In rendered experiment documents, synthesize adjacent children
into prose that develops an argument. Do not print noun names, field names, or
repeated inline labels. Use headings only for substantive document sections and
descriptively named investigations or methods.

Investigations remain one ordered flat collection. A rendering may place
consecutive investigations beneath custom descriptive subgroup headings when
that clarifies the experiment's scientific logic. These headings are optional
presentation, not nouns or additional hierarchy: they do not own, renumber, or
change the identity of an investigation, and no fixed subgroup taxonomy is
implied. Introduce each subgroup with prose that explains the shared question
and its relationship to neighbouring groups.

Rendered experiment documents contain scientific content, not repository
plumbing. Omit opaque run and campaign identifiers, commit hashes, checkpoint
keys, filenames, paths, manifests, commands, and implementation module names.
Retain those exact details in native artifacts and `ScientificCollectionState`.
Include a parameter in the document only when it is scientifically meaningful
to interpretation or reproduction; express model and data sources in
human-readable scientific language.

The experiment family uses the registered composition nouns defined below.
Each also independently triggers this skill.

## `ExpPlanAbstract`

A prospective summary stating the question, hypothesis, intervention, primary
estimand, and consequential outcome of an experiment plan. It contains no
observed result or disposition.

## `ExpSharedPlan`

The common prospective contract for identity, status, collection,
dependencies, scientific frame, inputs, controls, decision gates, budget,
assumptions, and scope. State shared material once here rather than duplicating
it across investigations.

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

1. `ExpPlanAbstract`
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
and protocol. It is conceptual evidence design, never an observed result.

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
show relative to the frozen expected patterns. Keep observations separate from
procedural detail and do not add a local disposition gate.

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

An executed scouting mission containing a frozen `ExpScoutPlan` and
`ScoutExecution`.

Preserve the prospective publication structure and attach execution evidence
locally. Replace the planned methods only in the readable rendering; retain the
frozen protocol for provenance. Do not rewrite expectations as observations or
create a separate results section. All evidence remains exploratory rather than
durable.

Number every body heading hierarchically. The publication title, figure
captions, equations, and inline structural labels such as `Relevance` and
visual-set labels are not body headings and remain unnumbered. Optional
trailing sections are omitted rather than emitted empty; retain the canonical
section number when one is present.

Attach completed measured mirrors with code and data provenance when planned.
Preserve failed or incomplete result slots. A scout without visual scaffolding
remains valid and may not be promoted or relabelled as durable evidence.

## `ExpStudyPlan`

A prospective durable-study contract containing an `ExpPlanAbstract`,
`ExpSharedPlan`, one or more `ExpInvestigationPlan` entries, an
`ExpMethodsPlan`, `ExpConclusionSlot`, and optional `ExpReferences` and
`ExpAppendices`.

It requires stronger estimands, sampling and seeds, controls, uncertainty,
falsifiers, rival discrimination, stopping rules, and robustness than a scout
plan. Freeze all choices before execution begins.

## `StudyExecution`

The durable execution overlay containing completion status, scientifically
meaningful actual configuration and deviations, observations and uncertainty,
rival discrimination, conclusions, limitations, and robustness results.

## `ExpStudy`

A durable executed scientific record containing a frozen `ExpStudyPlan` and
`StudyExecution`.

For each completed result, show its title, figure or output, and concise
interpretation. Never replace or rewrite planned expectations as observations.

## `ScientificCollectionState`

The current collection registration, writing metadata, referenced artifacts,
technical provenance, generated outputs, and publication blockers. Technical
provenance includes exact run and campaign identifiers, commit hashes,
checkpoint keys, filenames, paths, manifests, commands, and implementation
module names; it remains outside rendered experiment prose.
