# Global rules

## Visual index

- [Canvas ownership](#canvas-ownership)
- [Human-facing writing](#human-facing-writing)
- [Scientific lifecycle](#scientific-lifecycle)
- [Progressive teaching](#progressive-teaching)

Pinglab's scientific vocabulary consists of primitive nouns. The nouns are
prose-defined semantic structures rather than file types or formally
schema-validated objects. Ordinary conversation constructs and transforms them.
Repository evidence and publication outputs retain their native files alongside
the human-facing writing that interprets them.

## Human-facing writing

Apply this writing contract to abstracts, evidence capsules, experiment prose,
captions, conclusions, and other text intended for a reader. It adapts useful
parts of ASD-STE100 Simplified Technical English and William Zinsser's clarity,
simplicity, brevity, and humanity. It does not make the text ASD-STE100
compliant and does not import that standard's controlled dictionary.

Use plainspoken scientific notebook prose aimed at an informed lab colleague.
Prefer active, concrete explanations and define specialist terms when they
first matter. Be direct about observed evidence and restrained about
interpretation. Avoid compressed journal formality, promotional language, and
explanations that become childish or scientifically vague.

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

## Scientific lifecycle

Use this artifact-centred loop as lightweight orientation for scientific work:

1. Develop `Seed`, `Formulation`, `HypoBranches`, and grounding evidence in
   conversation. Ask the user to select the scientific direction when that
   choice is consequential.
2. Before creating an experiment, recommend `Scope a scout in COLLECTION` or
   the corresponding study request. `Scope` is read-only and identifies the
   question, collection, dependencies, outputs, implementation, resources, and
   completion conditions.
3. An authorized `go` mode writes the `ExpScoutPlan` or `ExpStudyPlan` directly
   into the experiment's persistent writing canvas, creates its
   `ExpImplementation`, and registers the experiment in managed
   `ScientificCollectionState`. The work remains a plan until its first
   successful evidence-bearing execution. Do not insert a chat-only plan stage
   or create a parallel noun file.
4. Within the established question and authorized scope, develop the writing
   and implementation, test them, run bounded local work, analyse outputs, and
   revise the experiment agentically. Return to conversation when the central
   scientific question or required authority changes.
5. Use the Demolab-configured development and build interfaces throughout this
   loop. While the development interface is running, it reacts to changes in
   writing and the selected evidence; presentation is continuous feedback, not
   a terminal publication step. Demolab does not select or accept evidence.
6. Each useful execution creates an experiment-scoped `ExperimentRun` in the
   collection's working `CollectionDataset`. A successfully finalized run is
   immutable and automatically becomes that experiment's official evidence.
   Failed, interrupted, or incomplete runs do not replace the latest successful
   run.
7. When integrated execution is warranted, construct a dry `CampaignPlan` from
   an explicit snapshot of `ScientificCollectionState`. Ask the user to review
   its included experiment versions, dependencies, resources, expected outputs,
   exclusions, and acceptance conditions, and to authorize any named paid
   compute target explicitly.
8. Execute the approved plan as `CampaignExecution`, producing one finalized
   `ExperimentRun` for each successfully completed experiment. Archival,
   verification, restoration, repair, and composition preserve run lineage.
9. Interpret the evidence and update the experiment's writing. Continue useful
   controls, visualizations, parameter changes, and additional investigations
   under the same experiment identity. An immutable dataset snapshot may later
   preserve a selected collection state for archival or publication.
   `PublicationView` materializes the current official evidence, and Demolab
   reacts to that view.
10. Update `ScientificRecord` and `ScientificCollectionState`, and turn valuable
   unresolved uncertainty, failures, or new findings into the next `Seed`.

Between human review gates, continue agentically within the authorized scope.
Return to human-agent conversation for a materially changed scientific
question, new authority, expanded scope, paid compute, destructive action,
promotion, activation, archival, publication, or an interpretation that the
available evidence cannot resolve. Never treat silence as approval. Work may
enter, leave, repeat, or move backward through the loop as the evidence
requires.

### Progressive teaching

Teach the lifecycle through use rather than reciting it. During ordinary
iteration, state the current experiment and run, what changed, the supported
interpretation, and one useful next action. Keep noun names backstage unless
they clarify a transition or the user asks about them. Prefer natural language
over requiring a formal lifecycle command. The labels `[CHAT]`, `[ARTIFACT]`,
`[GATE]`, and `[REACTIVE]` may make unfamiliar roles clear on first use.

Reduce this guidance as the user demonstrates familiarity: move from a short
explanation to the current state and one next operation. Do not quiz the user,
recite the whole lifecycle unless asked, require named artifacts for trivial
exploration, invent a proficiency score, or repeatedly explain familiar terms.
When asked where the work is in the lifecycle, report the current state, live
canvasses, latest run, supported interpretation, and one next operation.

After execution, prefer a compact report such as:

> `exp099/r003` completed and is now official. The observed transition is
> synchronous but not yet established as PING. Next useful test: remove the
> E-I cross-coupling.

## Canvas ownership

For an experiment, its configured `.typ` writing is the sole authored
scientific canvas. Compose its prospective plan, execution account, and other
experiment nouns within that writing. Its experiment code and tests are the
sole authored computational canvasses. Do not create parallel Markdown records,
one file per noun, or experiment-local collection-state documents. Run payloads
contain native evidence and provenance such as data, figures, manifests, logs,
source identity, and dirty patches; they do not become additional narrative
canvasses. Managed collection and dataset interfaces own
`ScientificCollectionState`, `CollectionDataset`, and evidence selection.
