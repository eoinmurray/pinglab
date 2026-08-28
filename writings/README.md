# Writing Guide

Version: **17.0.0**

The Writing Guide defines the conventions for Pinglab's published experiment
entries in `writings/expXXX.typ`. This file is the canonical guide.

## 1. Versioning

Version this guide independently of Pinglab and Demolab. Increment the major
version when changed requirements make previously compliant writing require
revision, the minor version for compatible additions, and the patch version for
corrections or clarifications that do not change requirements. Update the version
above and add a short entry to the version history when changing the guide.

### 1.1. Version history

- **17.0.0** — Require the plain `Results` heading, with no tagline or
  description. Remove the finding-summary requirement and migrate existing
  headings and dependent links; preserve scientific content and authored dates.

- **16.0.0** — Prohibit section numbering at every article heading level,
  including appendices and References. Require Results before Methods, migrate
  whole sections without rewriting their content, and repair named references.
- **15.0.0** — Results taglines must add a concrete evidence summary beyond
  the experiment title; prohibit copied titles and synonymous restatements.
- **14.0.0** — Require `Results: <terse finding>` headings that state the
  principal supported finding or reusable output, including negative results
  and trade-offs. Keep evidence limits explicit and migrate section links.
- **13.0.0** — Require an article-scoped Table of Contents before Abstract
  in every experiment entry, generated from rendered section headings. Replace
  manual contents lists; preserve scientific content and authored dates.
- **12.0.0** — Add monochrome icons to the exact status strings: `[≡ TXT]`
  and `[▦ DATA]`. Preserve brackets, availability meanings and recorded
  classifications; status-only edits preserve dates.
- **11.0.0** — Require the compact exact status strings `[TXT]` and `[DATA]`,
  without descriptive suffixes. Availability meanings and agent freshness
  responsibilities are unchanged.
- **10.0.0** — Replace research milestones with `[TXT] Article only` and
  `[DATA] Local data`. Agents maintain these local-data availability badges for
  affected articles and dependent comparisons; status-only edits preserve dates.
- **9.0.0** — Require milestone-based `meta.status` on every experiment entry,
  including reference documents. Define evidence-based authoring updates,
  explicit author review and pause decisions, and a `Drafted` milestone for
  written accounts without established implementation or results.
- **8.1.1** — Clarify that incorporating a present run advances the article's
  update date when displayed evidence changes substantively, even without prose
  changes; run completion and equivalent outputs alone do not qualify.
- **8.1.0** — Add prospective authored-date rules: update `meta.updated_at`
  after substantive article revisions, at calendar-day granularity; preserve
  creation dates and do not backfill unchanged articles.
- **8.0.0** — Restrict Methods items to substantive scientific operations. Remove
  the preferred minimum item count and prohibit standalone steps for routine
  presentation and summarization; keep necessary figure-reading details in
  captions and consequential measurement choices in the relevant scientific step.
- **7.0.0** — Require validated v3 present runs for article inputs, aligning with
  Storage Guide 2.0.0. Remove the v2 presentation allowance; article-scoped data
  bindings and scientific writing requirements remain unchanged.
- **6.3.0** — Remove the mandatory Inputs and outputs, Design Scope and Prior
  art sections. Retain technical data-access rules and scientific coverage in
  Methods without prescribing separate sections for these topics.
- **6.2.0** — Ground resampling in the live document, preserve scientific
  content and settled author decisions, and require explicit authorization to
  save or replace the candidate.
- **6.1.0** — Suggest evidence-led improvements while applying the guide,
  distinguishing reusable rules from experiment-specific choices and requiring
  explicit approval before changing the guide or broadening article edits.
- **6.0.0** — Make Methods concise and skimmable, with a 300–450-word target,
  short action-led steps, selected key equations and a compression pass that
  preserves scientific coverage. Clarify the boundary with appendix detail.
- **5.0.0** — Ground Methods in execution code and retained evidence, require a
  complete scientific procedure through selection, measurement and outputs, and
  include equations only when they clarify consequential operations.
- **4.0.0** — Make Results a scaffold of numbered headings,
  figures and concise captions, without narrative prose or prose placeholders.
  Update the example and remove conflicting guidance in Methods.
- **3.0.0** — Require code- and evidence-grounded abstracts for rapid browsing,
  leading with the experiment's finding or reusable output. Replace the abstract
  example with exp022's verified training-bank summary and allow 60–120 words.
- **2.0.0** — Apply repository-independent scientific writing and the ban on
  internal references to all rendered article content, not only Methods.
  Replace requirements to display run IDs and paths with scientific provenance.
- **1.0.1** — Require minimal, source-preserving edits when bringing existing
  writing into conformance. Retain the restriction of Methods to scientific
  procedures and evidence limitations, excluding repository, storage, and
  publication details. At the author's request, this version replaces the
  provisional, unpublished **2.0.0** label.
- **1.0.0** — Name and version the existing Writing Guide; writing requirements
  remain unchanged.

## 2. Location and naming

- Store each entry directly as `writings/expXXX.typ`, where `XXX` is the
  experiment's zero-padded three-digit identifier.
- Use the same identifier for related experiment code and artifact paths when
  they exist.

## 3. Global writing rules

These rules apply to every experiment article, including titles, prose,
captions, alternative text, tables, figures, and appendices.

1. Explain the science independently of the repository: model, data,
   interventions, numerical methods, training, selection criteria, measurements,
   and aggregation. Express scientifically relevant implementation choices as
   algorithms, numerical settings, or measurement definitions.
2. Exclude internal references: local paths, filenames, run IDs, schema fields,
   import histories, storage layouts, build commands, and publication mechanics.
   Use descriptive scientific names rather than bare internal identifiers.
   Reproducibility does not justify repository bookkeeping in the article;
   technical provenance belongs in run records and separate documentation.
3. Preserve scientific provenance and limitations without narrating file
   handling. Distinguish reused observations from new measurements and identify
   datasets, experimental conditions, and model-selection criteria scientifically.
   Literature citations and meaningful references to published scientific work,
   figures, equations, and sections remain appropriate.
4. This boundary concerns rendered content, not necessary source-code plumbing.
   Typst imports, data lookups, citation keys, and link targets may use internal
   identifiers without displaying them to readers. The technical instructions
   in this guide describe authoring machinery; they are not article content.

For example, write “Accuracy was evaluated at the validation-selected epoch;
population firing rates were measured at the final epoch.” Keep the selection
criterion and measurement definitions, but omit checkpoint filenames, JSON field
names, and the commands used to retrieve them. A numerical integration timestep
belongs in the scientific account; the directory containing the simulation does
not. Moving internal references into a caption or appendix does not satisfy
these rules.

### 3.1. Editing existing writing

Existing `.typ` files are authored documents that may contain manual revisions,
not disposable output to regenerate from this guide.

1. Read the current file immediately before editing and use it as the baseline,
   including uncommitted manual edits. Do not reconstruct it from an older
   revision, generated output, or a remembered draft.
2. Treat a conformance request as a minimal edit, not permission to rewrite.
   Stay close to the source's scientific substance, emphasis, terminology, and
   wording. Preserve its arguments, procedural details, equations, parameter
   values, caveats, citations, specification sheets, and plots unless the
   requested change requires a specific adjustment.
3. Limit edits to the requested section and strictly necessary dependent
   changes, such as contents links or citation numbering. Do not revise the
   title, abstract, results, or other sections merely for stylistic consistency.
4. Apply the requested guide version, not silently the latest one. If that
   version cannot be recovered, ask which available version to use. Explicit
   user instructions control the scope and any exceptions to the guide.
5. Prefer small changes to organization and phrasing over replacement prose.
   Conformance alone does not authorize new scientific claims, interpretations,
   results, literature surveys, or invented missing methods. If a requirement
   cannot be met without materially changing or removing scientific substance,
   identify the conflict and ask before doing so. Source preservation does not
   establish a claim's correctness or override evidence requirements.
6. Review the diff against the live starting text: every change must serve the
   requested conformance, with unrelated manual edits preserved. Report any
   unresolved conformance gaps rather than hiding them with new content.

For example, “update the Methods in exp022 to conform with Writing Guide
1.0.0” means adapt the existing Methods to that version's applicable rules.
Keep its scientific account and distinctive wording wherever possible; do not
replace it with a generic Methods section or rewrite the rest of the article.

### 3.2. Improving the guide through use

While applying this guide, identify concrete opportunities to improve its
reusable instructions. Use difficulties encountered in the task, weaknesses in
the resulting writing, and the author's corrections as evidence.

- Suggest improvements only when they would materially improve future writing;
  do not manufacture a suggestion for every task.
- Distinguish general lessons from experiment-specific choices. Check existing
  rules and prefer refining them over adding duplicates.
- For each suggestion, briefly state the observed problem, propose exact wording
  and its location, and explain the expected benefit.
- Complete the requested work first unless a conflict requires
  clarification. Present suggestions in the task response, never in the
  experiment article.
- Do not modify the guide or broaden article edits without explicit approval.
  Approved guide changes follow its versioning rules.

### 3.3. Resampling existing writing

Before resampling existing writing, read the current target and identify the
scientific content, author decisions, and explicit constraints that must survive.
Treat edits made directly or through an agent equally. Regenerate only the
requested target; do not restore superseded wording or undo settled decisions.
Present the candidate for review unless saving or replacement is explicitly
authorized. If a material choice cannot be recovered confidently, ask rather
than silently discarding it.

### 3.4. Authored update dates

Apply this rule to future article edits; existing articles do not need a date
change merely to adopt this guide.

- Set `meta.updated_at` to the author's local calendar date (`YYYY-MM-DD`) when
  completing and saving a substantive revision. Qualifying changes include
  revised claims, methods, results, figures, interpretations, meaningful
  corrections, or explanations that materially improve scientific understanding.
  A small correction to a consequential number or equation still qualifies.
- Use completed editing passes as checkpoints, with calendar-day granularity:
  multiple revisions on the same day share one date. Do not wait for a weekly
  interval, commit, build, or publication, and do not advance the date merely
  because time has passed. Unsaved review candidates do not change the article's
  metadata.
- Leave `updated_at` unchanged for spelling, punctuation, formatting, link
  repairs, or source plumbing that do not change scientific meaning, and for
  rebuilds or deployments alone.
- Completing a compute, analyse, or present run does not by itself advance the
  article's date. When a present run is incorporated into the article, set
  `meta.updated_at` to the date of incorporation if it substantively changes the
  displayed evidence, figures, or interpolated values—even when the article's
  prose and other Typst source remain unchanged. Switching to a run with
  substantively equivalent content does not qualify.
- Preserve `meta.created_at` (or the existing legacy `meta.date`). An update
  must not predate creation or move an existing update date backwards; flag a
  conflicting date rather than silently rewriting it. A new article may omit
  `updated_at` until its first substantive revision.
- Dates are explicitly authored metadata. Never infer them from Git,
  filesystem timestamps, run dates, builds, or deployment dates. Do not backfill
  unknown historical updates. Updating this field is a necessary dependent
  edit under section 3.1 when the requested revision qualifies, unless the
  author explicitly instructs otherwise.

### 3.5. Local-data availability

Every `writings/expXXX.typ` must declare one `meta.status` using an exact label
from the table below. The entire value must be `[≡ TXT]` or `[▦ DATA]`, including
brackets, with no descriptive suffix or surrounding whitespace. Demolab displays
this authored string; do not add a second status field or a separate icon field.

| Label | Local-data availability |
| --- | --- |
| `[≡ TXT]` | Article only: no usable, validated local presentation data is available for any declared article input, or the article declares no data inputs. |
| `[▦ DATA]` | Local data: usable, validated local presentation data is available for at least one declared article input, including reused upstream results. |

The badge reports availability in the working checkout at the last agent check,
not a live web-UI measurement. `[≡ TXT]` does not mean literally text without
diagrams. `[▦ DATA]` does not certify complete input coverage, successful rendering,
scientific quality, review or completion. Null and negative findings qualify
equally. For comparisons with only some inputs available, use `[▦ DATA]` and report
the missing inputs in the task summary, not new article prose.

- Agents own freshness. At the end of an authorized article revision, relevant
  implementation or execution task, or change to local data availability,
  reassess the affected article and all articles whose declared inputs depend
  on the affected data keys, including comparisons and syntheses. Update the
  badge in either direction when the evidence changes. This is a necessary
  dependent metadata edit under section 3.1; explicit author scope restrictions
  still take precedence.
- Read the current article's `inputs` and article-scoped bindings, and check
  their agreement with the publishing configuration. Run read-only
  `uv run pingstore discover` against the configured local source. Match the
  declared keys to discovery's authoritative `experiment` fields, not run-name
  substrings or the article ID alone. No inputs means `[≡ TXT]`; otherwise at
  least one qualifying input means `[▦ DATA]`. Availability need not mean that
  this run is currently selected for publication.
- Qualifying data comes from a completed, nonempty v3 present run validated
  under the Storage Guide, including layout, payload checksums and applicable
  input-provenance checks. Numbers, tables, figures and videos can qualify;
  image-only presentations need not have `numbers.json`. Code, remote jobs,
  compute/analyse-only runs, hidden incomplete runs, bookkeeping-only exports,
  prose claims or standalone illustrative diagrams do not establish `[▦ DATA]`.
- A failed discovery, inaccessible source or invalid provenance is an unresolved
  check, not an empty result. Do not guess or silently downgrade on that basis;
  preserve the existing badge and report the blocker. A successful check showing
  no matching local data does warrant `[≡ TXT]`, even if remote results exist.
- Maintain the literal string in source, not through a build-time calculation,
  scheduler callback or background monitor. Normal tests enforce the vocabulary
  without requiring another checkout or CI to contain the author's local data.
- For migration from the retired milestone labels, classify from current
  declared inputs and validated local data, not from the previous label. Change
  only status lines and necessary policy/tests; preserve scientific prose,
  dates and unrelated edits. Status-only changes do not advance `updated_at`;
  apply section 3.4 only when the underlying change qualifies independently.
- To migrate version 10.0.0 or 11.0.0 badges, remove any descriptive suffix
  and add the corresponding icon inside the brackets, preserving the recorded
  classification. Any separate availability reassessment follows the validation
  rules above; changing badge formatting alone is not reclassification.
- A status check authorizes no execution, input selection, materialization,
  publication, historical inspection, migration or mutation of stored runs.

## 4. Titles

For the experiment's overall title (`meta.title`), use a short, plain-English
phrase naming its main finding or controlled comparison. Prefer a specific
relationship ("Firing Rate Tracks Gamma Frequency") over vague topics or
promotional claims. State a finding only when supported by results; otherwise
name what is being tested. Aim for 5–10 words, retaining technical terms needed
for precision.

### 4.1. Table of Contents

Every `writings/expXXX.typ`, including reference pages and figure galleries,
must render exactly one `Table of Contents` at the beginning of its body, after
the title/metadata and before `Abstract`. Use the shared `contents.typ` helper:

```typst
#import "contents.typ": with-contents
// Define the article body and apply any dataset/report wrappers first.
#let body = with-contents(body)
```

- Apply this as the final body wrapper, outside data-readiness branches, so
  navigation is present in both populated and unavailable-data views.
- Generate linked entries from the current article's rendered level-2 (`==`)
  section headings, in document order, including Abstract, Datasets, appendices
  and References when present. Keep the list compact: omit deeper subsections,
  the title, the TOC itself, and headings belonging to other articles in a book.
  Links must work in HTML and PDF; do not maintain a manual link list or use an
  unscoped, document-wide outline.
- If an existing reference page or gallery has no Abstract, place the TOC before
  its first content instead. This navigation requirement does not authorize
  inventing an abstract or changing scientific prose. Unavailable-data views
  list only the sections they actually render, never unavailable results.
- Migrate existing entries by removing their old contents heading/list and
  adding the shared import and final wrapper. Preserve all other authored
  content, metadata and unrelated edits. TOC-only changes do not advance
  `meta.updated_at` (section 3.4); reassess availability under section 3.5.
- Validate every entry's wrapper and check rendered ordering, article scope,
  and link targets. Include an unavailable-data view and a reference page in
  the checks.

### 4.2. Unnumbered headings and section order

- Never number article section or subsection headings, whether manually or
  through Typst's `heading.numbering`. This applies at every depth, including
  Abstract, Results, Methods, appendices and References. Use `Rhythm frequency`,
  not `1. Rhythm frequency` or `2.1 Rhythm frequency`; use `Appendix: Training
  settings`, not `Appendix A: Training settings`. The Writing Guide's own
  numbered policy sections are not article headings.
- Preserve numbers that identify scientific content rather than section order,
  such as a `4D model` or a training condition's name. Figure, equation, citation,
  reference-list and method-step numbering are unaffected.
- In every article containing both sections, place the entire Results section,
  including its subsections, before the entire Methods section. Do not move
  only the heading or leave results figures beneath Methods. Existing reference
  pages without these sections do not need invented sections.
- Use descriptive section names with links for internal cross-references, not
  section numbers. When migrating, remove ordinal prefixes, repair links and
  textual references, and move complete sections as needed. Preserve scientific
  content, equation labels, citations, metadata and unrelated edits. Formatting
  and ordering changes alone do not advance authored dates (section 3.4).
- The shared article wrapper disables automatic heading numbering. Regression
  tests must check source headings and Results-before-Methods order in every
  article, including data-dependent bodies, plus rendered headings and links.

## 5. Abstracts

Write a standalone summary of roughly 60–120 words that lets a reader browsing
experiments quickly understand what was done, what happened, and why the result
is useful.

### Ground the abstract before writing

- Read `experiments/expXXX.py` and follow its execution into the relevant recipe,
  compute, analysis and presentation modules and helpers. Do not draft from the
  existing article alone.
- Establish the experiment’s actual purpose, model, task, dataset, comparisons,
  measurements and reusable outputs.
- Check completed-run provenance, retained configurations and results. Code
  describes intended behaviour; retained evidence establishes what actually
  happened. Resolve differences between current code and historical execution
  before making claims.
- Distinguish newly executed work from reused training, measurements or figures.
  Do not run experiments merely to generate an abstract.

### Write for rapid understanding

- Lead with the principal finding or concrete output, according to the
  experiment’s purpose. A training-bank experiment should foreground the bank;
  a hypothesis test should foreground its result.
- State what was done, including the comparison and the scale needed to
  interpret it.
- Report the decisive result with useful quantities, units and measurement
  context. Say how outcomes differed, not merely that differences existed.
- End with the supported conclusion or reuse value. Include limitations that
  materially change interpretation, without adding a generic disclaimer.
- Use plain, direct language, consistent terminology and one main idea per
  sentence. Write researcher-to-colleague prose. Retain necessary scientific
  terms and define unfamiliar terms or notation.
- Use past tense for completed work and future tense for planned work. Never
  substitute an expected outcome for an observation.
- Omit general background, citations, repository bookkeeping and unnecessary
  implementation details.
- Interpolate reported values from retained evidence where possible; do not
  hardcode them into article prose.

### Example: exp022

The values below illustrate the verified result; the article should interpolate
them from its selected evidence.

> We assembled a reusable bank of 102 spiking networks for MNIST handwritten-digit
> classification, covering 34 conditions with three random seeds each. Training
> lasted 50 epochs per network. Conditions compared feedforward controls with
> excitatory–inhibitory recurrent networks and varied activity penalties,
> inhibitory decay, numerical timestep, recurrent initialization and trainability,
> and input drive. In the baseline comparison, mean validation accuracy was 96.0%
> for feedforward networks and 91.5% for recurrent networks. Their final-epoch
> excitatory firing rates were 247.5 and 24.4 Hz, respectively. Retained models and
> learning histories support subsequent experiments; these training-recipe
> comparisons do not isolate a causal benefit of gamma timing.

## 6. Data access in Typst source

For staged experiments, the writing consumes a selected presentation run, not
compute checkpoints or live analysis. Pingstore resolves that input to the flat
`export/` of a validated v3 present run. V2 presentations are not permitted for
preview or publication. Existing readers that accept them are nonconforming,
not a compatibility exception; this guide does not authorize their migration. Keep
storage-version paths out of the writing: discovery supplies the resolved
directory, while `run.json` remains authoritative. Import `data-file` from the
local `run-inputs.typ` helper and bind it to the article. It reads only explicit
Demolab inputs, with no implicit filesystem fallback. Preview supplies selected
runs; publication can supply a fixed `demolab-data-inputs` inventory with an
engine that supports `build.sources`. Without an input for the article, show the
shared unavailable-data notice rather than evaluating report calculations:

```typst
#import "/.demolab/lib.typ": data-json, data-image
#import "run-inputs.typ": data-file, inputs-ready, pending-report
#let data-file = data-file.with(article: "exp022")
#let inputs = ("exp022",)
#let render-report(data-file) = [
  #let results = data-json(data-file("exp022/numbers.json"))
  // Interpret results and render figures here, after checking input availability.
]
#let body = if inputs-ready(data-file, inputs) {
  render-report(data-file)
} else {
  pending-report(data-file, inputs, [], ())
}
```

Declare every required logical data key in `inputs`, including upstream keys in
galleries and comparisons. The readiness check tests selected directories, not
the existence of `numbers.json`, so image-only runs work too. Keep data reads,
calculations, and result-dependent content inside `render-report`. A selected
run with a missing or corrupt file is an error, never an empty report. Builds
do not choose Latest, read browser selections, or execute experiment stages.

Editing prose does not require a new run. Scientific or presentation changes use
the independent stages in [the execution guide](../experiments/README.md).
Keep imported historical figures distinct from newly generated figures; run.json
records their source lineage, and captions must not imply a new simulation.

## 7. Results

Stub the Results section with unnumbered headings, figures and concise captions
only. Do not fill it with narrative prose.

1. Name the section exactly `Results` and place it before `Methods`.
   Do not append a tagline, description or other suffix, or add a
   section-number prefix (section 4.2).
2. Use unnumbered subsection headings such as `Rhythm frequency`, not
   `Plot 1`. Name the comparison or supported finding plainly.
3. Select only the key plots needed to show the experiment's results. There is
   no fixed plot count; a subsection may contain more than one related figure.
4. Use figures from retained experimental outputs. Keep captions concise and
   identify the measurement, conditions, aggregation and uncertainty needed to
   read each figure correctly. Distinguish illustrative probes and reused
   observations from new measurements.
5. An optional theory diagram may precede a data plot when useful. Its caption
   must identify it as an expectation or mechanism, not experimental evidence.
6. Do not add an introductory paragraph, prose before or after figures,
   expectation-versus-result commentary, or a concluding summary. Do not add
   prose placeholders asking a later pass to fill these in.

When migrating existing headings, remove their taglines and repair any authored
links to former anchors. The shared TOC picks up the plain heading automatically.
Preserve scientific content and authored dates; apply section 3.5 to availability.

Illustrative scaffold (replace the figure paths and captions with retained
outputs and their measurement details):

```typst
== Results

=== Rhythm frequency

#figure(
  data-image(data-file("expXXX/frequency-vs-decay.svg"), width: 100%),
  caption: [Gamma frequency across inhibitory decay times. Points show means
  across seeds; error bars show ±1 standard deviation.],
)

=== Population firing rates

#figure(
  data-image(data-file("expXXX/rate-vs-decay.svg"), width: 100%),
  caption: [Excitatory and inhibitory firing rates across conditions. Points
  show means across seeds; error bars show ±1 standard deviation.],
)

=== Classification accuracy

#figure(
  data-image(data-file("expXXX/accuracy-vs-decay.svg"), width: 100%),
  caption: [Test accuracy across inhibitory decay times. Points show means
  across seeds; error bars show ±1 standard deviation.],
)
```

## 8. Methods

Explain how the experiment was actually performed and how its reported
measurements and reusable outputs were obtained. Write for a
computational-neuroscience colleague who understands the field but does not know
this experiment. Aim for 300–450 words, excluding displayed equations but
including symbol definitions. This is a guide, not a quota: use fewer words for
simple experiments and do not pad the account. Exceed this only when scientific
completeness requires it.

### Ground the account before writing

- Read the experiment’s execution code, scientific definitions, analysis and
  relevant helpers. Do not draft from the existing article alone.
- Check completed-run provenance, retained configurations and outputs. Resolve
  differences between current code and historical execution before making claims.
- Outline the complete scientific procedure before writing prose. Distinguish
  newly executed work, reused evidence and planned work.
- Do not run experiments merely to write Methods.

### Write the procedure

1. Name the section `Methods` and place it after `Results`.
2. Begin with a short orientation explaining the experimental approach.
3. Use one flat numbered list containing only the substantive scientific
   operations needed to explain the experiment, with at most ten items. There is
   no minimum item count. Do not add steps to reach a target length. Do not use
   nested lists or subsection headings. Derive the operations from the experiment
   rather than imposing a fixed template.
4. Follow the actual dependencies: starting data and models, controlled changes,
   execution, selection and measurement, including substantive analysis where
   applicable. Do not stop at training when the report also contains evaluation
   or analysis. Cover the essential scientific procedure without creating a
   separate step for every output or routine operation.
5. Give each item a short action-led label and two to four concise sentences.
   Begin with what was done, then give the essential settings and what the
   operation produced. Equation-bearing items may include a compact definition
   paragraph. Explain consequential choices without inventing retrospective
   justifications.
6. Make the main account explain the complete procedure and its consequential
   choices. Put exhaustive parameter grids, initialization distributions and
   derivations in appendices. Do not defer essential model differences,
   selection criteria or measurement definitions.
7. Select the key equations before drafting: those defining the experiment's
   central model, intervention or measurement, usually one to three, not a quota.
   Number each equation and place it beside the operation it explains. Describe
   routine operations in words unless their mathematical form matters to the
   comparison. Define every symbol once, give units where applicable, and reuse
   notation consistently. Cite established methods where appropriate.
8. Explain how reported measurements were obtained, including relevant data
   partitions, model-selection criteria, measurement timing, repetitions and
   aggregation. Distinguish illustrative probes from population estimates and
   reused observations from new measurements. Do not create standalone Methods
   items for illustrative raster inspection, plotting, routine averaging across
   seeds, error-bar construction, or displaying retained training trajectories.
   Put necessary figure-reading details in concise captions; integrate
   consequential sampling or measurement choices into the relevant scientific
   step. A dedicated analysis step is appropriate when it defines a substantive
   estimator, statistical test, or analysis central to the experiment's question.
   Routine presentation and summarization alone do not qualify.
9. Use direct, concrete prose. Exclude repository bookkeeping, implementation
   narration and result interpretation. Finish with a compression pass: remove
   repeated definitions, textbook exposition, procedural signposting and details
   already supplied elsewhere. Preserve what was varied, what was held fixed,
   what was trained, and how outputs were selected and measured.

### Completion check

Can the reader recover the substantive procedure from the numbered labels and
understand each key equation locally? Together, Methods and concise figure
captions must make the source and meaning of reported measurements and reusable
outputs clear, without a separate Methods item for every output or routine
operation. The main account must explain the scientific procedure without
requiring code or reconstruction from appendices. Remove items that merely
narrate presentation or repeat captions. Flag missing evidence rather than
inventing a step.

For exp022, the applicable sequence is: data and splits; networks and controlled
conditions; encoding and simulation; class scores; training and optimization;
validation and model selection; measurements and reusable outputs. Its key
equations define the readouts and the sample-wise activity penalty; routine
Poisson and softmax expansions need not appear in the main account.

## 9. References

Apply these rules to every experiment entry:

1. Place an unnumbered `References` heading at the bottom of the entry, after
   any appendices. Number its reference-list entries, not the section heading.
2. Use `#cite(...)` for inline citations and `#reference-list(...)` for the
   reference list.
3. List sources in order of first citation, with authors, title, publication
   venue, year, and a DOI or stable URL where available.
4. Reuse the same number for repeated citations. Keep citation numbers
   synchronized with list positions.
5. Include only cited sources and verify that each supports its associated
   claim. Keep literature references distinct from upstream experiment and run
   provenance.
