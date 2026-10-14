---
title: Writing Rules
author: Eoin Murray
collection: notes
description: Scientific writing, evidence, terminology and editorial conventions for experiment articles.
created_at: "2026-10-14"
updated_at: "2026-10-14"
---

# Writing Guide

Version: **36.0.0**

## Agent responsibilities

The `experiments/expXXX/report/report.mdx` files are collectively known as "writings", or
"the writings".

Do not create automated tests for the writings.

Agents must maintain each article's separate `txt` or `data`,
`vX.Y.Z`, and optional author-assigned `reviewed` tags after
relevant article, implementation, execution, Writing Guide or local-data
changes, including dependent comparisons and syntheses. `X.Y.Z` records the
latest Writing Guide version applied and must remain current under Writing Guide
section 3.4. Only the author may add or remove `reviewed`. Follow that section
for validation and uncertainty handling; metadata-only edits do not change
dates.

## Astro relocation — 13 October 2026

This note is now the canonical Writing Guide. Version 36.0.0 and its
scientific and editorial requirements are preserved. Current authored entries
live at `experiments/expXXX/report/report.mdx`, with optional `parameters.mdx`,
`appendix.mdx`, and `references.mdx`; metadata lives in MDX frontmatter.
Supporting article files live in the same `report/` directory; experiment-specific
browser JavaScript lives beside it in `experiments/expXXX/js/`. This source
relocation preserves guide version 36.0.0, article tags, and authored dates.

The original guide below is retained verbatim for now. References to Typst
paths, Demolab tags and template APIs describe the former authoring system;
apply their scientific and editorial meaning to Astro reports rather than
reintroducing that system. Component-specific writing rules are preserved in
section 5, independently of the old executable templates. Current rendering
APIs are defined by `site/src/components/`, `site/src/layouts/`, and
`site/src/content.config.ts`. Relocation changes no scientific-writing rule or
author-assigned review decision.

The Writing Guide defines the conventions for Pinglab's published experiment
entries in `writings/expXXX.typ`. This file is the canonical source for shared
scientific-writing and editorial rules. It does not define the articles'
document schema or the APIs of their Typst templates; those contracts live with
the templates under `writings/templates/`.

## 1. Versioning

Version this guide independently of Pinglab and Demolab. Increment the major
version when changed requirements make previously compliant writing require
revision, the minor version for compatible additions, and the patch version for
corrections or clarifications that do not change requirements. Update the version
above and add a short entry to the version history when changing the guide.

### 1.1. Version history

- **36.0.0** — Standardize every cross-link between experiment articles as the
  separately linked experiment identifier and italicized article title, divided
  by an unlinked em dash.

- **35.4.0** — Make template comments authoritative for component-specific
  writing as well as schema, add an explicit template-routing table, and remove
  duplicated Abstract, Results, Methods, References, Dataset and equation rules
  from the shared guide.

- **35.3.0** — Split Results cards, article-scoped contents and displayed-equation
  numbering into their own template files, leaving `article-layout.typ` as the
  article-shell orchestrator, and migrate every writing to the owning imports.

- **35.2.0** — Add a template-owned References component and a single article
  shell that composes Dataset insertion, equation numbering and article-scoped
  contents without duplicating that ordering in every writing.

- **35.1.0** — Devolve document-schema and Typst-template contracts to the
  comments at the top of each file in `writings/templates/`. Keep this guide
  focused on shared scientific-writing, evidence, provenance and editorial
  rules, and distinguish guide-version conformance from template conformance.

- **35.0.0** — Replace the combined article status string with separate
  Demolab tags for local-data availability, author review and the latest
  Writing Guide version applied. Use the concise dotted `vX.Y.Z` tag and
  preserve recorded availability classifications and review decisions.

- **34.0.1** — Correct the publication-geometry rule: constrain final physical
  size and legibility, not the overall aspect ratio of compound figures, and
  require balanced scientific grouping without stretching plots.

- **34.0.0** — Standardize publication plot aspect ratios, physical figure
  widths and final-size legibility checks while preserving scientifically
  meaningful axis geometry.

- **33.0.0** — Place an article's Abstract before its Table of Contents so
  readers encounter the experiment summary before its navigation. Keep the
  Table of Contents before the first section on entries without an Abstract.

- **32.0.0** — Require every distinct panel in a multi-panel scientific figure
  to carry a consistent uppercase letter and require captions and prose to use
  those labels instead of positional references.

- **31.2.0** — Add the explicit author-assigned `◉ REVIEWED` article status.
  Keep review distinct from inferred local-data availability and record the
  applied Writing Guide version separately for reviewed articles.

- **31.1.0** — Remove the Writing Guide migration and grounded-regeneration
  procedure categories. Retain source-preserving editing, evidence checks and
  validation as direct requirements for ordinary article work.

- **31.0.0** — Require every displayed equation in every article section to
  carry a continuous article-wide number in every published target, and add a
  shared wrapper plus regression checks that enforce the requirement.

- **30.0.0** — Require agents to classify article work before editing, build a
  version-delta compliance record for migrations, distinguish targeted from
  complete grounded regeneration, apply provenance vocabulary throughout the
  article, complete a semantic claim audit and report the evidence of
  compliance. Align current article inputs with validated v4 present runs.

- **29.1.0** — Define a six-step grounded article-regeneration workflow that
  preserves important authored content while rebuilding the complete article
  from code, retained evidence and the current Writing Guide, and requires an
  enumerated report of potentially important cuts.

- **29.0.0** — Require every Results subsection to use a context-free visual
  card, and require every Methods section to organize continuously numbered
  scientific operations beneath the exact `Compute`, `Analyse` and `Present`
  groups. Presentation remains an evidence-mapping account rather than
  repository or rendering narration.

- **28.1.0** — Allow context-free Results cards with concise local context and
  optional notes, and allow longer Methods to use continuously numbered groups
  of short, single-purpose items while preserving substantive-procedure,
  equation and numbering safeguards.

- **28.0.0** — Require Writing Guide migrations to preserve authored update
  dates, including when conformance changes scientific meaning. Only a separate,
  independently requested substantive revision may advance `meta.updated_at`.

- **27.0.0** — Add the applied Writing Guide version to every article status
  badge and require agents to keep that version synchronized with this guide.

- **26.0.1** — Clarify that saved article text is canonical authored source
  regardless of how it originated, and prohibit guide migrations from
  regenerating existing sections.

- **26.0.0** — Adopt the conventional authorial first-person plural for the
  article's actions, decisions, interpretations and plans.

- **25.0.0** — Replace bullet-point abstracts with paragraph-form abstracts of
  around 65 words.

- **24.0.0** — Replace prose abstracts with three- or four-bullet experiment
  stories and move numerical results, scales and parameter values to Results.

- **23.0.0** — Number direct Results subsections, include them beneath Results
  in the article Table of Contents, and require each subsection to describe and
  contain exactly one figure.

- **22.0.0** — Distinguish evidence preserved from an earlier execution from
  measurements recorded for reuse and contents shown in the current figure.

- **21.0.0** — Require the `Dataset` section to appear after all main-text
  sections and before any appendices and References, while remaining in the
  Table of Contents.

- **20.0.0** — Give captions and figure-local Results prose distinct roles.
  Captions explain how to read a figure; prose states what the figure establishes.
  Prohibit repeating the same finding in both places and add a deletion test and
  worked example.

- **19.1.0** — Allow optional figure-local explanatory prose in Results when it
  states an evidence-grounded expectation, explains an observed pattern, or
  distinguishes observation from interpretation. Keep Results concise and
  evidence-led without requiring prose for every figure.

- **19.0.0** — Establish shared terminology and mathematical notation for
  recurrent entities across the experiment collection, including identities,
  time, populations, activity, readouts, training, statistics, rhythms,
  connectivity and dynamics. Require experiment-local definitions while
  prohibiting ambiguous reuse of recurring symbols and scientific terms.

- **18.0.0** — Specify first-person singular for the sole author's actions and
  interpretations, and distinguish past experimental work from present
  definitions, figure descriptions and conclusions, and explicitly planned work.
  Apply these rules throughout articles without changing scientific substance.

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

### 2.1. Template-owned components

Rules specific to one article component belong in that component's top-of-file
comment under `writings/templates/`. This includes its writing requirements,
public entry points, arguments, structure, ordering, cardinality, rendering and
division of responsibility with article-authored content. Prefer extending the
owning template contract over adding component-specific rules here.

Before creating or editing a component, read its complete contract:

| Template             | Authority                                                                   |
| -------------------- | --------------------------------------------------------------------------- |
| `article-layout.typ` | Mandatory final article shell and cross-component ordering.                 |
| `abstract.typ`       | Abstract grounding, prose roles, heading and contents marker.               |
| `contents.typ`       | Article-scoped contents placement, scope and rendering.                     |
| `dataset.typ`        | Evidence inputs, readiness, unavailable-data behaviour and Dataset section. |
| `equations.typ`      | Displayed-equation numbering and target-specific rendering.                 |
| `methods.typ`        | Methods grounding, prose, stages, operations and headings.                  |
| `references.typ`     | Source requirements, References placement and list rendering.               |
| `result-card.typ`    | Results prose, figures, cards, numbering and presentation.                  |

Title guidance remains in section 4 because `meta.title` is article-owned
metadata rather than output of a shared title component.

This guide is authoritative only for rules that apply across components:
repository-independent scientific explanation, general evidential strength and
provenance, source-preserving editing, authored metadata, tense and person, and
shared terminology and notation. Template conformance does not establish
compliance with those global rules. An article's `vX.Y.Z` tag records only the
Writing Guide version applied; template contracts are versioned with the
repository rather than with separate article tags.

If ownership is unclear, put a rule in the narrowest template whose complete
output it governs. If a template contract and a global guide rule appear to
conflict, report the conflict rather than silently choosing one.

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

Before editing, state the requested target and any permitted dependent edits.
Read the current Writing Guide in full before applying it; excerpts, searches,
prior familiarity and successful tests are not substitutes for that read.

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
7. Treat saved text as authored source regardless of origin. Once content has
   been saved in an existing `.typ` file, treat the live text as canonical
   authored content whether it was initially written by the author, drafted by
   an agent, or generated from experiment materials. A conformance edit must
   transform this live source; it must not regenerate or replace the target
   section from experiment code, a template, or a fresh model draft. Apply only
   the requested changes and their necessary dependencies, and preserve all
   compatible wording and scientific content. If compliance requires wholesale
   replacement or an uncertain substantive change, stop and present the
   conflict for explicit authorization.

#### Completion gate for article edits

Tests, compilation and rendering are necessary checks, but they do not establish
scientific or semantic compliance. Before completing any article edit, audit
every changed claim for evidential support, causal strength, execution status,
provenance vocabulary, numerical accuracy and agreement with displayed
evidence.

The final task report must state the target Writing Guide version, edited scope,
validation performed and unresolved gaps. A blocked requirement remains
explicit; it must not be hidden by passing tests or fluent prose.

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

### 3.3. Authored update dates

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
  edit under section 3.1 when a requested revision qualifies, unless the author
  explicitly instructs otherwise.

### 3.4. Article tags and local-data availability

Every `writings/expXXX.typ` must declare one `meta.tags` list containing the
following separate Demolab tag slugs:

| Tag             | Cardinality | Meaning                                                                                             |
| --------------- | ----------- | --------------------------------------------------------------------------------------------------- |
| `txt` or `data` | Exactly one | Local presentation-data availability.                                                               |
| `vX.Y.Z`        | Exactly one | Latest Writing Guide version applied to the article.                                                |
| `reviewed`      | Zero or one | The author explicitly reviewed and accepted the article in its current scientific and written form. |

Articles may add other Demolab subject or method tags. Do not use `meta.status`
or `meta.writing_guide`; availability, review and guide version are independent
tags. Keep tags unique and valid under Demolab's lowercase-slug syntax.

The `txt` tag means no usable, validated local presentation data is available
for any declared article input, or the article declares no data inputs. It does
not mean literally text without diagrams. The `data` tag means usable,
validated local presentation data is available for at least one declared
article input, including reused upstream results. It does not certify complete
input coverage, successful rendering, scientific quality, review or completion.
Null and negative findings qualify equally. For comparisons with only some
inputs available, use `data` and report the missing inputs in the task summary,
not new article prose.

The `reviewed` tag is author-assigned. Agents must never infer it from tests,
data, rendering or their own review, and only an explicit author request may add
or remove it. It coexists with `txt` or `data`, so review never hides the
underlying availability distinction. Reassess local data normally while
preserving `reviewed` unless the author explicitly requests its removal.

- Agents own availability freshness. At the end of an authorized article
  revision, relevant
  implementation or execution task, or change to local data availability,
  reassess the affected article and all articles whose declared inputs depend
  on the affected data keys, including comparisons and syntheses. Update the
  availability tag in either direction when the evidence changes, except that
  an author-assigned `reviewed` tag is preserved and the change is
  reported. This is a necessary dependent metadata edit under section 3.1;
  explicit author scope restrictions still take precedence.
- Agents also own Writing Guide version freshness. Whenever this guide is
  applied to an article, set its `vX.Y.Z` tag to the exact version applied.
  Whenever the guide version changes, update every article brought into
  conformance in the same editing pass; never advance an article's recorded
  version without applying all requirements introduced through that version.
  Normal tests must reject guide-version tags that differ from the current guide
  version once the repository declares all articles current. A version-only
  synchronization does not advance `meta.updated_at`.
- Read the current article's `inputs` and article-scoped bindings, and check
  their agreement with the publishing configuration. Run read-only
  `uv run pingstore discover` against the configured local source. Match the
  declared keys to discovery's authoritative `experiment` fields, not run-name
  substrings or the article ID alone. No inputs means `txt`; otherwise at least
  one qualifying input means `data`. Availability need not mean that
  this run is currently selected for publication.
- Qualifying data comes from a completed, nonempty v4 present run validated
  under Pingstore Rules, including layout, payload checksums and applicable
  input-provenance checks. Numbers, tables, figures and videos can qualify;
  image-only presentations need not have `numbers.json`. Code, remote jobs,
  compute/analyse-only runs, hidden incomplete runs, bookkeeping-only exports,
  prose claims or standalone illustrative diagrams do not establish the
  `data` classification.
- A failed discovery, inaccessible source or invalid provenance is an unresolved
  check, not an empty result. Do not guess or silently downgrade on that basis;
  preserve the existing availability tag and report the blocker. A successful
  check showing no matching local data does warrant `txt`, even if remote
  results exist.
- Maintain the literal tags in source, not through a build-time calculation,
  scheduler callback or background monitor. Normal tests enforce the vocabulary
  without requiring another checkout or CI to contain the author's local data.
- Migrate the former combined labels by carrying forward their availability
  and review classifications into separate tags. `[≡ TXT | vX.Y.Z]` becomes
  `txt` plus the current guide-version tag; `[▦ DATA | vX.Y.Z]` becomes `data`
  plus that tag; and `◉ REVIEWED` plus `writing_guide: "X.Y.Z"` becomes
  `reviewed`, the current guide-version tag and the separately reassessed
  availability tag. Change only metadata and necessary policy/tests; preserve
  scientific prose, dates and unrelated edits. Metadata-only changes do not
  advance `updated_at`; apply section 3.3 only when the underlying change
  qualifies independently.
- An availability check authorizes no execution, input selection, materialization,
  publication, historical inspection, migration or mutation of stored runs.

### 3.5. Tense and grammatical person

These are polished scientific articles, not diary entries. Apply the following
rules throughout rendered content, including abstracts, Methods, captions and
appendices; choose tense by the sentence's function, not by section alone.

- Use first-person plural (`we`, `our`, `us`) for the article's actions,
  decisions and interpretations: “We trained the networks”; “We interpret this
  as…”. This conventional authorial `we` does not by itself imply multiple human
  researchers. Attribute collaborators and reused work explicitly; changing
  person must not turn reuse into a claim of new execution.
- Do not force a pronoun into every sentence. Prefer the scientific subject
  when natural (“Accuracy increased”) and allow passive voice when the actor
  is unimportant (“Weights were held fixed”). In mathematical exposition,
  prefer “Substituting gives…” or “Consider…” to an ambiguous authorial `we`.
- Use past tense for completed procedures, choices and observations, including
  reused experiments: “We evaluated three seeds”; “Mean accuracy was 92%”.
  Do not narrate completed Methods as instructions or as work happening now.
- Use present tense for model and algorithm definitions, mathematical
  relationships, established knowledge, the article's scope, and what a figure
  or table displays: “The model contains two populations”; “Figure 2 shows
  mean accuracy”. A caption may mix present display descriptions with past
  measurement context, but observed findings belong in Results prose. Concise
  finding headings may use present tense.
- Use present tense for current interpretations and supported conclusions,
  keeping their evidential limits explicit: “This result suggests…”; “These
  measurements do not establish…”. An observed outcome remains past tense
  even when the interpretation is present tense.
- Mark unperformed work explicitly as a plan, possibility or prediction:
  “We plan to test…”; “The planned comparison will…”; “This could…”. Never
  change planned or uncertain work into a completed observation for stylistic
  consistency. Use present progressive only for work actually in progress.
- Short action-led Methods labels, mathematical instructions and reference-page
  instructions may remain imperative. Method-step prose must still distinguish
  completed work from plans. Preserve quoted wording, bibliographic titles,
  mathematical symbols and source-code identifiers.

For a tense/person conformance pass, preserve scientific meaning, execution
status, qualifications and authored dates. A grammatical change alone does not
advance `meta.updated_at`; a substantive correction follows section 3.3. Check
tense and attribution in context rather than replacing every present-tense verb
or every occurrence of a pronoun mechanically. Flag uncertain execution or
authorship rather than guessing.

### 3.6. Shared terminology and notation

The conventions below apply to recurring terms and symbols across the experiment
collection. They do not eliminate local definitions: define every nontrivial
symbol at its first use in each standalone article. Established theory may retain
its native notation when changing it would obscure the literature, but state its
mapping to the shared notation explicitly. Do not apply these rules as blind
textual replacements; preserve scientific meaning, units, executed provenance
and exact public interfaces.

#### Evidence provenance and display

These terms apply throughout the article, including metadata, abstracts,
Results, Methods, captions, alternative text, tables and appendices:

- Use **retained** only for evidence deliberately preserved from an earlier
  execution.
- Use **recorded** for data or measurements produced during the execution being
  described.
- Use **reused** for prior models, data, measurements or figures used again.
- Use **shown** or **displayed** for content visible in the current figure or
  table.

Choose the term from the relationship being described, not from the fact that a
file currently exists. Audit every occurrence during the completion gate;
grammatical plausibility is not sufficient.

#### Scientific entities and identities

1. Use **experiment** for the scientific investigation identified by `expXXX`.
2. Use **article** for its written account.
3. Use **run** for one completed compute, analyse or present execution.
4. Use **condition** for one defined point or intervention in an experimental
   design.
5. Use **training replicate** for one independently initialized and trained
   network.
6. Use **seed** for the stochastic-stream identifier, not for the resulting
   network.
7. Use **presentation** for one stimulus exposure; use **encoding draw** for one
   stochastic realization of that presentation.
8. Use **neuron** for an individual simulated cell.
9. Use **component** for a reusable model object containing one or more
   populations.
10. Avoid bare **cell** where it could mean a neuron, condition–seed network or
    multi-population component.

#### Time and indexing

11. Reserve $t$ for physical time.
12. Use $k$ for a discrete simulation-step index.
13. Relate discrete and physical time by $t_k=k\Delta t_{\mathrm{sim}}$, where
    $t_k$ is time at step $k$ and $\Delta t_{\mathrm{sim}}$ is the integration
    timestep.
14. Use $N_t$ for the number of simulation timesteps.
15. Use $T_{\mathrm{present}}$ for physical presentation duration.
16. Use $T_{\mathrm{readout}}$ for the duration over which readout evidence is
    accumulated.
17. Reserve $\tau$ for a time constant rather than an observation duration.
18. Use $\Delta t_{\mathrm{sim}}$ for integration timestep and
    $\Delta t_{\mathrm{bin}}$ for analysis-bin width.
19. Use $T_\gamma$ for gamma-cycle period rather than alternating between
    $T_\gamma$ and $P_\gamma$.
20. Attach units to time quantities at code or schema boundaries, such as
    `duration_ms` and `duration_s`. Keep these internal names out of rendered
    prose under section 3.

#### Populations, neurons and state

21. Use $N_E$ and $N_I$ for excitatory and inhibitory population sizes.
22. Use $N_{\mathrm{in}}$ and $N_{\mathrm{out}}$ for input and output population
    sizes.
23. Use $B$ only for the current minibatch size.
24. Use $n_{\mathrm{spike}}$ for a spike count.
25. Use $s[k]$ for a dimensionless binary spike indicator at simulation step
    $k$.
26. Use $s(t)=\sum_j\delta(t-t_j)$ for a continuous impulse train, where $t_j$
    is event time $j$ and $\delta$ is the Dirac impulse.
27. Do not use the discrete indicator $s[k]$ and continuous impulse train $s(t)$
    interchangeably without stating their relationship.
28. Use $V_m$ for physical membrane voltage and $V_{\mathrm{candidate}}$ for
    the pre-threshold candidate voltage.
29. Use a distinct symbol such as $u_{\mathrm{out}}$ for a dimensionless output
    state.
30. Preserve $C_m$, $g_L$, $E_L$, $V_{\mathrm{th}}$,
    $V_{\mathrm{reset}}$, $\tau_{\mathrm{AMPA}}$, $\tau_{\mathrm{GABA}}$ and
    $\tau_{\mathrm{ref}}$ as the standard biophysical vocabulary.

#### Rates, counts and activity

31. Use $r_E$ and $r_I$ for per-neuron excitatory and inhibitory firing rates.
32. Report per-neuron firing rates in hertz unless another unit is explicitly
    stated.
33. Use $n_P[k]$ for the spike count of population $P$ in analysis bin $k$.
34. Call $N_E r_E$ a population-total spike rate, measured in spikes per second,
    rather than a spike count.
35. Convert a population-total spike rate to a count by multiplying it by the
    observation duration in seconds.
36. Identify every rate's population, time window and averaging axes.
37. Use $r_{\mathrm{input,max}}$ for maximum-pixel or maximum-channel encoding
    rate.
38. Use $r_{E,\mathrm{ceil}}$ for a hidden-excitatory firing-rate ceiling.
39. Do not use $r_{\max}$ for both an encoding-rate maximum and a
    hidden-population ceiling.
40. Use **activity** only as an umbrella term when the specific quantity could
    be voltage, spikes, counts or rate.
41. Distinguish a tested **activity ceiling**, an empirical **rate plateau** and
    a demonstrated **lower bound**.
42. Use $p_{\mathrm{event}}$ for Bernoulli event probability.
43. Use $p_{\mathrm{part}}$ for measured cell-per-cycle participation.
44. Use $\beta_{rf}$, not $p$, for a fitted rate–frequency slope.
45. Treat $r_E\approx p_{\mathrm{part}}f_\gamma$ as a model-dependent
    approximation, not the definition of participation.

#### Readouts and classification

46. Name a readout by both its state model and its evidence operation.
47. Use **output LIF with mean pre-reset voltage** for a spiking output layer
    scored by averaged pre-reset voltage.
48. Use **output LIF with spike-count readout** for a spiking output layer scored
    by output counts.
49. Use **non-spiking leaky-integrator readout** for output units without
    threshold, reset or refractory dynamics.
50. Distinguish mean-voltage, final-voltage, spike-count, spike-rate and
    cumulative-potential readouts.
51. State the window over which readout evidence is accumulated.
52. State whether readout state resets at presentation boundaries.
53. State whether hidden neuronal state resets at presentation boundaries.
54. State whether the decoder knows presentation boundaries.
55. Use $z_c$ for the score or logit associated with class $c$.
56. Use $y$ for the true class and $\hat y$ for the predicted class.
57. Do not use $\hat y$ for both a score vector and a predicted class.
58. Do not call softmax-normalized evidence a calibrated probability or
    confidence unless calibration was measured.

#### Training, checkpoints and losses

59. Distinguish **training loss**, **validation loss** and **test metrics**.
60. Determine a metric's data split from the executed protocol rather than a
    legacy filename or panel label.
61. Use **selected checkpoint** as a role, not as the name of a universal
    selection algorithm.
62. When reporting checkpoint selection, state the split, primary metric,
    aggregation, tie rule and eligible epochs.
63. Use **final checkpoint** for parameters at the final completed training
    update or epoch.
64. Distinguish a parameter snapshot from a resumable checkpoint containing
    optimizer, random-stream and data-order state.
65. Use **Adam** and **AdamW** according to the executed optimizer identity.
66. Record weight decay separately from optimizer identity.
67. Use $L_{\mathrm{CE}}$ for cross-entropy loss and $L_{\mathrm{total}}$ for the
    complete objective.
68. Use $\lambda_{\mathrm{rate}}$ for the rate-penalty coefficient rather than
    an unqualified $\lambda$.
69. Use **voltage-gradient damping** as the scientific term.
70. Preserve exact implementation names such as `v_grad_dampen` and
    `voltage_grad_dampen` only when referring to those fields directly.
71. Use $d_{\mathrm{grad}}$ for the dimensionless damping divisor and
    $\alpha_{\mathrm{grad}}=1/d_{\mathrm{grad}}$ for its multiplier.
72. State which population and update term receive gradient damping.

#### Statistics and aggregation

73. Identify the axis or distribution represented by every mean.
74. Distinguish an expectation of a nonlinear response from the response
    evaluated at the mean input.
75. Qualify distribution parameters as $\mu_{\mathrm{init}}$,
    $\sigma_{\mathrm{init}}$, $\mu_I$, $\sigma_V$, $\sigma_{\mathrm{jitter}}$
    or $\sigma_{\mathrm{smooth}}$.
76. Label uncertainty as SD, SEM or a confidence interval explicitly.
77. State the number and kind of independent replicates underlying uncertainty.
78. Do not change an uncertainty definition when reusing a figure.
79. Use $R^2_{\mathrm{fit}}$ only for the coefficient of determination of a
    specified fitted model.
80. Use **accuracy** for the proportion of correct decisions and
    **percentage points** for absolute differences between percentages.
81. Use separate formatters for fractions converted to percentages and numbers
    already expressed as percentages.
82. State the handling of undefined estimator values rather than silently
    replacing them with zero.

#### Rhythms, frequency and phase

83. Use $f_{\mathrm{peak}}$ for a spectral-peak frequency.
84. State whether $f_{\mathrm{peak}}$ is a raw frequency-bin maximum or a
    sub-bin interpolated estimate.
85. State whether spectra were averaged before or after peak selection.
86. Use $f_{\mathrm{Hopf}}$ for a frequency inferred from a dynamical eigenvalue
    crossing.
87. Use $f_\gamma$ only after defining the gamma-frequency estimator used in
    that context.
88. Treat a frequency search interval as an estimator setting, not as the
    definition of gamma.
89. Use $\omega$ for angular frequency and state whether its units are radians
    per millisecond or radians per second.
90. When $\omega$ is in radians per millisecond, multiply by $1000/(2\pi)$ to
    obtain frequency in hertz.
91. Use $\Delta f_{\mathrm{bin}}$ for spectral-bin spacing.
92. Use $\Delta f_{\mathrm{detune}}$ for inter-network frequency detuning.
93. Use $\phi$ for phase and state whether it is measured in radians or cycles.
94. Use $R_{\mathrm{phase}}$ for circular phase concentration.
95. Use $R_{\mathrm{contrast}}$ for autocorrelation lobe–trough contrast.
96. Do not use **rhythmicity**, **coherence**, **contrast**,
    **phase concentration** and **phase locking** as interchangeable metric names.

#### Connectivity and coupling

97. Use $W$ for a realized connection matrix.
98. State the source and target axes of every connection matrix.
99. State pathway direction separately from matrix-storage orientation.
100.  Document any transpose between graph-declaration and runtime matrix
      orientations.
101.  Use $w_{\mathrm{event}}$ for the conductance increment caused by one
      presynaptic event.
102.  Use $G_{A\rightarrow B}$ for summed conductance from population $A$ to
      population $B$.
103.  Use $J_{A\rightarrow B}$ for current-valued or driving-force-rescaled
      coupling.
104.  Distinguish an individual edge weight, expected edge strength, realized
      summed conductance and current-valued coupling.
105.  State whether a reported coupling is per-edge, fan-in normalized or
      population summed.
106.  Distinguish excitatory or inhibitory polarity from the postsynaptic
      population receiving the conductance.
107.  Use separate notation for a signed driving force and its positive magnitude.
108.  Use $q_{\mathrm{zero}}$ for an initial zero fraction rather than $s$.
109.  Distinguish initial zeros, a permanent connectivity mask, a disabled
      projection and a frozen parameter.
110.  Do not call a lower-clamped Gaussian a truncated Gaussian unless negative
      values were resampled from the conditional distribution.

#### Dynamics and technical terminology

111. Use **silent** only for zero or criterion-defined negligible spiking.
112. Use **non-oscillating equilibrium** for a stationary state that may have
     nonzero firing rates.
113. Distinguish equilibrium, fixed point, steady state and time-averaged
     operating point when more than one appears.
114. Use $J_{\mathrm{flow}}$ for a continuous-time vector-field Jacobian and
     $J_{\mathrm{step}}$ for a discrete update Jacobian.
115. Use $\lambda_J$ for a dynamical eigenvalue rather than sharing $\lambda$
     with a loss coefficient.
116. Use $A_{\mathrm{pp}}$ for peak-to-peak oscillation amplitude rather than a
     bare $A$.
117. Use $A_{\mathrm{corr}}$ for an autocorrelation function if $A$ is otherwise
     needed for amplitude or accuracy.
118. Qualify **exact**, **stable**, **deterministic**, **equivalent** and
     **invariant** with the tested object and conditions.
119. Distinguish backend, executor, target, provider and device.
120. Preserve exact public API, schema, command-line and citation spellings
     rather than silently translating them into prose conventions.

### 3.7. Cross-links between experiment articles

Render every link from one experiment article to another in the exact form
`expXXX — Title`, using the target's zero-padded identifier and exact current
`meta.title`. Link the identifier and title separately to the same canonical
article URL. Leave the em dash unlinked, set only the title in italics, and do
not substitute descriptive labels or navigation text such as `Previous` or
`Next`. This rule applies in prose, captions, lists, tables and navigation.

When the cross-link ends a sentence, include the full stop in the italicized
title link so its underline remains visually continuous:

```typst
#link("/exp023/")[exp023] — #link("/exp023/")[_Turning the PING Loop On._]
```

## 4. Titles

For the experiment's overall title (`meta.title`), use a short, plain-English
phrase naming its main finding or controlled comparison. Prefer a specific
relationship ("Firing Rate Tracks Gamma Frequency") over vague topics or
promotional claims. State a finding only when supported by results; otherwise
name what is being tested. Aim for 5–10 words, retaining technical terms needed
for precision.

## 5. Preserved component writing contracts

These are verbatim leading comments from the former article templates. They
preserve component-specific scientific and editorial requirements; Typst API
spellings and rendering details are historical reference.

### abstract.typ

```text
Reusable Abstract writing template.

The four structured arguments are source-level prompts only; they are
rendered as two unlabelled paragraphs. `body` instead wraps an existing
authored abstract without changing its wording or paragraph structure.
Supply complete authored sentences, including their punctuation. Keep the
complete abstract concise (normally around 65 words), qualitative, and
independent of repository or run bookkeeping.

`question`
  The experiment's question or purpose.

`approach`
  What was reused, changed, compared, or tested.

`finding`
  The main qualitative observation. Never substitute an expectation or plan.

`scope`
  What the finding establishes, its reuse value, or its main limitation.

GROUNDING AND PROSE

Before writing, read the experiment entry point and follow its execution into
the relevant recipe, compute, analysis, presentation and helper code. Check
completed-run provenance, retained configurations and results: code describes
intended behaviour, while retained evidence establishes what happened. Resolve
discrepancies and distinguish new execution from reused evidence. Never run an
experiment merely to fill an Abstract.

Use plain, direct language for a reader familiar with the wider project but
not this experiment. Omit numerical results, sample sizes, parameter grids,
units, citations, general background and implementation detail. Never replace
an observation with an expectation. Apply the Writing Guide's global rules for
evidence, provenance, tense, person and repository-independent prose.

PUBLIC ENTRY POINT

`journal-abstract(body: none, question: none, approach: none, finding: none,
scope: none, status: none)` accepts either an existing authored `body` or all
four structured roles, never a mixture. It owns the level-2 Abstract heading, two-paragraph
structured rendering and the post-Abstract contents marker.
```

### article-layout.typ

```text
Shared article-shell contract.

`journal-article(article, inputs, body, dataset-placed: false,
dataset-status: none, dataset: true)` is the final
wrapper for every writing. Documentation-themed collection articles pass
`dataset: false` and omit explicit `run-view` calls. Other articles receive
the Dataset section unless the body already placed it. The shell then applies
article-wide equation numbering and the article-scoped contents wrapper in
that exact order. Pass `dataset-placed:
true` only when the body already renders the same Dataset view.

Keep this shell outside data-readiness branches. When present, order complete
level-2 sections as Abstract, Results, Methods, Dataset, appendices,
References. Other main-text sections may appear where scientifically
appropriate before Dataset. Reference pages and galleries do not need
invented sections.

The imported templates own their local contracts. Use `result-card.typ`
directly for Results wrapping and cards; the article shell does not re-export
component APIs.
```

### contents.typ

```text
Shared article-scoped contents contract.

Apply `with-contents(body)` once as the final navigation wrapper. It disables
general heading numbering and renders exactly one linked contents list from
the current article's level-2 headings. Direct level-3 Results subsections are
included as a nested numbered list; other deeper headings and neighbouring
book entries are excluded.

In an article with an Abstract, place exactly one `contents-here()` marker
after the complete Abstract and before the next level-2 section. With no
Abstract, omit the marker and `with-contents` places the list first. Keep this
wrapper outside data-readiness branches so populated and unavailable views
both receive navigation. Never maintain a manual contents list.
https://typst.app/docs/reference/introspection/query/
```

### dataset.typ

```text
Article-owned data-input and Dataset-section contract.

Bind `data-file` to the article identifier, declare every required logical
input key (including reused upstream inputs), and keep all data reads,
calculations and result-dependent content behind `inputs-ready`. A selected
input with a missing or corrupt file is an error. With no ready input, render
`pending-report`; do not substitute zeros, historical values or publication
paths. Preview and publication consume only explicit Demolab selections of
validated v4 present runs and never choose a run or launch an upstream stage.
Editing prose does not require a new run. Declare every scientific input the
article depends on, including reused upstream evidence in comparisons and
syntheses. Distinguish imported historical figures from newly generated
figures, and never imply that reused evidence came from a new simulation.
The versioned Pingstore Rules at `md/notes/pingstore-rules.md` remains
authoritative for run validity, layout and provenance.

Apply `with-datasets(article, inputs, body)` to add exactly one level-2
`Dataset` section. It belongs after all main-text sections and before the
first appendix or References; pass `placed: true` only when a data-backed body
already places the same `run-view` explicitly inside its required style scope.
Present exports and their projected provenance are display inputs, not a
second authority for scientific claims.

Public article entry points:

- `data-file(rel, article: none)` resolves one safe logical key or file path;
  bind the required `article` argument before use.
- `inputs-ready(data-file, inputs)` checks that every declared key resolves.
- `pending-report(data-file, inputs, question, figures, json-inputs: ())`
  renders the shared unavailable-data view.
- `with-datasets(article, inputs, body, placed: false, status: none)` inserts the Dataset
  section according to the placement contract above.
- `run-view(article, inputs, status: none)` renders that section directly when required.
- `input-assets(article, inputs)` projects selected media for publication.
- `video(source, ..args)` renders a selected presentation video.
Upstream and downstream dependency identifiers link to their canonical
experiment-article URLs; the projection does not carry article titles.
URL renders and prepared builds read Pinglab's own validated projection.
Legacy paths remain below until the separate retirement step is approved.
```

### equations.typ

```text
Shared displayed-equation numbering contract.

Apply `with-numbered-equations(body)` once to the complete article body. It
resets the article-wide counter and numbers every displayed equation
continuously as `(1)`, `(2)`, and so forth in HTML and PDF. Inline mathematics
remains unnumbered. Do not add local counter resets or individually assigned
display numbers around this component. Displaying an equation does not require
citing it in prose, and numbering does not justify displaying trivial
arithmetic. In HTML, mathematics uses the surrounding text size; displayed
equations receive paragraph-like margins and extra separation when adjacent.
```

### methods.typ

```text
Reusable Methods writing template.

Methods is a sequence of cards. Each card contains one scientific operation
or one existing authored content block. Structured `method-card(...)` values
become continuously numbered cards; authored content preserves its existing
lists and numbering. The template owns the Methods heading, with no required
stage groups or subheadings.

`orientation` (optional)
  A short introduction that adds information not repeated by the first step.

`body`
  The scientific procedure in dependency order, as structured steps or prose.

GROUNDING AND PROSE

Explain how the experiment was actually performed and how its reported
measurements and reusable outputs were obtained. Write for a computational-
neuroscience colleague who understands the field but not this experiment.
Use the shortest account that remains scientifically reproducible. There is
no minimum length. Use compact numbered operations
and add or lengthen operations only when a distinct procedure, definition,
decision or limitation is needed. Every sentence must be necessary to reproduce
or interpret the experiment; if deleting it loses neither, delete it. There is
no target length to fill and no permission to pad a complex subsection merely
because another operation is longer.

Read the execution code, scientific definitions, analysis and relevant
helpers. Check completed-run provenance, retained configurations and outputs;
resolve differences between current code and historical execution. Outline
the complete procedure, distinguishing newly executed, reused and planned
work. Never run an experiment merely to write Methods.

Cover applicable starting data, models, interventions, simulation, training,
primary measurements, selection, estimators, aggregation, comparisons,
inferential decisions and displayed evidence. Follow actual dependencies and
give each item one scientific purpose. Do not invent work to fill a category.
Use direct prose: what was done, essential settings, and what it produced.
Exclude repository bookkeeping, implementation narration and interpretation.

Put exhaustive grids, initialization distributions and derivations in
appendices, but keep essential model differences, selection criteria and
measurement definitions here. Select only equations central to the model,
intervention or measurement; define every symbol and unit, prefer scientific
notation over implementation-shaped indices, and omit decorative arithmetic.
Explain data partitions, model selection, timing, repetitions, aggregation,
consequential evidence selection, transformations and uncertainty. Leave
ordinary figure decoding to captions. Finish with a compression pass.
The finished account must be understandable without code or reconstruction
from appendices; each key equation must be understandable locally. Remove
presentation narration and repeated captions, and flag missing evidence rather
than inventing a procedure.

PUBLIC ENTRY POINTS

- `method-card(label, body)` creates one numbered scientific-operation card.
  Pass `none` as the label when the opening sentence already carries the
  subject.
- `journal-methods(body:, orientation: none, status: none)` owns the complete
  section and card container. Pass a step tuple or an authored content block
  as `body`; a content block becomes one card.
- Legacy `compute:`, `analysis:`/`analyse:` and `presentation:`/`present:`
  arguments remain accepted for existing articles, concatenated in that order
  without stage headings. Do not combine them with `body`.
- `methods-heading(status: none)` supports separately wrapped section bodies.
- Legacy `methods-stage(title, status: none)` emits no heading; existing
  article content and explicit numbering remain in place.
```

### parameters-table.typ

```text
Article parameter-table contract.

Use `parameters-table(headers, rows, caption:, columns: auto, align: left,
table-label: none)`
for a compact comparison of scientific parameters across conditions,
reference systems or empirical ranges. The component owns the level-2
heading, always exactly "Parameters", followed by the table and its caption.
Do not add a separate heading, introduction, notes or other prose to this
section. The caption appears below the table using the article's standard
figure-caption styling and table numbering.

`headers` must contain at least two content cells. Every row must contain
exactly the same number of cells as the header.
Pass either an integer column count or an explicit array of relative/absolute
column widths through `columns`; the default creates equal-width columns.
`align` accepts the same values as Typst's table alignment argument.
`table-label` optionally labels the generated figure for numbered references.

Supply an authored `caption` explaining how to read the table, including any
necessary definitions, evidential qualifications and citations. State units
in the cells or headers, define distribution summaries, and distinguish
measured values from model settings and broad literature comparisons. Keep
scientific interpretation in the appropriate article section. Do not imply
that unlike quantities are directly calibrated.
```

### references.typ

```text
Reusable References-section contract.

`journal-references(entries, status: none)` renders exactly one level-2 References section
from a nonempty tuple of reference records. Each record requires authored
`text` and may provide a DOI through `doi`. Keep entries in order of first
citation, pass the same tuple once, and place this component after every
appendix as the article's final section. Inline citations remain article-owned;
reuse the same citation number for repeated citations.

The underlying Demolab helper owns the heading, ordered-list rendering,
anchors and DOI links. Do not add a separate References heading or manually
number entries around this component.

Include only cited sources. Verify that each source supports its associated
claim. Author each entry with authors, title, publication venue, year, and a
DOI or stable URL where available. Keep literature references distinct from
upstream experiment and run provenance. Apply the Writing Guide's global
evidence and repository-independence rules.
```

### result-card.typ

```text
Reusable lab-journal Results-card template (var2).

CARD CONTRACT

One invocation represents one direct Results subsection. It combines a
self-contained title, exactly one `figure(...)`, and up to three short prose
paragraphs. Cards render the first needed prose paragraph before the figure,
then the figure and caption, followed only by any additional prose paragraphs
and notes. A one-paragraph card therefore has no prose below its figure. The
surrounding Results section still belongs inside
`with-result-sections[...]`, which owns Results numbering. `contents.typ`
independently discovers those headings for the article-scoped contents.
The role names below are source-level prompts only; they are not rendered as
labels. Omit unused roles instead of adding empty or generic prose.

Each card must work in isolation: a reader entering at the card should be
able to recover what was tested, what the figure displays, what happened,
and the main evidential limit. It should also advance the article's argument.
Integrate figure references into the claims they support, using a stable
numbered cross-reference and the relevant panel when applicable, for example
"the inhibitory population remained silent
(#result-figure-ref(<fig:example>, panel: "C"))". Do not add a generic
standalone sentence such as "The figure shows..." merely to satisfy this
rule, and do not use layout-dependent "above" or "below".

MINIMUM SUFFICIENT PROSE

Default to one compact observation paragraph. Add orientation only when the
title and caption do not establish the local question, and add expectation
only when a genuine prediction or diagnostic distinction changes how the
evidence should be read. Normally use no more than two prose paragraphs and
100 words across `orientation`, `expectation` and `observation`; this is a
working ceiling, not a target. Exceed it only when additional prose is needed
to interpret the evidence or state its limitation accurately. Delete scene
setting, procedural recap, transitions, and details already carried by the
title, figure, caption or Methods.

`title`
  A plain string containing a concise, context-free description of the
  subject, comparison,
  measurement, distinctive displayed condition, or supported outcome. Avoid
  generic titles such as "Overview" or "Results plot". Do not add a manual
  number. After whitespace is collapsed, the authored title must contain no
  more than 38 characters, including spaces and punctuation, and no more than
  7 words; the generated Results number is not counted. These limits keep the
  heading on one line at the standard article width and default zoom. Use a
  shorter title when unusually wide glyphs or multi-digit numbering require
  it; never shrink the heading or force unbreakable overflow. Read together in
  order, the card titles should form a compressed account of the experiment's
  reasoning and findings.

`lead` (optional, exceptional)
  One brief orientation sentence that takes the pre-figure position ahead of
  the ordinary prose roles, permitted only when a schematic or unfamiliar
  diagnostic cannot be interpreted cold. Do not use it merely to announce,
  summarize or duplicate the figure. Ordinary cards omit it; the first used
  role among orientation, expectation and observation then occupies the
  pre-figure position automatically.

`visual`
  Exactly one `figure(...)`, containing a retained scientific image, video,
  or table. Its caption explains how to read the evidence: panel mappings,
  variables, conditions, visual encodings, sampling unit, aggregation,
  uncertainty, and illustrative, reused, or newly measured status. The caption
  must not state or repeat the scientific finding from `observation`.

`orientation` (optional)
  Essential local context: what was tested, on what system, and why this
  figure exists. It may connect to the preceding card by naming the specific
  unresolved question, but never rely on "the previous result" without
  restating that result or question. Include only the minimum Methods detail
  needed to understand this card.

`expectation` (optional)
  An evidence-grounded prediction or the diagnostic distinction being tested.
  State what observable pattern would support or distinguish the proposed
  explanation. Omit this paragraph for exploratory displays rather than
  inventing a retrospective hypothesis. Keep expectations distinct from
  observations. For a schematic or theory diagram, describe the candidate
  mechanism or comparison and explicitly identify the diagram as non-evidence.

`observation` (optional)
  What the displayed evidence establishes: the decisive measurement or
  qualitative pattern, the scientific consequence when useful, and the main
  limitation. Prefer a few discriminating values over exhaustive reporting.
  Separate observation from interpretation; label post-hoc interpretations.
  Match causal strength to the intervention and evidence. Omit this paragraph
  when a diagram or exploratory figure supports no finding beyond its display.

`notes` (optional)
  A bullet list limited to selection provenance, caveats, or secondary audit
  details. Notes must not carry the primary result or become extra Methods.

DIVISION OF LABOUR

- The title advertises the card's subject or supported takeaway.
- The first needed prose paragraph supplies the context or claim before the
  figure; an exceptional lead takes this position when indispensable.
- The figure supplies the evidence immediately after that first paragraph.
- The caption supplies the information needed to decode that evidence.
- Orientation explains why the evidence is being shown.
- Expectation records what would discriminate the possibilities.
- Observation states what happened and what can safely be concluded.
- Notes preserve optional audit information.

Apply the deletion test. Without the prose, the figure and caption must remain
readable. Without the caption, the prose must not become a substitute legend
listing panels, variables, encodings, aggregation, or uncertainty. Never state
the same finding in both prose and caption.

STORY ACROSS CARDS

Result-card headings are presented together in the article contents. Read only
that heading sequence as an acceptance test: it must give a fast, progressive
account of the experiment rather than a set of interchangeable topic labels.
Each heading must add the next necessary step, so a reader can recover the
system or question, baseline, tested contrast, main result, and any mechanism,
robustness result or limit without opening the cards.

Prefer this investigation sequence when the experiment supports it:

  1. system or question
  2. baseline
  3. intervention or comparison
  4. discriminating response or parameter range
  5. main outcome
  6. mechanism, robustness result, failure mode or evidential limit

Omit stages that the experiment does not contain; do not manufacture figures
to fill the sequence. A later card should name the concrete uncertainty left
by the earlier evidence and then answer it with its own figure. Reword or
reorder headings when the contents sequence does not progress, while keeping
every card independently understandable. Do not add generic Results introductions,
transitions, summaries, or placeholders merely to make the sequence explicit.

FIGURE GEOMETRY

Design every figure at final publication size. Unless a target journal
overrides it, use approximately 90 mm for a single-column figure or 180 mm for
a double-column figure, with a maximum height near 170 mm. Use 4:3 for an
ordinary plot panel, 1:1 where equal scaling or matrix geometry matters, 3:2
for dense time series, and wider panels for rasters or long temporal axes when
scientifically justified. Do not impose one aspect ratio on a compound figure.

Arrange compound panels in balanced scientifically meaningful rows or grids.
Avoid empty cells and misleading hierarchy. Never stretch plots; change the
canvas, source layout or decomposition instead. Verify final-size labels,
legends, annotations, uncertainty marks and raster detail, using approximately
5–7 pt as the minimum final text range unless the target journal says otherwise.
In HTML, static card images are capped at 48% of the viewport height so one
figure does not consume the reading space on a laptop. Authored widths remain
upper bounds, and the browser preserves each image's intrinsic aspect ratio.

PUBLIC ENTRY POINTS

- `with-result-sections(body, number-subsections: false)` owns automatic
  Results numbering and the HTML/PDF Results container. Enable nested numbering
  only for an explicitly designed hierarchy.
- `result-card(body)` wraps an existing complete level-3 Results subsection
  without changing its authored structure.
- `result-figure-ref(target, panel: none)` renders a stable compact reference
  such as "Fig. 1" or "Fig. 1C" from a figure label.
- `journal-result-card(title:, visual:, lead: none, orientation: none,
  expectation: none, observation: none, notes: none, status: none)` owns one structured
  level-3 subsection, renders its first supplied prose paragraph before the
  figure and only additional prose paragraphs after it, and enforces the title
  limits above.
```

### status.typ

```text
Shared component status contract.

Titled components accept `status: none` (default; no indicator) or
`status: "locked"` (small muted text beside the title). A lock records the
author's instruction to preserve that component's authored content. Only an
explicit author request may change locked content or remove its lock.
This is an editorial status, not filesystem protection or article review.
Wrappers without a title do not have an independent status.
`component-title` retains the plain title for contents links and inspection.
```
