# Writing Guide

Version: **27.0.0**

The Writing Guide defines the conventions for Pinglab's published experiment
entries in `writings/expXXX.typ`. This file is the canonical guide.

## 1. Versioning

Version this guide independently of Pinglab and Demolab. Increment the major
version when changed requirements make previously compliant writing require
revision, the minor version for compatible additions, and the patch version for
corrections or clarifications that do not change requirements. Update the version
above and add a short entry to the version history when changing the guide.

### 1.1. Version history

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
7. Treat saved text as authored source regardless of origin. Once content has
   been saved in an existing `.typ` file, treat the live text as canonical
   authored content whether it was initially written by the author, drafted by
   an agent, or generated from experiment materials. A Writing Guide migration
   must transform this live source; it must not regenerate or replace the target
   section from experiment code, a template, or a fresh model draft. Derive the
   required changes from the difference between the requested guide versions,
   apply only those changes and their necessary dependencies, and preserve all
   compatible wording and scientific content. If compliance requires wholesale
   replacement or an uncertain substantive change, stop and present the conflict
   for explicit authorization.

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
from the table below. The entire value must be `[≡ TXT | vX.Y.Z]` or
`[▦ DATA | vX.Y.Z]`, including brackets, where `X.Y.Z` is the latest Writing
Guide version applied to that article. Demolab displays this authored string;
do not add a second status field, version field or icon field.

| Label | Local-data availability |
| --- | --- |
| `[≡ TXT \| v27.0.0]` | Article only: no usable, validated local presentation data is available for any declared article input, or the article declares no data inputs. |
| `[▦ DATA \| v27.0.0]` | Local data: usable, validated local presentation data is available for at least one declared article input, including reused upstream results. |

The badge reports availability in the working checkout at the last agent check,
not a live web-UI measurement. `[≡ TXT | vX.Y.Z]` does not mean literally text
without diagrams. `[▦ DATA | vX.Y.Z]` does not certify complete input coverage, successful rendering,
scientific quality, review or completion. Null and negative findings qualify
equally. For comparisons with only some inputs available, use the `[▦ DATA | ...]`
form and report the missing inputs in the task summary, not new article prose.

- Agents own freshness. At the end of an authorized article revision, relevant
  implementation or execution task, or change to local data availability,
  reassess the affected article and all articles whose declared inputs depend
  on the affected data keys, including comparisons and syntheses. Update the
  badge in either direction when the evidence changes. This is a necessary
  dependent metadata edit under section 3.1; explicit author scope restrictions
  still take precedence.
- Agents also own Writing Guide version freshness. Whenever this guide is
  applied to an article, set the badge version to the exact version applied.
  Whenever the guide version changes, update every article brought into
  conformance in the same migration; never advance an article's badge version
  without applying all requirements introduced through that version. Normal
  tests must reject badge versions that differ from the current guide version
  once a repository-wide migration declares all articles current. A badge-only
  version synchronization does not advance `meta.updated_at`.
- Read the current article's `inputs` and article-scoped bindings, and check
  their agreement with the publishing configuration. Run read-only
  `uv run pingstore discover` against the configured local source. Match the
  declared keys to discovery's authoritative `experiment` fields, not run-name
  substrings or the article ID alone. No inputs means the `[≡ TXT | ...]` form;
  otherwise at least one qualifying input means `[▦ DATA | ...]`. Availability need not mean that
  this run is currently selected for publication.
- Qualifying data comes from a completed, nonempty v3 present run validated
  under the Storage Guide, including layout, payload checksums and applicable
  input-provenance checks. Numbers, tables, figures and videos can qualify;
  image-only presentations need not have `numbers.json`. Code, remote jobs,
  compute/analyse-only runs, hidden incomplete runs, bookkeeping-only exports,
  prose claims or standalone illustrative diagrams do not establish the
  `[▦ DATA | ...]` classification.
- A failed discovery, inaccessible source or invalid provenance is an unresolved
  check, not an empty result. Do not guess or silently downgrade on that basis;
  preserve the existing badge and report the blocker. A successful check showing
  no matching local data does warrant `[≡ TXT | ...]`, even if remote results exist.
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

### 3.6. Tense and grammatical person

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
advance `meta.updated_at`; a substantive correction follows section 3.4. Check
tense and attribution in context rather than replacing every present-tense verb
or every occurrence of a pronoun mechanically. Flag uncertain execution or
authorship rather than guessing.

### 3.7. Shared terminology and notation

The conventions below apply to recurring terms and symbols across the experiment
collection. They do not eliminate local definitions: define every nontrivial
symbol at its first use in each standalone article. Established theory may retain
its native notation when changing it would obscure the literature, but state its
mapping to the shared notation explicitly. Do not apply these rules as blind
textual replacements; preserve scientific meaning, units, executed provenance
and exact public interfaces.

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
100. Document any transpose between graph-declaration and runtime matrix
     orientations.
101. Use $w_{\mathrm{event}}$ for the conductance increment caused by one
     presynaptic event.
102. Use $G_{A\rightarrow B}$ for summed conductance from population $A$ to
     population $B$.
103. Use $J_{A\rightarrow B}$ for current-valued or driving-force-rescaled
     coupling.
104. Distinguish an individual edge weight, expected edge strength, realized
     summed conductance and current-valued coupling.
105. State whether a reported coupling is per-edge, fan-in normalized or
     population summed.
106. Distinguish excitatory or inhibitory polarity from the postsynaptic
     population receiving the conductance.
107. Use separate notation for a signed driving force and its positive magnitude.
108. Use $q_{\mathrm{zero}}$ for an initial zero fraction rather than $s$.
109. Distinguish initial zeros, a permanent connectivity mask, a disabled
     projection and a frozen parameter.
110. Do not call a lower-clamped Gaussian a truncated Gaussian unless negative
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
#import "contents.typ": with-contents, with-result-sections
// Define the article body and apply any dataset/report wrappers first.
#let body = with-contents(body)
```

- Apply this as the final body wrapper, outside data-readiness branches, so
  navigation is present in both populated and unavailable-data views.
- Generate linked entries from the current article's rendered level-2 (`==`)
  section headings, in document order, including Abstract, Dataset, appendices
  and References when present. Beneath `Results`, include every direct level-3
  (`===`) Results subsection as a nested linked entry with its generated number.
  Omit every other deeper heading, the title, the TOC itself, and headings
  belonging to other articles in a book.
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

- Never number article headings except direct Results subsections. Number those
  subsections automatically and sequentially as `1.`, `2.`, and so forth through
  the shared Results wrapper; authors must not type ordinal prefixes into source
  headings. All level-2 sections, other subsections, appendices and References
  remain unnumbered. Use `=== Accuracy across input rates` in source, which
  renders as `1. Accuracy across input rates`; use `Appendix: Training settings`,
  not `Appendix A: Training settings`. The Writing Guide's own numbered policy
  sections are not article headings.
- Preserve numbers that identify scientific content rather than section order,
  such as a `4D model` or a training condition's name. Figure, equation, citation,
  reference-list and method-step numbering are unaffected.
- In every article containing both sections, place the entire Results section,
  including its subsections, before the entire Methods section. Do not move
  only the heading or leave results figures beneath Methods. Existing reference
  pages without these sections do not need invented sections.
- Name the section exactly `Dataset`. Place it after all main-text sections,
  including Methods when present, and before any appendices and References.
  If an entry has neither appendices nor References, place Dataset at the end.
  Keep it as a level-2 heading so it remains in the Table of Contents.
- Use descriptive section names with links for internal cross-references, not
  section numbers. When migrating, remove ordinal prefixes, repair links and
  textual references, and move complete sections as needed. Preserve scientific
  content, equation labels, citations, metadata and unrelated edits. Formatting
  and ordering changes alone do not advance authored dates (section 3.4).
- The shared article wrapper disables general heading numbering, while the
  shared Results wrapper numbers direct Results subsections only. Regression
  tests must check source headings, Results-before-Methods order, one figure per
  Results subsection, sequential rendered subsection numbers, nested TOC entries,
  and links in every article, including data-dependent bodies.

## 5. Abstracts

Write a standalone prose abstract, split across paragraphs, that quickly restores
the story of the experiment for a reader already familiar with the project. Aim
for around 65 words and do not use bullet points.

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

- State the question or purpose of the experiment.
- Describe the experimental move: what was reused, changed, compared or tested.
- State the main qualitative finding.
- State what the finding establishes, its reuse value, or the limitation that
  most constrains its interpretation.

Distribute these roles clearly across the paragraphs, combining adjacent roles
where that improves the flow.

Omit numerical results, sample sizes, parameter grids and units. Those belong in
Results. Use qualitative comparisons sufficient to distinguish the outcome.

Use plain, direct language and assume familiarity with the wider project, but
do not assume the reader remembers this experiment's particular setup. Retain
necessary scientific terms, while omitting general background, citations,
repository bookkeeping and implementation details.

Follow section 3.6 for tense and person. Never substitute an expected outcome
for an observation.

### Example: exp082

> This experiment asked whether networks trained on separate MNIST digits could
> classify digits presented continuously. Recurrent state was preserved between
> digits, while the output decision was reset at each boundary.
>
> Classification improved with longer, stronger inputs, whereas weak inputs often
> produced no output spikes. The results demonstrate continuous-stream
> classification, but do not separate viewing time from decision time or establish
> a causal role for gamma.

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

Build Results from numbered subsections, key figures, concise captions and
optional explanatory prose. Give captions and prose distinct roles: captions
explain how to read a figure; prose states what the figure establishes. Results
remain evidence-led; prose is not required when a figure has no supported finding
beyond the display itself.

1. Name the section exactly `Results` and place it before `Methods`.
   Do not append a tagline, description or other suffix, or add a
   section-number prefix (section 4.2).
2. Put the Results body inside the shared Results wrapper. Write each direct
   subsection as an unprefixed level-3 source heading; the wrapper generates its
   sequential number and the Table of Contents repeats that number.
3. Each Results subsection must contain exactly one figure. A compound image or
   a table wrapped in `figure(...)` counts as one figure; separate `figure(...)`
   calls require separate subsections.
4. Give each subsection a concise, self-contained title describing what its
   figure presents. Name the subject, comparison, measurement or distinctive
   displayed condition. Avoid generic titles such as `Training trajectories`,
   `Overview` or `Results plot`. Do not type the generated number into the title.
5. Use figures from retained experimental outputs. Keep captions concise and
   identify the variables, conditions, visual encodings, aggregation and
   uncertainty needed to read each figure correctly. Distinguish illustrative
   probes and reused observations from new measurements. Do not state the
   figure's scientific finding or interpretation in its caption. Reserve
   *retained* for evidence deliberately preserved from an earlier execution;
   use *shown* or *displayed* for the contents of the current figure, *recorded*
   for stored measurements, and *reused* for prior outputs analysed again.
6. A theory diagram may occupy its own Results subsection when useful. Its
   title and caption must identify it as an expectation or mechanism, not
   experimental evidence.
7. Explanatory prose may appear immediately before or after the subsection's
   figure. Use it for
   observed relationships, scientific consequences and clearly labelled
   interpretations. Keep it local to that figure and concise. Distinguish
   mathematical or theoretical expectations from observations, and label
   post-hoc interpretations. Do not repeat the caption or Methods, introduce
   unsupported causal claims, or add generic introductory or concluding prose.
   Do not add prose placeholders asking a later pass to fill these in.
8. Never state the same finding in both prose and the caption. Apply a deletion
   test: without the prose, the figure and caption must still be readable;
   without the caption, the prose must not substitute for the missing variables,
   conditions, encodings, aggregation or uncertainty.

When migrating existing headings, remove their taglines and repair any authored
links to former anchors. The shared TOC picks up the plain heading automatically.
Preserve scientific content and authored dates; apply section 3.5 to availability.

Illustrative scaffold (replace the figure paths and captions with retained
outputs and their measurement details):

```typst
== Results

#with-result-sections[

=== Gamma frequency and accuracy across inhibitory decay times

The oscillation slowed as inhibitory decay increased, but accuracy peaked at
the intermediate decay time. Slower rhythms therefore did not consistently
improve classification.

#figure(
  data-image(data-file("expXXX/frequency-and-accuracy.svg"), width: 100%),
  caption: [Gamma frequency and test accuracy across inhibitory decay times.
  Points show means across eight training replicates; error bars show ±1
  standard deviation.],
)
]
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
