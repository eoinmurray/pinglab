# Writing Guide

Version: **6.2.0**

The Writing Guide defines the conventions for Pinglab's published experiment
entries in `writings/expXXX.typ`. This file is the canonical guide.

## 1. Versioning

Version this guide independently of Pinglab and Demolab. Increment the major
version when changed requirements make previously compliant writing require
revision, the minor version for compatible additions, and the patch version for
corrections or clarifications that do not change requirements. Update the version
above and add a short entry to the version history when changing the guide.

### 1.1. Version history

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

## 4. Titles

For the experiment's overall title (`meta.title`), use a short, plain-English
phrase naming its main finding or controlled comparison. Prefer a specific
relationship ("Firing Rate Tracks Gamma Frequency") over vague topics or
promotional claims. State a finding only when supported by results; otherwise
name what is being tested. Aim for 5–10 words, retaining technical terms needed
for precision.

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

## 6. Inputs and outputs

### 6.1. Data access in Typst source

For staged experiments, the writing consumes a selected presentation run, not
compute checkpoints or live analysis. Pingstore resolves that input to the flat
`export/` of a v3 present run, or `presentation/` of a compatible v2 run. Keep
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

### 6.2. Scientific inputs and outputs

Apply these rules to every experiment entry:

1. Name the section `Inputs and outputs` exactly and place it immediately after
   the abstract, before `Design Scope`.
2. Use two short paragraphs labelled `Inputs` and `Outputs`. Identify upstream
   scientific work by its descriptive name and identify the reusable outputs
   this experiment provides, such as trained models, parameter sets, datasets,
   or measured responses. Explain briefly what each input is used for and what
   each output is suitable for.
3. Describe the scientific provenance of inputs and outputs through their data,
   conditions, seeds, and selection criteria where relevant. Do not display run
   IDs, storage paths, or filenames; exact technical provenance remains in the
   authoritative run records outside the article.
4. State explicitly when there are no upstream experiment inputs or no reusable
   outputs.
5. Label planned outputs as planned, distinguish them from outputs actually
   produced, and never invent an input, output, or provenance claim.
6. Keep the section very short. Do not inventory every plot or repeat the
   conclusions.

Illustrative example (adapt the conditions and outputs to retained evidence):

```typst
== Inputs and outputs

*Inputs:* Uses trained models and parameter settings for seeds 42–44 from
the #link("/exp022/")[Training Runs] study as starting states for inhibition
sweeps.

*Outputs:* Provides measured frequency responses across inhibitory decay
conditions for subsequent comparisons of network timing.
```

## 7. Design Scope

Apply these rules to every experiment entry:

1. Name the section `Design Scope` and place it after `Inputs and outputs`.
2. Describe the experiment's parameter space in plain English: what system is
   studied, what changes, the tested values or ranges, and what remains fixed.
   Explain what each varied parameter means physically or operationally.
3. State the relevant starting conditions, comparison groups, and sampling
   scale, including independent seeds or repetitions where applicable.
4. Explain why these comparisons address the experiment's question and identify
   the main boundaries: what the experiment does not vary or test.
5. Keep this section short. Prefer ordinary prose over notation or configuration
   dumps; leave execution details and equations to `Methods`. Distinguish planned
   settings from those actually tested.

## 8. Prior art

Apply these rules to every experiment entry:

1. Name the section `Prior art` and place it after `Design Scope`, before
   `Results`.
2. Give a short, plain-English account of relevant work by others: what they
   studied, how they approached it, and what they established.
3. Explain how this experiment relates to that work—such as reproducing,
   adapting, extending, or testing a limitation—without implying novelty merely
   because the implementation differs.
4. Cite the sources beside the claims they support. Prefer original research for
   specific findings; use reviews for broader context.
5. Include only background needed to understand this experiment. If no directly
   relevant work has been identified, state that limitation rather than claiming
   none exists.

## 9. Results

Stub the Results section with numbered headings, figures and concise captions
only. Do not fill it with narrative prose.

1. Name the section `Results` exactly and place it before `Methods`.
2. Use numbered subsection headings such as `1. Rhythm frequency`, not
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

Illustrative scaffold (replace the figure paths and captions with retained
outputs and their actual measurement details):

```typst
== Results

=== 1. Rhythm frequency

#figure(
  data-image(data-file("expXXX/frequency-vs-decay.svg"), width: 100%),
  caption: [Gamma frequency across inhibitory decay times. Points show means
  across seeds; error bars show ±1 standard deviation.],
)

=== 2. Population firing rates

#figure(
  data-image(data-file("expXXX/rate-vs-decay.svg"), width: 100%),
  caption: [Excitatory and inhibitory firing rates across conditions. Points
  show means across seeds; error bars show ±1 standard deviation.],
)

=== 3. Classification accuracy

#figure(
  data-image(data-file("expXXX/accuracy-vs-decay.svg"), width: 100%),
  caption: [Test accuracy across inhibitory decay times. Points show means
  across seeds; error bars show ±1 standard deviation.],
)
```

## 10. Methods

Explain how the experiment was actually performed and how its reported
measurements and reusable outputs were obtained. Write for a
computational-neuroscience colleague who understands the field but does not know
this experiment. Aim for 300–450 words, excluding displayed equations but
including symbol definitions. Use fewer words for simple experiments; exceed this
only when scientific completeness requires it.

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
3. Prefer five to eight scientific operations in one flat numbered list, with
   at most ten. Do not use nested lists or subsection headings. Derive the
   operations from the experiment rather than imposing a fixed template.
4. Follow the actual dependencies: starting data and models, controlled changes,
   execution, selection, measurement, aggregation and outputs. Include applicable
   stages; do not stop at training when the report also contains evaluation or
   analysis.
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
   reused observations from new measurements.
9. Use direct, concrete prose. Exclude repository bookkeeping, implementation
   narration and result interpretation. Finish with a compression pass: remove
   repeated definitions, textbook exposition, procedural signposting and details
   already supplied elsewhere. Preserve what was varied, what was held fixed,
   what was trained, and how outputs were selected and measured.

### Completion check

Can the reader recover the procedure from the numbered labels, understand each
key equation locally, and trace every reported measurement and reusable output
to its source? The main account must explain the procedure without requiring
code or reconstruction from appendices. Flag missing evidence rather than
inventing a step.

For exp022, the applicable sequence is: data and splits; networks and controlled
conditions; encoding and simulation; class scores; training and optimization;
validation and model selection; measurements and reusable outputs. Its key
equations define the readouts and the sample-wise activity penalty; routine
Poisson and softmax expansions need not appear in the main account.

## 11. References

Apply these rules to every experiment entry:

1. Place a numbered `References` section at the bottom of the entry, after any
   appendices.
2. Use `#cite(...)` for inline citations and `#reference-list(...)` for the
   reference list.
3. List sources in order of first citation, with authors, title, publication
   venue, year, and a DOI or stable URL where available.
4. Reuse the same number for repeated citations. Keep citation numbers
   synchronized with list positions.
5. Include only cited sources and verify that each supports its associated
   claim. Keep literature references distinct from upstream experiment and run
   provenance.
