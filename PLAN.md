# exp110 shortening and consolidation plan

- Created: 2026-09-09
- Manuscript: [writings/exp110.typ](writings/exp110.typ)
- Writing Guide reviewed: 36.0.0
- Progress: 0 of 9 items completed. Next: item 1.

## Aim and starting point

Turn the individually grounded Methods paragraphs into a concise research
article by removing repetition and assigning each explanation a clear home.
Preserve the scientific procedures, evidence, qualifications and reproducibility.
Work through the numbered items one at a time; do not make a simultaneous
whole-manuscript rewrite.

The read-only review found approximately 5,600 Methods words, 2,250 Results
words and 2,100 figure-caption words, excluding equations, table contents,
editorial notes and appendix scaffolds. There are nine figures with 53 panels.
Introduction and Discussion are not yet present; appendices remain scaffolds.

There is no mandatory word-count target. Results at approximately 2,250 words
and captions averaging approximately 230 words per figure are reasonable.
The main opportunity is duplicated Methods content, rather than their length
alone. Broad working ranges discussed were 2,000–4,000 Methods words,
1,500–3,000 Results words and 100–300 words per caption; these are editorial
planning ranges, not measured averages or universal journal limits. The earlier
3,000–3,500-word Methods target is provisional. Do not force Results down to
1,800–2,000 words or captions to 1,300–1,500 words merely to meet earlier estimates.

Journal context: [Nature](https://www.nature.com/nature/for-authors/formatting-guide)
guides Methods toward no more than 3,000 words and legends below 300 words each;
[Nature Neuroscience](https://www.nature.com/neuro/content) permits 4,500 words
for Introduction, Results and Discussion combined, excluding Methods and legends;
[PLOS Computational Biology](https://journals.plos.org/ploscompbiol/s/submission-guidelines)
has no manuscript word-count or figure-count restriction. Select a final budget
when the target journal is known.

## Ordered checklist

P1–P30 refer to the paragraph identities used during drafting; locate them by
their content because the completed manuscript no longer displays those labels.

- [ ] **1. Establish the shared evaluation protocol once.**
  Consolidate P5–P6, P10, P18–P27 and P29: define the common test subset,
  nominal presentation duration, matched encoding policy and replication once.
  Keep Table 1 as the checkpoint-assignment map and remove P10's duplicate
  experiment-by-experiment checkpoint list. Shorten each experimental account
  to its intervention, measurements and exceptions. Define SD/SEM centrally
  while keeping each figure's estimator unambiguous. Retain distinct validation
  draws, reference-image probes, streaming samples and realized durations.
  **Complete when:** repeated baseline descriptions are removed and every
  experiment still has an unambiguous protocol, checkpoint and denominator.

- [ ] **2. Make Table 2 the primary home for shared numerical settings.**
  Consolidate the numerical inventories in P2–P4 and the training paragraphs
  against Table 2. Keep parameter names, symbols, units and values together;
  prose should explain mechanisms and choices rather than repeat the table.
  Preserve locally intelligible equations, input zeroing and its ×20 rescaling,
  the absence of a permanent mask, and the distinct readout initialization.
  Keep both tables compact, centered and consistently captioned.
  **Complete when:** shared settings have one primary specification and all
  exceptions remain explicit without widening the tables unnecessarily.

- [ ] **3. Consolidate shared output-neuron dynamics in P7 and P26.**
  P7 now defines the ten spiking output units, decay, threshold, subtractive
  reset and absence of a refractory period. Reference that shared model in P26.
  Keep P26 focused on spike-count logits, readout initialization, variable-rate
  training, state continuity and output resets at supplied image boundaries.
  Preserve the distinction between ordinary mean-pre-reset-state logits and
  streaming spike-count logits; do not describe ordinary outputs as nonspiking.
  **Complete when:** the dynamics are explained once and both readout rules
  remain fully specified.

- [ ] **4. Consolidate the mean-field account with Appendix C.**
  Keep P15–P17's model equations, principal assumptions, oscillatory-onset
  definition and basis for the numerical criticality assessment in Methods.
  Relocate detailed continuation grids, finite-difference increments,
  initialization perturbations and regression conventions to Appendix C,
  merging overlaps with its scaffold and numerical-settings table.
  Preserve the unfitted effective-noise assumption and distinction between
  numerical classification and analytical proof; retain the red 4-mV note.
  **Complete when:** relocated details exist as actual appendix text, not just
  instructions to add them, and the main account remains interpretable alone.

- [ ] **5. Give measurement algorithms one home.**
  Consolidate P13–P14 and P22 with Appendix B. Keep measurement definitions,
  essential binning/smoothing choices, the cycle definition, pooling rules and
  exclusions in Methods. Put interpolation arithmetic, implementation details
  and boundary handling in the appendix without describing them twice.
  Preserve missing-versus-zero contrast handling and the exclusion of 12
  burst-free presentations from cycle participation only.
  **Complete when:** measurements and denominators are clear in Methods and
  all relocated algorithmic details are preserved in the appendix.

- [ ] **6. Shorten the illustrative-stream account in P28.**
  Keep its sampling procedure, predetermined candidate order, five-correct
  selection criterion and the fact that the first candidate qualified.
  Give the exact illustrated duration–rate sequence one primary location,
  using the existing Fig. 9 caption rather than repeating it in P28.
  Keep exact reproducibility seeds recorded. Consider writing the softmax
  transformation inline while defining its symbols and retaining the facts
  that shares are uncalibrated and the 0.5 line is not a decision threshold.
  **Complete when:** selection remains transparent without duplicated display
  descriptions. Recount Methods after items 1–6 before assessing further cuts.

- [ ] **7. Trim captions selectively.**
  Start with Figs. 4, 6 and 7, where initialization or measurement procedures
  repeat Methods. Keep panel identities, units, replicate counts, uncertainty
  definitions and interpretation-critical qualifications. Keep enough context
  for each figure to be understood independently. Move unique procedure
  details into Methods before removing them from captions. Allocate internal
  source-experiment links to web/reproducibility documentation for a journal
  version, preserving traceability. Do not shorten reasonable captions merely
  because their combined word count is large.
  **Complete when:** each caption clearly decodes its figure without reproducing
  a full experimental protocol, and necessary source links remain available.

- [ ] **8. Make a light Results pass and remove revision-history narration.**
  Compress the Fig. 2 raster paragraph where it repeats the coupling-grid
  progression. Remove closing restatements elsewhere when they add no inference
  or qualification. Preserve numerical comparisons and the boundaries of the
  conclusions. Move P3's account of which measurements were recomputed into
  reproducibility documentation while retaining actual refractory/timestep
  policies and scientifically relevant reuse. Consolidate similar maintenance
  narration where encountered, without removing evidence provenance.
  **Complete when:** repetition is reduced without changing findings or their
  interpretation. A major reduction in Results length is not required.

- [ ] **9. Consider supplementary placement of supporting panels — optional.**
  Revisit only if figure scope is explicitly reopened. First candidates are
  Fig. 7C's timestep control and Fig. 5D–G's detailed weight summaries;
  Fig. 9F repeats the 200-ms row of Fig. 9E and is another candidate.
  Preserve the main findings and their supporting evidence. Any chosen move
  must update panel lettering, captions, cross-references and associated Methods
  together. Moving material shortens the main paper but does not reduce total
  scientific content.
  **Complete when:** a chosen restructuring is implemented and verified.
  If no move is chosen, record this item as deferred rather than completed.

## Rules for completing each item

- Read the live manuscript immediately before editing and preserve concurrent
  work. Read and apply the current Writing Guide before manuscript changes.
- Keep the scientific claims, numerical values, units and substantive caveats.
  Preserve test/validation separation, checkpoint roles, independent replicate
  counts, SD/SEM distinctions, pooled denominators and selection criteria.
  Do not reinstate the discarded claim that replicate seeds independently
  randomized minibatch order and training encodings.
- Preserve red notes for unresolved scientific or publication questions. Do not
  remove them to reduce the apparent word count, invent their resolution or
  expand unrelated appendix scaffolds.
- Check the focused diff and scientific meaning; run `uv run demolab build`,
  inspect generated text/math, and run read-only `uv run pingstore discover`
  to maintain availability tags. Preserve author-assigned review status and
  keep the applied Writing Guide tag current. Run `git diff --check`.
- Do not create automated tests for the writings or inspect the localhost
  rendering with an in-app browser. Give the author the article link for visual
  review. No experiment execution, data mutation or publication is needed for
  prose consolidation; separately scope implementation work for optional item 9.
- After the edit and checks, mark that item's checkbox complete, update the
  progress line, and add a dated completion-log entry with scope, validation,
  unresolved issues and approximate before/after counts. Distinguish deleted
  repetition from material relocated to an appendix or supplement. Report the
  completed item before proceeding to the next numbered item.

## Completion log

No consolidation items completed yet. The nine-item plan was recorded on
2026-09-09; creating this plan did not change the manuscript.
