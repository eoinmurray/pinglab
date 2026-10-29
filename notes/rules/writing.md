---
title: Writing Rules
author: Eoin Murray
collection: notes
description: Brief, plain-English writing grounded in code and data.
created_at: "2026-10-14"
updated_at: "2026-10-29T15:21:45Z"
---

# Writing Rules

Writing-only edits do not require test suites, builds, browser inspection or
desktop/mobile layout checks. Do not run these by default. Read the affected
source and make the requested edit; preserve scientific evidence and provenance.

<a id="default-article-structure"></a>

## 1. Default article structure

1. Put findings and evidence in the report; put model specifications and supporting
   detail in the appendix. Keep both in the experiment's `report/` directory:

   ```text
   experiments/expXXX/report/
       report.mdx
           Abstract
           Results
               One subsection per result or related figure group
       appendix.mdx
           Supporting methods and derivations, when needed
           Network diagrams
           Parameters
   ```

2. Use this default for new reports. Add Introduction, Methods or Discussion only
   when useful; adapt structure and figure types to the science. Training curves
   and rasters are not required.

<a id="javascript-animations"></a>

### 1.1. JavaScript animations

1. Keep experiment-specific browser scripts and helpers in `report/js/`; import
   them relatively, for example `./js/neuron.js`.
2. Keep Astro components, styles and captions in `report/`; keep shared components
   and scripts in `site/src/`.

<a id="optional-structure-for-complex-methods"></a>

### 1.2. Optional structure for complex methods

1. Add Methods after Results when connected derivations, dependent analysis stages
   or numerical procedures would interrupt the findings.
2. Use numbered named method steps and substeps, such as 2 and 2.1. Put supporting
   derivations in lettered appendices, with subsections such as A1 and A2.
3. Keep `report.mdx` and `appendix.mdx`; retain diagrams and parameters in the appendix.

<a id="report-first-drafts"></a>

### 1.3. Report first drafts

1. For first drafts, include Abstract, Introduction, Results, Discussion and the
   appendix, including network diagrams and parameter tables. Omit Methods and
   supporting methods text from the appendix. Results should contain only figures
   and captions, with subsection headings where useful; defer explanatory Results
   prose to later drafts. Put essential context, measurement conventions and
   limitations in the captions. This overrides the usual requirement for prose
   before each figure during first drafting only.

<a id="abstract"></a>

## 2. Abstract

1. Write one short paragraph covering the question, comparison, essential
   conditions, main numerical result and scope.
2. Describe only completed work. A single run supports that run's finding,
   not robustness across untested seeds or conditions.

<a id="results"></a>

## 3. Results

1. Order figures around the scientific question. Before each figure or group,
   briefly state the measured result and interpretation.
2. Keep essential method details nearby, normally only non-standard steps;
   use equations when they clarify the method.
3. Explain panels, colours, symbols, units, counting windows and reading conventions
   in captions. Keep construction details in Methods and avoid repeating prose.
4. State material limitations beside the evidence. If the current recipe differs
   from the one used, identify the relevant difference.

<a id="methods-optional"></a>

## 4. Methods (optional)

1. Use numbered, named steps in execution order, with numbered substeps for dependent
   operations. Link each stage to its result or figure panel.
2. Give enough detail to reproduce the procedure: sweep ranges, initial conditions,
   solvers, tolerances, measurement windows and decision criteria where applicable.
3. Report verification checks and limitations that affect interpretation. Keep
   routine settings in the parameter table.
4. Keep procedure-defining equations in Methods; link longer derivations to numbered
   appendix equations. A step may simply point to its derivation.

<a id="appendix"></a>

## 5. Appendix

1. Every report must include a network diagram and a parameter table in
   `appendix.mdx`; both are always required. Order the appendix as supporting
   methods and derivations, network diagrams, then the compact parameter summary
   and connectivity matrix specified below.
   Start with diagrams when no supporting text is needed.
2. Use named, lettered appendices for substantial topics, with numbered subsections
   such as A1 and A2. Develop derivations in order and define symbols and units.
3. Distinguish exact algebra, approximations and closure assumptions; state what
   each assumption omits. Give referenced equations unique numbers and stable
   anchors, linked directly from Methods.
4. Include each distinct network layout, preferably generated from the executed
   model. Do not create a separate `parameters.mdx`.
5. Every report must use [exp133's parameter layout](/exp133#parameters): one
   compact grouped summary followed by a source-to-target connectivity matrix.
   Use the shared `Table` component for both; do not present an exhaustive
   configuration dump or additional expandable parameter, pathway or analysis tables.
6. Combine matching E/I dynamics with paired values and units, for example
   `Membrane time constant, E / I` → `20 / 5.0 ms`. Explain pair order and derived values;
   distinguish absent populations, inapplicable settings and unavailable values.
   Population sizes must still have a separate row for every population.
7. Separate population or connection names from parameter descriptions with a space,
   without a colon: `E→I weights`, `Input→E synapse / delay`, `Output neurons`.

<a id="parameter-table-data-binding"></a>

### 5.1. Parameter-table data binding

1. Render SNNLab parameter tables from saved native snapshots, following the selected
   presentation's recorded compute ancestry. Select snapshots for the weights shown,
   such as final-training or best-validation weights.
2. Read input, training and analysis settings from the same run's recorded configuration.
3. Keep row selection, labels, grouping, units and display-only derivations in reports.
   Do not use separate `parameter_table` payloads, preformatted rows or copied
   parameter dictionaries from analysis or presentation outputs.
4. Shared report loaders may resolve selected lineage and read native snapshots;
   they must not replace native serialization or silently select another run.
5. Show missing snapshots or fields as unavailable, or fail explicitly. Never
   fall back to current recipe defaults or old table rows.

<a id="parameter-table-layout"></a>

### 5.2. Required parameter-table layout

1. Start the summary with **Simulation**, including integration timestep `dt`,
   simulated or analysed duration, relevant event times and network/input seeds.
   Distinguish a diagnostic excerpt's duration from the full simulation. For
   calculations without a timestep, state the applicable solver or that `dt` is
   inapplicable; do not invent simulation settings.
2. Follow with **Populations**. Report every neuronal population separately with
   its symbol, size and model, including each circuit's E and I populations and
   any readout populations. Give every external input pool its own row with its
   symbol and channel count, including silent attention inputs. Define symbols
   and any aliases used between protocols in the caption.
3. Follow with **Neuron dynamics**, **Input and manipulation**, and **Measurement**
   where applicable. Keep neuron dynamics to membrane time constants and refractory
   periods by default. Omit initial state, capacitance, leak conductance and
   resting/reset/threshold voltages from the summary. Include input rates,
   experimental manipulations and sweeps, and the primary measurement window.
   Add compact Training or Readout groups only when the experiment requires them.
   Use comparison columns for distinct models or protocols, as in exp133.
4. Select the rows needed to understand the experiment. Describe scientifically
   necessary additional settings, pathway-specific synaptic dynamics, initial
   conditions and detailed analysis criteria in Methods or concise captions.
   Complete native snapshots and recorded configurations remain the reproducibility
   record; do not duplicate them as supplementary or expandable parameter tables.
5. Place the **connectivity matrix immediately after the summary**. Rows are
   sources and columns are targets. Include every neuronal population as a source
   and target, and every external input pool as a source. Show intrapopulation
   connectivity on the diagonal and all interpopulation pathways off the diagonal,
   covering both local circuit connections and connections between circuits.
6. Each connected matrix cell gives eligible-pair connection density and nominal
   weight per connection, with mean ± standard deviation for random initializers.
   State weight units in the caption. Derive values using the saved normalization,
   execution scaling and relevant scheduled gain. Distinguish nominal initializer
   parameters from measured weight statistics and total incoming conductance.
7. Write `Absent` only when no projection exists. A zero weight, temporarily zero
   gain or silent input does not make a projection absent. Explain excluded
   self-connections, exact fan-in, the gain phase represented and the selected
   condition when the matrix illustrates one point of a sweep.
8. Share a matrix across compared protocols only when their native connectivity
   and effective nominal parameter conventions agree; otherwise show separate
   labelled matrices. Check equivalence before display rounding. Keep all values
   tied to the same selected evidence as the figures, following section 5.1.
9. Use mathematical population labels with `{ math: "..." }` cells and set
   `rowHeaders` on the matrix. Column and row headers must use the shared table
   header styling, matching grouped summary headings. Keep body-cell backgrounds
   uniform; do not add block shading or decorative separators for circuit groups.
10. Display numeric quantities to **two significant figures** throughout the tables,
    including quantities embedded in strings, Markdown, mathematics and captions.
    Numeric cells are formatted by `Table`; use `formatTableNumber` from
    `site/src/lib/table-number.ts` when composing rich cells or strings. Keep seed,
    run and other identifiers exact as strings. Perform calculations and comparisons
    using unrounded evidence, and retain exact values in native evidence.

<a id="evidence-and-reproducibility"></a>

## 6. Evidence and reproducibility

1. Ground scientific claims in code and displayed data. Omit unsupported claims
   and unperformed work.
2. Bind figures, numerical summaries and parameters to the same selected evidence.
   Use core’s resolved current runset once per production build or development
   page request; do not pin run IDs in report metadata, figures, thumbnails or animations.
   Prefer data-bound numerical summaries where supported.
3. For SNNLab simulations, read model parameters from exported `SnapshotSpec`
   evidence and input, training and analysis settings from recorded configuration.
4. For analytical models, identify imported and derived parameters, assumptions
   and experimental choices. Use settings that produced the displayed calculation.
5. Never replace displayed results' settings with current recipe settings.
6. Distinguish training, validation, test and diagnostic examples. Identify checkpoint
   selection, measurement conventions and seed count; label diagnostics clearly.

<a id="style-components-and-maintenance"></a>

## 7. Style, components and maintenance

1. Use plain English and keep writing brief.
2. Define terms, symbols and units; use standard variable names.
3. Reuse site components for figures, tables, diagrams and shared presentation;
   prefer existing Astro components to custom markup.
4. Expand explanations or derivations and depart from these rules when needed
   to make the science clearer.

<a id="editing-and-dates"></a>

### 7.1. Editing and dates

1. For reports and notes, set `created_at` once. Update `updated_at` for substantive
   saved prose or figure revisions, including incorporated results.
2. Preserve authored update dates for housekeeping and execution alone.
   Use quoted UTC datetimes (`YYYY-MM-DDTHH:mm:ssZ`).
3. Correct article evidence and availability or version tags when code or run
   changes invalidate them.

<a id="grid-safe-prose"></a>

### 7.2. Grid-safe prose

1. Article bodies and appendices use a CSS grid. Standalone MDX expressions
   returning strings or numbers produce anonymous grid items, which cannot receive
   the text-column placement rule and may squeeze into a narrow spare column.
   Render bound prose in an explicit paragraph, for example
   `{data && <p>Measured frequency: {data.frequency_hz} Hz.</p>}`.
   Expressions inside ordinary Markdown paragraphs already have a paragraph wrapper.
2. Keep conditional figures, tables and other block components as block elements;
   do not wrap them in paragraphs. Never use width overrides or forced line breaks
   to disguise unwrapped text.
3. The shared HTML middleware rejects non-whitespace text directly in article and
   appendix grids, including the transparent report-appendix wrapper. Development
   shows a layout error with the URL and offending text; production builds fail.
