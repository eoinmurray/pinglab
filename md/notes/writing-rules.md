---
title: Writing Rules
author: Eoin Murray
collection: notes
description: Brief, plain-English writing grounded in code and data.
created_at: "2026-10-14"
updated_at: "2026-10-22T10:31:33Z"
---

# Writing Rules

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
   methods and derivations, network diagrams, then one grouped parameter table.
   Start with diagrams when no supporting text is needed.
2. Use named, lettered appendices for substantial topics, with numbered subsections
   such as A1 and A2. Develop derivations in order and define symbols and units.
3. Distinguish exact algebra, approximations and closure assumptions; state what
   each assumption omits. Give referenced equations unique numbers and stable
   anchors, linked directly from Methods.
4. Include each distinct network layout, preferably generated from the executed
   model. Do not create a separate `parameters.mdx`.
5. Group parameters as applicable: Network, Neurons, Synapses and weights, Input
   and simulation, Readout, Training, and Analysis. Use columns for compared models.
6. Combine matching E/I parameters with paired values and units, for example
   `Leak conductance, E / I` → `0.05 / 0.1 µS`. Explain pair order and derived values;
   distinguish absent populations, inapplicable settings and unavailable values.
7. Separate population or connection names from parameter descriptions with a space,
   without a colon: `E→I weights`, `Input→E synapse / delay`, `Output neurons`.

<a id="parameter-table-data-binding"></a>

### 5.1. Parameter-table data binding

1. Render SNNLab parameter tables from saved native snapshots, following the pinned
   presentation's recorded compute ancestry. Select snapshots for the weights shown,
   such as final-training or best-validation weights.
2. Read input, training and analysis settings from the same run's recorded configuration.
3. Keep row selection, labels, grouping, units and display-only derivations in reports.
   Do not use separate `parameter_table` payloads, preformatted rows or copied
   parameter dictionaries from analysis or presentation outputs.
4. Shared report loaders may resolve pinned lineage and read native snapshots;
   they must not replace native serialization or silently select another run.
5. Show missing snapshots or fields as unavailable, or fail explicitly. Never
   fall back to current recipe defaults or old table rows.

<a id="evidence-and-reproducibility"></a>

## 6. Evidence and reproducibility

1. Ground scientific claims in code and displayed data. Omit unsupported claims
   and unperformed work.
2. Bind figures, numerical summaries and parameters to the same pinned evidence.
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
