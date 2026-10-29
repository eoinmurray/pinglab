---
title: Writing Rules
author: Eoin Murray
collection: notes
description: Brief, plain-English writing grounded in code and data.
created_at: "2026-10-14"
updated_at: "2026-10-29T17:43:19Z"
---

# Writing Rules

Writing-only edits require source/diff review, not tests, builds or browser/layout
checks. For executable changes, use focused checks and broaden only when failures
or scope justify it. Simulation-backed checks require execution authorization.

<a id="default-article-structure"></a>
<a id="abstract"></a>
<a id="results"></a>
<a id="methods-optional"></a>
<a id="optional-structure-for-complex-methods"></a>
<a id="appendix"></a>

## Reports

1. Keep findings in `report/report.mdx`: a short Abstract and Results, with
   Introduction, Methods or Discussion when useful. Abstract: question, comparison,
   essential conditions, main numerical finding and scope. Explain each result
   before its figure; captions define panels, symbols, units and measurement windows.
2. Every report needs `appendix.mdx` with each distinct network diagram and the
   required parameter tables; no separate `parameters.mdx`. Put supporting methods
   and derivations before diagrams and parameters. Number method steps, letter
   substantial appendices and give referenced equations stable numbered anchors.
   Include enough procedure to reproduce results; distinguish exact algebra,
   approximations and closure assumptions. Adapt structure when science needs it.
3. <a id="report-first-drafts"></a>First drafts include Abstract, Introduction,
   Results, Discussion and appendix diagrams/tables. Omit Methods and supporting methods text. Results contain
   figures, captions and optional subsection headings; defer explanatory prose.
   Put essential context and limitations in captions.

<a id="evidence-and-reproducibility"></a>
<a id="parameter-table-data-binding"></a>

## Evidence binding

1. Describe completed work only; distinguish diagnostics, training, validation and
   test evidence, seed counts and checkpoint selection. State material limitations
   and differences between the current recipe and displayed evidence.
2. Bind figures, summaries and parameters to core's same resolved runset per page
   request/build. Do not hard-code report run IDs. Read model parameters from the
   selected ancestry's native snapshots and other settings from recorded configuration;
   analytical models must identify imported/derived settings and assumptions.
3. Keep display derivations and formatting in reports; do not export copied
   parameter dictionaries or preformatted table payloads. Missing evidence must be
   unavailable or fail explicitly, never fall back to recipe defaults or old rows.

<a id="parameter-table-layout"></a>

## Parameter tables

Use exp133's compact grouped summary followed immediately by a source-to-target
matrix, through shared `Table`; no exhaustive or expandable extra tables. Preserve
Simulation-first ordering, every population/input pool, two significant figures,
exact identifiers, standard headers and uniform body backgrounds. Read
[Parameter Tables](/notes/guides/parameter-tables) when creating or changing them.

<a id="style-components-and-maintenance"></a>
<a id="javascript-animations"></a>
<a id="grid-safe-prose"></a>

## Style and components

1. Use brief plain English; define terms, symbols and units. Prefer shared site
   components. Keep experiment components/styles/captions in `report/`, browser
   scripts in `report/js/` with relative imports, shared components/scripts in `site/src/`.
2. Wrap standalone bound prose in explicit paragraphs; keep conditional block
   components outside paragraphs. Shared markup validation rejects anonymous grid
   text; do not disguise it with width overrides or forced breaks.

## Editing and dates

Set `created_at` once; update `updated_at` for substantive saved prose or figure
revisions, not housekeeping or execution alone. Use quoted UTC datetimes
(`YYYY-MM-DDTHH:mm:ssZ`). Correct evidence and availability/version tags when invalidated.
