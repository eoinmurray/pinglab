---
title: Parameter Tables
author: Eoin Murray
collection: notes
description: The required compact summary, connectivity matrix and numeric conventions.
created_at: "2026-10-29T17:42:34Z"
updated_at: "2026-10-29T17:55:32Z"
---

# Parameter tables

Read when creating or changing parameter tables. Evidence binding is defined in
[Writing Rules](/notes/rules/writing#parameter-table-data-binding).
Follow [exp133](/exp133#parameters): use the shared `Table` component for a compact
grouped summary immediately followed by a source-to-target connectivity matrix.
Do not add exhaustive or expandable parameter, pathway or analysis tables.

1. Combine matching E/I dynamics with paired values and units, explaining pair
   order and derived values. Keep a separate row for every population. Distinguish
   absent populations, inapplicable settings and unavailable evidence.
2. Separate population or connection names from descriptions with a space, not a
   colon: `E→I weights`, `Input→E synapse / delay`, `Output neurons`.

<a id="parameter-table-layout"></a>

## Required layout

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
   Include units inline beside each dimensional value in the matrix cells,
   including weights; caption-only units are insufficient. Derive values using the saved normalization,
   execution scaling and relevant scheduled gain. Distinguish nominal initializer
   parameters from measured weight statistics and total incoming conductance.
7. Write `Absent` only when no projection exists. A zero weight, temporarily zero
   gain or silent input does not make a projection absent. Explain excluded
   self-connections, exact fan-in, the gain phase represented and the selected
   condition when the matrix illustrates one point of a sweep.
8. Share a matrix across compared protocols only when their native connectivity
   and effective nominal parameter conventions agree; otherwise show separate
   labelled matrices. Check equivalence before display rounding. Keep all values
   tied to the same selected evidence as the figures, following [evidence binding](/notes/rules/writing#parameter-table-data-binding).
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
