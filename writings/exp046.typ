#import "/demolab-engine/build/lib.typ": numbers-table, provenance-footer

#let meta = (
  title: "Near-strict 1-spike-per-cycle ceiling: 99.45% of (cell, cycle) pairs",
  date: "2026-06-04",
  description: "Counting E spikes per gamma cycle across exp041's 18 checkpoints shows the architecture is overwhelmingly one-spike-per-cycle.",
  collection: "gamma-gated-sparsity",
  status: "draft",
)


#let body = [
  == Abstract

  #link("/exp041/")[exp041]'s slope $p approx 0.20$ was interpreted as _"each E cell joins a cycle with ≈ 20% probability"_. That reading assumes the per-cycle spike count is bounded by 1 — an E cell either participates in this cycle or not. This notebook measures that directly: walking through every gamma cycle on the test set and counting how many spikes each E cell actually emits, on all 18 trained checkpoints in exp041's $tau_"GABA"$ sweep.

  == Methods

  For each of exp041's 18 trained cells (6 $tau_"GABA"$ × 3 seeds):

  + Run inference on the MNIST test set; capture per-trial $(T, B, N_E)$ and $(T, B, N_I)$ spike tensors.
  + Detect I-burst times per trial: smooth the population I rate with a 1-ms Gaussian, run scipy peak detection with min-distance set to half the cell's own $1 \/ f_gamma$.
  + Cycle boundaries are the midpoints between consecutive I-burst peaks (first cycle starts at $t = 0$, last ends at trial end).
  + For each (cell, cycle, trial), count the number of E spikes within the cycle window.
  + Bucket counts globally into ${0, 1, 2, >= 3}$ and aggregate by $tau_"GABA"$.

  The cycle anchor is the I-burst — this is the right anchor because the cycle is operationally defined as _"the time between one inhibitory blanket and the next"_, not as the time between E bursts (which can be silent on a given cycle).

  == Results

  #figure(
    block(width: 100%, height: 4cm, inset: 1em, stroke: 0.5pt + gray, radius: 3pt, fill: luma(245))[#text(fill: gray)[pending re-run with new canonical data]],
    caption: [Distribution of E spike count per gamma cycle per cell, by $tau_"GABA"$, three seeds aggregated. Across *48 million (cell, cycle) pairs*, the architecture is overwhelmingly bimodal: each cell either emits zero spikes in a given cycle (≈ 77% of the time) or exactly one (≈ 22% of the time). Two-or-more events occur in 0.55% of cycles; three-or-more in 0.04%.],
  )
]
