#import "contents.typ": with-contents
#import "/.demolab/lib.typ": data-json, data-image
#import "run-inputs.typ": data-file, inputs-ready, pending-report
#import "run-view.typ": with-datasets, run-view
#import "run-inputs.typ": input-assets
#let data-file = data-file.with(article: "exp046")

#let meta = (
  status: "[▦ DATA]",
  title: "One Spike per Gamma Cycle",
  date: "2026-06-04",
  updated_at: "2026-08-28",
  description: "Counting E spikes per gamma cycle across exp041's 18 checkpoints shows the architecture is overwhelmingly one-spike-per-cycle.",
  collection: "gamma-gated-sparsity",
)

#let inputs = ("exp046",)
#let preview-figures = (
  (path: "exp046/spikes_per_cycle_distribution.svg", label: "spikes per cycle distribution"),
  (path: "exp046/ceiling_vs_fgamma.svg", label: "ceiling vs fgamma"),
)

// Keep calculations lazy: absent inputs never become fabricated results.
#let render-report(data-file) = [
#let body = [
  == Abstract

  #link("/exp041/")[exp041]'s slope $p approx 0.20$ was interpreted as _"each E cell joins a cycle with ≈ 20% probability"_. That reading assumes the per-cycle spike count is bounded by 1 (an E cell either participates in this cycle or not). This notebook measures that directly: walking through every gamma cycle in the shared 1,000-image held-out evaluation subset and counting how many spikes each E cell actually emits, on all 18 trained checkpoints in exp041's $tau_"GABA"$ sweep.

  #run-view("exp046", inputs)

  == Results: Silent cell-cycles dominate; repeated spikes within a cycle are rare

  #figure(
    data-image(data-file("exp046/spikes_per_cycle_distribution.svg"), width: 100%,
      alt: "Six bar charts, one per τ_GABA, of the probability an E cell emits 0, 1, 2, or ≥3 spikes in a gamma cycle; every panel is dominated by the 0 and 1 bars."),
    caption: [Distribution of E spike count per gamma cycle per cell, by $tau_"GABA"$, three seeds aggregated. Across *179 million (cell, cycle) pairs*, the architecture is overwhelmingly bimodal: each cell either emits zero spikes in a given cycle (≈ 79% of the time) or exactly one (≈ 20% of the time). Two-or-more events occur in ≈ 1.1% of cycles; three-or-more in ≈ 0.14%. Pooled over the sweep, 98.9% of pairs carry at most one spike.],
  )

  #figure(
    data-image(data-file("exp046/ceiling_vs_fgamma.svg"), width: 100%,
      alt: "Per-cell E rate against measured gamma frequency; the busiest cell tracks the one-spike-per-cycle line while the median cell tracks exp041's shallower slope."),
    caption: [Per-cell E rate versus measured gamma frequency $f_gamma$ across the $tau_"GABA"$ sweep. The busiest cell in each network tracks the one-spike-per-cycle ceiling $r = f_gamma$ (max-cell fit $r = 0.97 f_gamma$, $R^2 = 0.88$), while the median cell sits on exp041's shallower $r approx 0.20 f_gamma$ participation slope. The ceiling is near-strict: even the most active cell rarely exceeds one spike per cycle.],
  )

  == Methods

  Cycle statistics are evaluated from the final-epoch checkpoints used by exp041, so this experiment audits the same endpoint gamma dynamics. The source training horizon is read from those checkpoint configurations and retained in the generated provenance.

  For each of exp041's 18 trained cells (6 $tau_"GABA"$ × 3 seeds):

  + Run inference on the fixed 1,000-image subset of the official MNIST test partition; capture per-trial $(T, B, N_E)$ and $(T, B, N_I)$ spike tensors.
  + Detect I-burst times per trial: smooth the population I rate with a 1-ms Gaussian, run scipy peak detection with min-distance set to half the cell's own $1 \/ f_gamma$.
  + Cycle boundaries are the midpoints between consecutive I-burst peaks (first cycle starts at $t = 0$, last ends at trial end).
  + For each (cell, cycle, trial), count the number of E spikes within the cycle window.
  + Bucket counts globally into ${0, 1, 2, >= 3}$ and aggregate by $tau_"GABA"$.

  The cycle anchor is the I-burst: this is the right anchor because the cycle is operationally defined as _"the time between one inhibitory blanket and the next"_, not as the time between E bursts (which can be silent on a given cycle).

]
#body
]

#let report-body = if inputs-ready(data-file, inputs) {
  render-report(data-file)
} else {
  pending-report(
    data-file, inputs,
    [How often does an excitatory cell spike within one gamma cycle? Count spikes between inhibitory volleys across the exp041 inhibitory-timescale sweep.],
    preview-figures, json-inputs: (),
  )
}

#let meta = meta + (assets: input-assets("exp046", inputs))
#let body = with-datasets("exp046", inputs, report-body, placed: inputs-ready(data-file, inputs))
#let body = with-contents(body)
