#import "contents.typ": contents-here, with-contents, result-card, with-numbered-equations, with-result-sections
#import "/.demolab/lib.typ": data-json, data-image
#import "run-inputs.typ": data-file, inputs-ready, pending-report
#import "run-view.typ": with-datasets, run-view
#import "run-inputs.typ": input-assets
#let data-file = data-file.with(article: "exp046")

#let meta = (
  tags: ("data", "v35.0.0"),
  title: "One Spike per Gamma Cycle",
  created_at: "2026-06-04T00:00:00Z",
  updated_at: "2026-08-31T00:00:00Z",
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

  Asked whether exp041's rate–frequency relationship reflects excitatory neurons
  participating in gamma cycles without repeatedly firing. Reused the
  inhibitory-timescale sweep and counted each excitatory neuron's spikes between
  inhibitory population bursts.

  Excitatory neurons were usually silent within a cycle and, when active,
  overwhelmingly emitted a single spike; the busiest cells tracked that ceiling.
  Supports the one-spike-per-cycle approximation for this sweep, but does not
  turn the population relationship into a universal participation law.

  #contents-here()

  == Results

  #with-result-sections[

  #result-card[
  === Spike-count distribution per neuron and gamma cycle

  Across 179 million neuron–cycle pairs, E neurons emitted zero spikes in
  approximately 79% of cycles and one spike in approximately 20%. Two-or-more
  events occurred in approximately 1.1% of pairs and three-or-more in 0.14%;
  pooled over the sweep, 98.9% contained at most one spike.

  #figure(
    data-image(data-file("exp046/spikes_per_cycle_distribution.svg"), width: 100%,
      alt: "Six bar charts, one per τ_GABA, of the probability an E neuron emits 0, 1, 2, or ≥3 spikes in a gamma cycle; every panel is dominated by the 0 and 1 bars."),
    caption: [Distribution of E spike count per gamma cycle per neuron at
      $tau_"GABA"$ values *(A–F)* 4.5, 6, 9, 12, 18 and 27 ms, respectively,
      aggregating three seeds and 179 million neuron–cycle pairs.],
  )

  ]

  #result-card[
  === Per-neuron firing rate against gamma frequency

  The busiest neuron in each network tracked the one-spike-per-cycle ceiling
  $r = f_gamma$ (fit $r = 0.97 f_gamma$, $R_"fit"^2 = 0.88$), whereas the
  median neuron followed the shallower $r approx 0.20 f_gamma$ participation
  slope from exp041. Even the most active neuron rarely exceeded one spike per
  cycle, making the ceiling near-strict in these measurements.

  #figure(
    data-image(data-file("exp046/ceiling_vs_fgamma.svg"), width: 100%,
      alt: "Per-neuron E rate against measured gamma frequency; the busiest neuron tracks the one-spike-per-cycle line while the median neuron tracks exp041's shallower slope."),
    caption: [Per-neuron E rate versus measured gamma frequency $f_gamma$
      across the $tau_"GABA"$ sweep. Curves show the busiest and median neurons
      in each network, the one-spike-per-cycle reference and the exp041
      participation slope.],
  )

  ]
  ]

  == Methods

  Cycle statistics were evaluated from the final-epoch checkpoints used by exp041, so this experiment audits the same endpoint gamma dynamics. The source training horizon was read from those checkpoint configurations and recorded in the generated provenance.

  For each of exp041's 18 trained networks (6 $tau_"GABA"$ × 3 seeds):

  === Compute

  + *Run fixed-trial inference.* We ran inference on the fixed 1,000-image subset of the official MNIST test partition; we captured per-trial $(T, B, N_E)$ and $(T, B, N_I)$ spike tensors.

  === Analyse

  #set enum(start: 2)

  + *Detect inhibitory bursts.* We detected I-burst times per trial: we smoothed the population I rate with a 1-ms Gaussian, and used scipy peak detection with min-distance set to half the network's own $1 \/ f_gamma$.
  + *Define cycles.* Cycle boundaries were the midpoints between consecutive I-burst peaks (the first cycle started at $t = 0$ and the last ended at trial end).
  + *Count excitatory spikes.* For each (neuron, cycle, trial), we counted the number of E spikes within the cycle window.

  === Present

  #set enum(start: 5)

  + *Aggregate displayed counts.* We bucketed counts globally into ${0, 1, 2, >= 3}$ and aggregated by $tau_"GABA"$.

  The cycle anchor is the I-burst: this is the right anchor because the cycle is operationally defined as _"the time between one inhibitory blanket and the next"_, not as the time between E bursts (which can be silent on a given cycle).

]
#body
  #run-view("exp046", inputs)

]

#let report-body = if inputs-ready(data-file, inputs) {
  render-report(data-file)
} else {
  pending-report(
    data-file, inputs,
    [How often does an excitatory neuron spike within one gamma cycle? Count spikes between inhibitory volleys across the exp041 inhibitory-timescale sweep.],
    preview-figures, json-inputs: (),
  )
}

#let meta = meta + (assets: input-assets("exp046", inputs))
#let body = with-datasets("exp046", inputs, report-body, placed: inputs-ready(data-file, inputs))
#let body = with-numbered-equations(body)
#let body = with-contents(body)
