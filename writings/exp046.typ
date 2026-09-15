#import "templates/article-layout.typ": journal-article
#import "templates/result-card.typ": result-figure-ref, result-card, with-result-sections
#import "/.demolab/lib.typ": data-json, data-image
#import "templates/dataset.typ": data-file, inputs-ready, pending-report, run-view, input-assets
#import "templates/abstract.typ": journal-abstract
#import "templates/methods.typ": journal-methods
#let data-file = data-file.with(article: "exp046")

#let meta = (
  tags: ("data", "v36.0.0"),
  title: "One Spike per Gamma Cycle",
  created_at: "2026-06-04T00:00:00Z",
  updated_at: "2026-09-14",
  description: "Pooled and equally weighted network distributions support predominantly one-spike gamma-cycle participation across 18 trained networks, with exceptions.",
  collection: "gamma-gated-sparsity",
)

#let inputs = ("exp046",)
#let preview-figures = (
  (path: "exp046/spikes_per_cycle_distribution_equal_network.svg", label: "equal-network spikes per cycle distribution"),
  (path: "exp046/ceiling_vs_fgamma.svg", label: "ceiling vs fgamma"),
)

// Keep calculations lazy: absent inputs never become fabricated results.
#let render-report(data-file) = [
#let body = [
  #journal-abstract(body: [
  We asked whether excitatory firing reflects participation in gamma cycles
  without repeated spikes. We reused the inhibitory-timescale sweep's spike
  recordings, comparing pooled observations with equally weighted network
  distributions.

  Excitatory neurons were usually silent within a cycle and, when active,
  predominantly emitted a single spike. Both averages support this approximation
  across the sweep. Repeated spikes and above-reference firing rates preclude
  a strict one-spike ceiling in these measurements; no universal participation
  law is established.
  ])

  == Results

  #with-result-sections[

  #result-card[
  === Spikes per neuron-cycle

  Across the balanced 18-network design, the equal-network mean assigned 76.35% of pairs
  to zero spikes, 22.09% to one spike and 1.56% to two or more spikes. Thus,
  98.44% contained at most one spike
  (#result-figure-ref(<fig:exp046-result-1>)).

  #figure(
    data-image(data-file("exp046/spikes_per_cycle_distribution_equal_network.svg"), width: 100%,
      alt: "Six black bar charts in one row, one per τ_GABA, showing equal-network mean fractions for 0, 1, 2, or ≥3 excitatory spikes per neuron and gamma cycle."),
    caption: [Equal-network distributions of excitatory spikes per neuron–cycle
      at $tau_"GABA"$ values *(A–F)* 4.5, 6, 9, 12, 18 and 27 ms. Each
      separately trained network's counts of pairs with 0, 1, 2 or ≥3 spikes
      were normalized to sum to one; black bars are arithmetic means across
      the three networks per condition, irrespective of their detected
      cycle totals. These distributions reuse epoch-50 spike recordings from
      the inhibitory-timescale sweep. Presentations without detected inhibitory
      bursts are excluded; no uncertainty interval is shown.],
  ) <fig:exp046-result-1>

  For comparison, opportunity pooling across 167,178,240 neuron–cycle pairs
  assigned 75.24% of pairs to zero spikes and 23.62% to one spike. Two-or-more
  events occurred in 1.15% of pairs and three-or-more in 0.10%;
  pooled over the sweep, 98.85% contained at most one spike. Among active
  pairs, 95.37% contained exactly one spike, declining from 98.93% at 4.5-ms
  inhibitory decay to 83.78% at 27 ms. Single-spike participation predominated,
  but its frequency depended on the decay condition.

  Giving each trained network equal weight within its decay condition changed
  each spike-count fraction by less than 0.20 percentage points relative to
  opportunity pooling; the largest shift was 0.191 percentage points in the
  one-spike fraction at $tau_"GABA" = 27$ ms. Pooling samples a neuron–cycle
  pair uniformly, whereas equal-network averaging first samples a network
  uniformly. The latter also balances the six decay conditions; pooling gives
  more weight to the faster rhythms that produced more cycles.

  ]

  #result-card[
  === Rate versus gamma frequency

  Maximum per-neuron excitatory firing rate, $r_(E,"max")$, scaled with
  spectral-peak frequency, $f_gamma$, with through-origin fit
  $r_(E,"max") = 0.977 f_gamma$ and coefficient of determination
  $R_"fit"^2 = 0.915$ (#result-figure-ref(<fig:exp046-result-2>)). Both
  rates are in hertz. Four of the 18 network maxima exceeded the
  one-spike-per-cycle reference, including 80.97 Hz at 67.21 Hz. Proximity
  of the fitted slope to one therefore does not establish a strict ceiling.

  #figure(
    data-image(data-file("exp046/ceiling_vs_fgamma.svg"), width: 100%,
      alt: "Maximum and median excitatory firing rates against spectral-peak frequency for 18 networks, including four maxima above the one-spike-per-cycle reference."),
    caption: [Per-neuron E rate versus measured gamma frequency $f_gamma$
      across the $tau_"GABA"$ sweep. Triangles and circles show each network's
      maximum and median neuron rates, respectively, averaged over all test
      presentations; colours group decay conditions. The dashed line is
      $r_E = f_gamma$, where $r_E$ is per-neuron E rate. The dotted
      $0.20 f_gamma$ line is a fixed visual reference, not a fitted
      participation estimate. Neither line is the fitted maximum-rate model.],
  ) <fig:exp046-result-2>

  ]
  ]

  #journal-methods(
    orientation: [
  We reused spike recordings from the epoch-50 checkpoints of the
  inhibitory-timescale sweep: six decay conditions and three independently
  trained networks per condition (seeds 42–44). The equal-network comparison
  reanalysed these recordings without retraining or new inference.
    ],
    compute: [
  + *Evaluate fixed test presentations.* The reused evaluation presented the
    same fixed 1,000-image subset of the official MNIST test partition to each
    network for 200 ms per image, with a 0.1-ms timestep. Recordings contained
    spikes from all 1,024 excitatory and 256 inhibitory neurons and mean
    per-neuron excitatory firing rates over the complete evaluation.
    ],
    analyse: [
  #set enum(start: 2)

  + *Detect inhibitory bursts.* We smoothed inhibitory population spike counts
    with a unit-sum Gaussian of 1-ms standard deviation, extending ±4 ms.
    Peaks reached at least 5% of each presentation's maximum smoothed count
    and were separated by at least half the network's spectral period,
    rounded down to simulation steps. We reused $f_gamma$, the interpolated
    5–150-Hz peak of each network's trial-averaged excitatory Welch spectrum.
  + *Define cycles.* Boundaries were integer midpoints between consecutive
    inhibitory peaks, rounded down; edge intervals extended to presentation
    boundaries and could contain partial cycles. A single burst defined one
    full-presentation interval. Presentations without detected bursts were
    excluded from cycle counts, but not whole-presentation firing rates.
  + *Count excitatory spikes.* We counted each neuron's spikes separately in
    each interval, including its starting step and excluding its ending step.
    Inhibitory bursts supplied the cycle anchor even when an excitatory neuron
    remained silent.
    ],
    present: [
  #set enum(start: 5)

  + *Estimate opportunity and network distributions.* We bucketed spike counts
    into ${0, 1, 2, >= 3}$. For the opportunity-pooled distribution, we summed
    bucket counts across neurons, cycles and networks before normalization. For
    the equal-network distribution, we normalized each network's bucket counts
    separately and then took their arithmetic mean, giving the three networks
    equal weight within each $tau_"GABA"$ condition and all 18 networks equal
    weight in the across-sweep summary. A network with no detected cycles would
    have an undefined distribution and abort this analysis; none did.
    Active-pair fractions excluded zero-spike pairs. Neurons and cycles were
    not independent training replicates; no uncertainty interval was estimated
    for the network distributions.
  + *Compare rates and frequency.* We fitted maximum per-neuron E rate against
    $f_gamma$ by least squares through the origin across all 18 networks,
    weighting networks equally. The reported $R_"fit"^2$ used the centred total
    sum of squares. Median-neuron rates and the fixed reference lines were
    displayed separately from this fit.
    ],
  )
]
#body
  #run-view("exp046", inputs)

]

#let report-body = if inputs-ready(data-file, inputs) {
  render-report(data-file)
} else {
  pending-report(
    data-file, inputs,
    [How often does an excitatory neuron spike within one gamma cycle? Count spikes between inhibitory volleys across the inhibitory-timescale sweep.],
    preview-figures, json-inputs: (),
  )
}

#let meta = meta + (assets: input-assets("exp046", inputs))
#let body = journal-article("exp046", inputs, report-body, dataset-placed: inputs-ready(data-file, inputs))
