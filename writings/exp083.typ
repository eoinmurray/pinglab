#let meta = (
  title: "When does the default PING circuit start to sing?",
  date: "2026-08-17",
  description: "A fixed SNNLANG graph tests whether the default PING component develops a reproducible gamma rhythm as homogeneous Poisson drive increases.",
  collection: "snnlang",
  status: "draft",
  order: 9,
)

#let r = json("/artifacts/data/exp083/numbers.json")
#let resolved = r.conditions.filter(row => row.gamma_resolved_fraction > 0)

#let body = [
  == Abstract

  This experiment asks a deliberately basic question about the reusable default `snn.components.ping()` circuit: as homogeneous Poisson drive increases, does it pass through a reproducible gamma regime? One compiled graph is held fixed across every condition. Only the input rate changes. The result is therefore both a first scientific example for SNNLANG and a test of what its advertised defaults actually do—without tuning them until the plot becomes photogenic.

  == Methods

  === Fixed authored graph

  #figure(
    image("/artifacts/data/exp083/network.svg", width: 86%, alt: "Compiled SNNLANG graph containing one spike input and the default excitatory-inhibitory PING component."),
    caption: [The single compiled graph reused at every drive condition.],
  )

  A 128-channel typed spike input drives one default PING component containing 80 excitatory and 20 inhibitory cells. The timestep is #r.config.dt_ms ms. Default neuron, synapse, delay, initializer, and constraint specifications are left untouched. The graph digest is #raw(r.graph.digest).

  === Registered drive sweep and measurements

  Homogeneous Poisson drive spans #(r.config.rates_hz.map(value => str(value) + " Hz").join(", ")) per input channel. Each condition contains #r.config.trials deterministic trials paired across rates, lasts #r.config.t_ms ms, and discards the first #r.config.burn_ms ms.

  Gamma frequency uses the named #raw(r.gamma_frequency.name) policy on the excitatory raster: a #r.gamma_frequency.band_hz.at(0)--#r.gamma_frequency.band_hz.at(1) Hz single-trial Welch spectrum, three-point sub-bin interpolation, rejection of band-edge maxima, and a peak-to-band-median power ratio greater than #r.gamma_frequency.min_prominence_ratio. Silence and failed quality gates remain unresolved rather than becoming 0 Hz. E--I lag is a separate descriptive cross-correlation measurement and does not decide whether gamma resolved.

  == Results

  === Exact responses at preselected drives

  #figure(
    image("/artifacts/data/exp083/representative_rasters.png", width: 92%, alt: "Excitatory and inhibitory spike rasters at three input rates fixed before the primary sweep."),
    caption: [Exact E and I rasters at the three rates fixed before inspecting the primary sweep. The dashed line marks the end of burn-in.],
  )

  === Drive-response curve

  #figure(
    image("/artifacts/data/exp083/response.png", width: 100%, alt: "Population rates, resolved gamma fraction, and resolved gamma frequency across input drive."),
    caption: [Population activation and the standardized gamma-frequency result across homogeneous drive. Error bars are standard deviations across paired trials.],
  )

  #if resolved.len() == 0 [
    No drive condition produced a resolved gamma peak under the registered default policy. The default component may activate, but this sweep provides no evidence that it enters a reproducible gamma regime.
  ] else [
    The standardized estimator resolved at least one trial in #resolved.len() of #r.conditions.len() drive conditions. Resolution fraction, rather than the existence of a numerical spectral maximum, determines where the evidence supports rhythmic activity.
  ]

  === Spectral shape

  #figure(
    image("/artifacts/data/exp083/spectra.png", width: 78%, alt: "Mean excitatory population spectra at the three preselected input rates."),
    caption: [Mean E-population spectra at the preselected low, transitional, and high drives. Curves are peak-normalized for shape comparison; resolution uses unnormalized power.],
  )

  == Interpretation

  The experiment is intentionally allowed to disappoint. A robust resolved interval supports the default component as a useful out-of-the-box PING regime. Activation without resolution says the defaults produce spikes but not convincing gamma under this protocol. Abrupt, seed-sensitive, or saturated responses instead identify where the reusable defaults need revision. All three outcomes are more useful than tuning in secret and publishing the prettiest survivor.
]
