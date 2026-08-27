#import "/.demolab/lib.typ": data-json, data-image
#import "run-inputs.typ": data-file, inputs-ready, pending-report
#let data-file = data-file.with(article: "exp083")

#let meta = (
  title: "When does the default PING circuit start to sing?",
  date: "2026-08-17",
  description: "A fixed SNNLANG graph tests whether the default PING component develops a reproducible gamma rhythm as homogeneous Poisson drive increases.",
  collection: "demo",
  order: 9,
)

#let inputs = ("exp083",)
#let preview-figures = (
  (path: "exp083/network.svg", label: "network"),
  (path: "exp083/response.png", label: "response"),
  (path: "exp083/representative_rasters.png", label: "representative rasters"),
  (path: "exp083/spectra.png", label: "spectra"),
)

// Keep calculations lazy: absent inputs never become fabricated results.
#let render-report(data-file) = [
#let r = data-json(data-file("exp083/numbers.json"))
#let active = r.conditions.filter(row => row.frequency_resolved_fraction == 1)
#let transition = r.conditions.filter(row => row.input_rate_hz == 50).first()
#let first-active = active.first()
#let last-active = active.last()

#let body = [
  == Abstract

  The default `snn.components.ping()` circuit is silent at weak drive, then switches abruptly into a strongly rhythmic state near #transition.input_rate_hz Hz per input channel. The dominant rhythm rises from #calc.round(transition.rhythm_frequency_median_hz, digits: 2) to #calc.round(last-active.rhythm_frequency_median_hz, digits: 2) Hz across the active sweep and remains predominantly below the conventional gamma band. One compiled graph produces the entire response curve. No model parameter is tuned between conditions.

  == Methods

  + *Author one graph.* A typed #(r.config.n_input)-channel spike input drives the default PING component, which contains #r.config.n_e excitatory (E) and #r.config.n_i inhibitory (I) cells. The timestep is #r.config.dt_ms ms. Neuron, synapse, delay, initializer, and constraint specifications remain at their component defaults.

  + *Vary only input drive.* Homogeneous Poisson input spans #(r.config.rates_hz.map(value => str(value) + " Hz").join(", ")) per channel. Every condition uses the same compiled graph and #r.config.trials paired deterministic trials, each lasting #r.config.t_ms ms. Measurements exclude the first #r.config.burn_ms ms.

  + *Resolve the dominant rhythm.* The #raw(r.frequency_analysis.name) policy computes a single-trial Welch spectrum from the E-population raster. It searches #r.frequency_analysis.band_hz.at(0)--#r.frequency_analysis.band_hz.at(1) Hz, interpolates the peak between bins, rejects maxima at the band edge, and requires peak power to exceed #r.frequency_analysis.min_prominence_ratio times the band median. If a peak at half the selected frequency carries at least #calc.round(r.frequency_analysis.subharmonic_ratio * 100)% of its power, the lower frequency is reported. This prevents a strong harmonic from masquerading as the population rhythm.

  + *Score rhythmicity.* The standard exp054 metric bins the E-population spike count at 1 ms, computes its normalized autocorrelogram, and applies a three-point smoothing kernel. If $L$ is the first side-lobe height and $T$ the following trough height, the dimensionless Michelson contrast is $R = (L - T) / (L + T)$. Silence is reported as $R = 0$.

  + *Measure E/I timing separately.* Post-transient population spikes are binned at 1 ms. The lag of the strongest E/I cross-correlation within plus or minus 20 ms is descriptive and does not enter the gamma-resolution rule.

  #figure(
    data-image(data-file("exp083/network.svg"),
      width: 82%,
      alt: "Compiled SNNLANG graph with one typed spike input driving the default PING component.",
    ),
    caption: [
      Compiled graph reused at every input condition. A time-by-batch-by-#r.config.n_input spike tensor drives one default PING component with #r.config.n_e E cells and #r.config.n_i I cells. Only the input event probability changes across the sweep; the graph and its parameters remain fixed.
    ],
  )

  == Results

  The network is silent through #r.conditions.at(1).input_rate_hz Hz per channel. At #transition.input_rate_hz Hz it activates, the median rhythmicity score reaches #calc.round(transition.rhythmicity_score_median, digits: 2), and the dominant rhythm is #calc.round(transition.rhythm_frequency_median_hz, digits: 2) Hz. Rhythmicity therefore appears discontinuously, while the rhythm frequency increases smoothly with drive. The default circuit sings, but under this protocol it mostly sings below gamma.

  #figure(
    data-image(data-file("exp083/response.png"),
      width: 100%,
      alt: "Population firing rates, lobe-trough rhythmicity contrast, and dominant rhythm frequency across homogeneous input drive.",
    ),
    caption: [
      Response of one fixed default PING graph to homogeneous Poisson drive. Top: mean E and I firing rates in hertz, with standard deviations across #r.config.trials paired trials. Middle: median lobe-trough rhythmicity contrast $R$, with error bars showing half the interquartile range; $R = 0$ is flat and $R$ approaches 1 as periodic structure strengthens. Bottom: median dominant E-population frequency in hertz after subharmonic correction; the grey region marks 5--30 Hz. Rhythmicity switches on abruptly at #transition.input_rate_hz Hz per channel, while the dominant rhythm increases smoothly from #calc.round(transition.rhythm_frequency_median_hz, digits: 2) Hz.
    ],
  )

  #figure(
    data-image(data-file("exp083/representative_rasters.png"),
      width: 100%,
      alt: "Stacked excitatory and inhibitory rasters at three input rates selected before the primary sweep.",
    ),
    caption: [
      Single-trial E and I rasters at the three input rates fixed before the primary sweep. Time is in milliseconds; E cells are black, I cells are red, and the dashed vertical line marks the #r.config.burn_ms ms transient exclusion. The right labels report input rate per channel and post-transient mean population rates in hertz. The #r.representative_rates_hz.first() Hz condition is silent, whereas the higher-drive conditions show repeated population volleys.
    ],
  )

  #figure(
    data-image(data-file("exp083/spectra.png"),
      width: 100%,
      alt: "Mean excitatory population spectra at three input rates selected before the primary sweep.",
    ),
    caption: [
      Mean E-population power spectra at the preselected input rates. Frequency is in hertz and power is normalized to each curve's maximum for shape comparison; frequency resolution uses the unnormalized single-trial spectra. The grey region marks the former 30--80 Hz gamma-only window. The dominant peaks sit below it, while visible harmonics fall inside it, explaining why a gamma-restricted search gave a misleading frequency curve.
    ],
  )
]
#body
]

#let body = if inputs-ready(data-file, inputs) {
  render-report(data-file)
} else {
  pending-report(
    data-file, inputs,
    [Under which input conditions does the default PING circuit produce gamma? Inspect its topology, response, rasters, and spectra.],
    preview-figures, json-inputs: ("exp083",),
  )
}
