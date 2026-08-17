#let meta = (
  title: "When does the default PING circuit start to sing?",
  date: "2026-08-17",
  description: "A fixed SNNLANG graph tests whether the default PING component develops a reproducible gamma rhythm as homogeneous Poisson drive increases.",
  collection: "snnlang",
  status: "draft",
  order: 9,
)

#let r = json("/artifacts/data/exp083/numbers.json")
#let active = r.conditions.filter(row => row.gamma_resolved_fraction == 1)
#let transition = r.conditions.filter(row => row.input_rate_hz == 50).first()
#let first-active = active.first()
#let last-active = active.last()

#let body = [
  == Abstract

  The default `snn.components.ping()` circuit is silent at weak drive, begins firing near #transition.input_rate_hz Hz per input channel, and settles into reproducible gamma activity from #first-active.input_rate_hz to #last-active.input_rate_hz Hz. Across that resolved interval, the median gamma frequency rises from #calc.round(first-active.gamma_frequency_median_hz, digits: 2) to #calc.round(last-active.gamma_frequency_median_hz, digits: 2) Hz. One compiled graph produces the entire response curve. No model parameter is tuned between conditions.

  == Methods

  + *Author one graph.* A typed #(r.config.n_input)-channel spike input drives the default PING component, which contains #r.config.n_e excitatory (E) and #r.config.n_i inhibitory (I) cells. The timestep is #r.config.dt_ms ms. Neuron, synapse, delay, initializer, and constraint specifications remain at their component defaults.

  + *Vary only input drive.* Homogeneous Poisson input spans #(r.config.rates_hz.map(value => str(value) + " Hz").join(", ")) per channel. Every condition uses the same compiled graph and #r.config.trials paired deterministic trials, each lasting #r.config.t_ms ms. Measurements exclude the first #r.config.burn_ms ms.

  + *Resolve gamma consistently.* The #raw(r.gamma_frequency.name) policy computes a single-trial Welch spectrum from the E-population raster. It searches #r.gamma_frequency.band_hz.at(0)--#r.gamma_frequency.band_hz.at(1) Hz, interpolates the peak between bins, rejects maxima at the band edge, and requires peak power to exceed #r.gamma_frequency.min_prominence_ratio times the band median. Failed checks retain an unresolved reason; they are not reported as 0 Hz.

  + *Measure E/I timing separately.* Post-transient population spikes are binned at 1 ms. The lag of the strongest E/I cross-correlation within plus or minus 20 ms is descriptive and does not enter the gamma-resolution rule.

  #figure(
    image(
      "/artifacts/data/exp083/network.svg",
      width: 82%,
      alt: "Compiled SNNLANG graph with one typed spike input driving the default PING component.",
    ),
    caption: [
      Compiled graph reused at every input condition. A time-by-batch-by-#r.config.n_input spike tensor drives one default PING component with #r.config.n_e E cells and #r.config.n_i I cells. Only the input event probability changes across the sweep; the graph and its parameters remain fixed.
    ],
  )

  == Results

  The network is silent through #r.conditions.at(1).input_rate_hz Hz per channel. At #transition.input_rate_hz Hz it activates, but gamma resolves in only #calc.round(transition.gamma_resolved_fraction * 100)% of trials. From #first-active.input_rate_hz Hz onward, every trial resolves. Both population rates and gamma frequency then increase with input drive. The untouched default component therefore has a broad gamma regime, but firing begins before the spectral criterion becomes reliable across trials.

  #figure(
    image(
      "/artifacts/data/exp083/response.png",
      width: 100%,
      alt: "Population firing rates, resolved gamma fraction, and gamma peak frequency across homogeneous input drive.",
    ),
    caption: [
      Response of one fixed default PING graph to homogeneous Poisson drive. Left: mean E and I firing rates in hertz, with standard deviations across #r.config.trials paired trials. Middle: fraction of trials passing the registered gamma-resolution rule. Right: median resolved E-population frequency in hertz; the grey region marks the registered #r.gamma_frequency.band_hz.at(0)--#r.gamma_frequency.band_hz.at(1) Hz search band. Activity begins at #transition.input_rate_hz Hz per channel, but resolution becomes reproducible only at #first-active.input_rate_hz Hz.
    ],
  )

  #figure(
    image(
      "/artifacts/data/exp083/representative_rasters.png",
      width: 100%,
      alt: "Stacked excitatory and inhibitory rasters at three input rates selected before the primary sweep.",
    ),
    caption: [
      Single-trial E and I rasters at the three input rates fixed before the primary sweep. Time is in milliseconds; E cells are black, I cells are red, and the dashed vertical line marks the #r.config.burn_ms ms transient exclusion. The right labels report input rate per channel and post-transient mean population rates in hertz. The #r.representative_rates_hz.first() Hz condition is silent, whereas the higher-drive conditions show repeated population volleys.
    ],
  )

  #figure(
    image(
      "/artifacts/data/exp083/spectra.png",
      width: 100%,
      alt: "Mean excitatory population spectra at three input rates selected before the primary sweep.",
    ),
    caption: [
      Mean E-population power spectra at the preselected input rates. Frequency is in hertz and power is normalized to each curve's maximum for shape comparison; gamma resolution uses the unnormalized single-trial spectra. The grey region marks #r.gamma_frequency.band_hz.at(0)--#r.gamma_frequency.band_hz.at(1) Hz. Stronger drive shifts the dominant resolved rhythm upward within the registered band.
    ],
  )
]
