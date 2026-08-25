#let meta = (
  title: "Can inhibitory recovery tune default PING into gamma?",
  date: "2026-08-17",
  description: "A one-parameter SNNLANG sweep tests whether inhibitory synaptic decay controls the default PING component's rhythm frequency.",
  collection: "demo",
  status: "ExpScout",
  order: 10,
)

#let r = json("/artifacts/data/exp084/numbers.json")
#let default = r.conditions.filter(row => row.tau_gaba_ms == 9).first()
#let fastest = r.conditions.first()
#let slowest = r.conditions.last()

#let body = [
  == Abstract

  At a fixed #r.config.input_rate_hz Hz/channel drive, shortening inhibitory synaptic decay from #slowest.tau_gaba_ms to #fastest.tau_gaba_ms ms accelerates the default SNNLANG PING rhythm from #calc.round(slowest.rhythm_frequency_median_hz, digits: 2) to #calc.round(fastest.rhythm_frequency_median_hz, digits: 2) Hz. Decays of #r.conditions.at(1).tau_gaba_ms ms and below enter the 30--80 Hz gamma band without weakening the standard rhythmicity score. This experiment varies one public component parameter while preserving the authored circuit, paired inputs, and analysis policy established in exp083.

  == Methods

  + *Fix the active operating point.* A typed #(r.config.n_input)-channel spike input drives #r.config.n_e excitatory (E) and #r.config.n_i inhibitory (I) cells at #r.config.input_rate_hz Hz per channel. Each condition reuses the same #r.config.trials deterministic input tensors and network seed. Trials last #r.config.t_ms ms; the first #r.config.burn_ms ms are excluded.

  + *Vary inhibitory recovery only.* The public `tau_gaba` argument of `snn.components.ping()` spans #(r.config.tau_gaba_ms.map(value => str(value) + " ms").join(", ")). The default is #default.tau_gaba_ms ms. Every other neuron, synapse, delay, initializer, and constraint specification remains unchanged.

  + *Apply the registered rhythm analyses.* The #raw(r.frequency_analysis.name) estimator reports the dominant E-population rhythm over #r.frequency_analysis.band_hz.at(0)--#r.frequency_analysis.band_hz.at(1) Hz, including its prominence and subharmonic rules. Rhythmicity is the standard autocorrelation lobe-trough contrast $R$ used in exp083. E/I lag is measured separately from the peak cross-correlation of 1 ms population counts.

  #figure(
    image(
      "/artifacts/data/exp084/network.svg",
      width: 82%,
      alt: "Compiled default PING graph at the default inhibitory decay condition.",
    ),
    caption: [
      Compiled SNNLANG graph at the #default.tau_gaba_ms ms default condition. Across the sweep, only the decay time attached to the I-to-E GABA projection changes. The input tensor, populations, weights, delays, and execution seeds are paired across conditions.
    ],
  )

  == Results

  Inhibitory decay produces a direct test of the textbook recovery-timescale account. The default #default.tau_gaba_ms ms circuit resolves at #calc.round(default.rhythm_frequency_median_hz, digits: 2) Hz. Faster inhibition shifts the rhythm to #calc.round(fastest.rhythm_frequency_median_hz, digits: 2) Hz, while slower inhibition yields #calc.round(slowest.rhythm_frequency_median_hz, digits: 2) Hz.

  #figure(
    image(
      "/artifacts/data/exp084/response.svg",
      width: 100%,
      alt: "Population rates, rhythmicity, and dominant frequency across inhibitory decay time.",
    ),
    caption: [
      Response to inhibitory synaptic decay at fixed #r.config.input_rate_hz Hz/channel input. Top: mean E and I firing rates with standard deviations across #r.config.trials paired trials. Middle: median standard rhythmicity contrast $R$, with half-interquartile-range error bars. Bottom: median dominant E-population frequency; the grey region marks 30--80 Hz gamma. The dashed line marks the #default.tau_gaba_ms ms component default.
    ],
  )

  #figure(
    image(
      "/artifacts/data/exp084/representative_rasters.png",
      width: 100%,
      alt: "Stacked excitatory and inhibitory rasters at fast, default, and slow inhibitory decay.",
    ),
    caption: [
      Single paired-trial E and I rasters at the preselected fast, default, and slow inhibitory decay conditions. Time is in milliseconds; E cells are black, I cells are red, and the dashed vertical line marks the transient exclusion. Right labels give inhibitory decay and the resolved dominant frequency.
    ],
  )
]
