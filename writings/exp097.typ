#let meta = (
  title: "Can a PING cycle be seen as a running engine?",
  date: "2026-08-23",
  description: "Tests whether recurrent excitatory and inhibitory conductances trace a coherent simulated PING cycle.",
  collection: "snnlang",
  status: "complete",
  order: 11,
)

#let r = json("/artifacts/data/exp097/numbers.json")
#let result = r.results
#let loop-video(src) = context {
  if target() == "html" {
    html.elem("video", attrs: (src: src, controls: "", loop: "", playsinline: "", style: "max-width:100%;width:100%"))[]
  } else {
    text(size: 9pt, style: "italic", fill: gray)[[Video — view the web edition to play.]]
  }
}

#let body = [
  == Abstract

  Pyramidal–interneuron gamma (PING) is often described as alternating excitatory and inhibitory volleys. This scout asks whether the two recurrent conductances form a coherent state portrait of that cycle. Across five simulated trials, the conductance pair repeatedly traced an oriented loop at gamma frequency. The result supports an engine-like visualization of this simulated operating point without claiming that two conductances fully describe the network.

  The engine picture shows the rhythm clearly, but it does not contain the whole machine.

  == The proposed engine

  The frozen scout used one 80-excitatory, 20-inhibitory conductance-based PING network at its active gamma operating point. Five predeclared Poisson input realizations drove the same network. Each simulated trial lasted 500 ms, and the first 100 ms was excluded as settling time.

  The engine has two main displayed state variables:

  $ g_E(t) = "mean conductance of the recurrent E→I projection", $

  $ g_I(t) = "mean conductance of the recurrent I→E projection". $

  These are separate target-local conductances, not a signed quantity moving between two reservoirs. E spikes increase excitatory conductance onto I cells. I spikes increase inhibitory conductance onto E cells. Each conductance then decays locally. Their ordered rise and fall produces the engine-like cycle.

  == 1. Does the conductance pair form a cycle?

  A useful engine portrait should return through the same joint states in the same order. The frozen analysis aligned every complete post-transient cycle to its E-population volley, then traced $(g_E, g_I)$ through time. It measured trajectory orientation, enclosed area, cycle duration, and E-to-I volley lag.

  *Expected patterns.* A coherent PING cycle should raise $g_E$ after an E volley, recruit an I volley, and then raise $g_I$ while E activity is suppressed. Repeated cycles should traverse a bounded loop in one direction. A collapsed line, inconsistent orientation, or broad cloud would weaken the two-conductance account.

  *Planned visual evidence.* The trial nearest the median rhythm frequency supplies five complete cycles. Equal-sized conductance and voltage instruments, simulated traces, volley marks, and a moving phase-plane point share one clock. Screen positions are normalized, but the video reports biological time.

  *Simulation result.* The five trials supplied #result.cycles_total complete cycles: #(result.cycles_per_trial.map(str).join(", ")) per trial. Every cycle moved #result.modal_orientation through the conductance plane. The median period was #calc.round(result.median_period_ms, digits: 1) ms, or #calc.round(result.median_frequency_hz, digits: 1) Hz. The median signed loop area was #calc.round(result.median_signed_area_uS2, digits: 3) $mu S^2$. The conductance pair therefore forms a coherent cycle at this operating point.

  #figure(
    loop-video("measured_engine.mp4"),
    caption: [
      Simulation result. Five continuous cycles from the representative simulated trial, looped in the web view. The top row groups the conductance family: pistons, time traces, and joint trajectory. The bottom row groups membrane voltage and activity: voltage pistons, voltage traces, and stochastic input with E and I population volleys. The footer separates biological time from playback rate.
    ],
  )

  == Executed methods

  === 2.1 Simulation and sampling

  + Simulate 80 excitatory and 20 inhibitory cells with 128 homogeneous Poisson input channels at 100 Hz per channel, a 0.1 ms timestep, and a 2 ms inhibitory decay.
  + Hold the network realization fixed and use input seeds #(result.per_trial.map(row => str(row.seed)).join(", ")). Simulate 500 ms per trial and exclude the first 100 ms.
  + Record E and I population spikes, membrane voltages, recurrent E-to-I AMPA conductance, and recurrent I-to-E GABA conductance at every timestep. This investigation analyzes the spikes and recurrent conductances.

  === 2.2 Cycle measurements

  + Compute $g_E(t)$ from the recurrent E→I AMPA projection and $g_I(t)$ from the recurrent I→E GABA projection. Average over batch and target cells only after retaining the full per-cell traces.
  + Detect E volleys from the smoothed population spike count after the transient. Define a complete cycle between consecutive detected E volleys.
  + Compute signed phase-plane area, orientation, period, and E-to-I volley lag from each complete $(g_E, g_I)$ path.

  === 2.3 Visual mapping

  + Select the illustrative trial by the frozen median-frequency rule. Downsample five cycles to 300 display frames; retain native-resolution arrays for every analysis.
  + Keep E and I identity stable across pistons, traces, spikes, and the phase portrait. State biological time and playback rate together.

  === 2.4 Frozen interpretation gate

  + Stop the engine interpretation if complete cycles do not share a trajectory orientation.
  + Restrict every conclusion to this network realization and operating point.

  == Conclusion

  The scout passes its coherence gate: all #result.cycles_total detected cycles formed an oriented conductance loop with a median period of #calc.round(result.median_period_ms, digits: 1) ms. The two recurrent conductances therefore support the running-engine visualization at this simulated operating point.

  This bounded result does not establish that the conductance pair is a complete network state or that the same portrait generalizes across network realizations, parameters, or biological gamma rhythms.
]
