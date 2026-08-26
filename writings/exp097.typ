#let meta = (
  title: "Can a PING cycle be seen as a running engine?",
  created_at: "2026-08-23",
  updated_at: "2026-08-24",
  description: "Tests whether recurrent excitatory and inhibitory conductances trace a coherent simulated PING cycle.",
  collection: "demo",
  order: 11,
  status: "ExpScout",
)

#let r = json("/.artifacts/exp097/numbers.json")
#let result = r.results
#let loop-video(src) = context {
  if target() == "html" {
    html.elem("video", attrs: (src: src, controls: "", loop: "", playsinline: "", style: "max-width:100%;width:100%"))[]
  } else {
    text(size: 9pt, style: "italic", fill: gray)[[Video — view the web edition to play.]]
  }
}

#let body = [
  == 1. Abstract

  This exploratory scout asks whether recurrent excitatory and inhibitory conductances provide a useful moving portrait of a PING cycle. The pair traces a consistent loop under drive and disappears with activity when drive is removed. The rhythm remains below gamma, and voltage adds predictive information.

  The engine picture shows the rhythm clearly, but it does not contain the whole machine.

  == 2. Design and scope

  One fixed 800-excitatory, 200-inhibitory network supports both investigations. The display follows $g_E(t)$, mean recurrent excitation onto inhibitory cells, and $g_I(t)$, mean recurrent inhibition onto excitatory cells.

  The scout asks whether $(g_E, g_I)$ repeatedly traces one oriented loop under constant drive and follows activity from silence back to silence. An inconsistent path would stop the engine interpretation. A below-gamma rhythm or improved prediction after adding voltage requires revision. The scout includes no training, parameter search, biological data, or robustness claim.

  #block(inset: 10pt, fill: rgb("eef4f8"), radius: 3pt)[
    No separately frozen prospective plan predates this run. The expectations and gates above are reconstructed from the preserved experimental record. They remain distinct from the observations but cannot substitute for prospective registration.
  ]

  Both investigations completed locally without a known scientific deviation. The disposition is *revise*: the loop is coherent but below gamma and predictively incomplete.

  == 3. Investigations

  === 3.1 Constant drive

  A coherent cycle should raise $g_E$, recruit inhibition, and then raise $g_I$ while excitatory activity is suppressed. Complete cycles are aligned to excitatory volleys, and the representative trial is selected by a fixed median-frequency rule. A collapsed or inconsistently oriented path would weaken the engine account.

  #figure(
    loop-video("measured_engine.mp4"),
    caption: [
      Simulation result. Five continuous cycles from the representative simulated trial, looped in the web view. The top row shows recurrent conductance as paired instruments, time traces, and a joint trajectory. The bottom row shows membrane voltage and activity. Panel 6 contains native-time spikes from 24 fixed input channels, 40 of 800 excitatory cells, and 20 of 200 inhibitory cells. Thin envelopes show size-normalized rates from each full population.
    ],
  )

  The five trials supplied #result.cycles_total complete cycles, all moving #result.modal_orientation with a median frequency of #calc.round(result.median_frequency_hz, digits: 1) Hz. The loop is coherent but below gamma. Adding population-mean voltage reduced held-out phase error from #calc.round(result.prediction.two_phase, digits: 3) to #calc.round(result.prediction.four_phase, digits: 3) cycles and next-volley error from #calc.round(result.prediction.two_timing, digits: 1) to #calc.round(result.prediction.four_timing, digits: 1) ms, so conductance alone is not a complete predictive state.

  === 3.2 From silence to activity and back

  The same display should become active only while external drive rises and falls. One fixed input realization applies a 0–50–0 Hz command between silent periods; activity before or after that interval would weaken the claim.

  #figure(
    loop-video("input_ramp_engine.mp4"),
    caption: [
      Simulation result. The commanded input rises from zero to 50 Hz per channel, returns to zero, and ends with 200 ms of silence. The dashed curve in Panel 6 is the command; sampled spikes and solid rate envelopes are simulated responses. The same network and display mapping are used as in the constant-drive video.
    ],
  )

  Both populations were silent before and after the drive. During it, excitatory cells averaged #calc.round(result.drive_transients.ramp.e_rate_active_hz, digits: 1) Hz and inhibitory cells #calc.round(result.drive_transients.ramp.i_rate_active_hz, digits: 1) Hz per neuron. The detector found #result.drive_transients.ramp.active_e_volleys excitatory volleys while driven and #result.drive_transients.ramp.post_e_volleys afterward. In this trial, the engine follows the command back to silence.

  == 4. Methods

  + Simulate 800 excitatory and 200 inhibitory cells with 128 homogeneous Poisson input channels, a 0.1 ms timestep, and a 2 ms inhibitory decay.
  + Hold the network realization fixed. For constant drive, use input seeds #(result.per_trial.map(row => str(row.seed)).join(", ")) at 50 Hz per channel. Simulate 500 ms per trial and exclude the first 100 ms.
  + Record external-input spikes, excitatory and inhibitory population spikes, membrane voltages, recurrent E-to-I AMPA conductance, and recurrent I-to-E GABA conductance at every timestep.
  + Compute $g_E(t)$ from the recurrent E-to-I AMPA projection and $g_I(t)$ from the recurrent I-to-E GABA projection. Retain per-cell traces before averaging over target cells.
  + Detect excitatory volleys from the smoothed population spike count after the transient. Define a complete cycle between consecutive detected volleys.
  + Measure signed phase-plane area, orientation, period, and excitatory-to-inhibitory volley lag for each complete $(g_E, g_I)$ path.
  + Compare nearest-neighbour prediction from $(g_E, g_I)$ with prediction from $(g_E, g_I, V_E, V_I)$ on held-out cycles. Measure circular phase error and absolute time-to-next-volley error.
  + Select the illustrative constant-drive trial by the fixed median-frequency rule. Downsample five cycles to 300 display frames while retaining native-resolution arrays for analysis.
  + Display native timestamps for 24 fixed input channels, 40 fixed excitatory cells, and 20 fixed inhibitory cells. Overlay per-neuron rate envelopes from each complete population.
  + For the second video, apply a linear 0–50–0 Hz input-rate schedule between two 200 ms silent periods. Use one fixed input realization and the unchanged network.
  + Restrict every conclusion to this network realization, its tested operating points, and simulated evidence.

  == 5. Conclusion

  The scout supports *revision*, not escalation. The display captures an oriented drive-dependent cycle, but the rhythm is below gamma and conductance alone is predictively incomplete. One network cannot establish generality or biological relevance. A new prospective scout should first locate a genuine gamma operating point before testing the portrait across network realizations.
]
