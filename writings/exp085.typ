#import "/.demolab/lib.typ": data-json, data-image
#import "run-inputs.typ": data-file, inputs-ready, pending-report
#import "/.demolab/lib.typ": cite, reference-list
#let data-file = data-file.with(article: "exp085")

#let meta = (
  status: "Implemented",
  title: "Lowet 2015",
  date: "2026-08-19",
  description: "Distinguish the pathways that phase-lock two cortical PING rhythms.",
  collection: "demo",
  order: 1,
)

#let inputs = ("exp085",)
#let preview-figures = (
  (path: "exp085/network.svg", label: "network"),
  (path: "exp085/uncoupled.png", label: "uncoupled"),
  (path: "exp085/phase_response_examples.png", label: "phase response examples"),
  (path: "exp085/phase_response.png", label: "phase response"),
  (path: "exp085/pathway_comparison.png", label: "pathway comparison"),
  (path: "exp085/event_aligned_mechanism.png", label: "event aligned mechanism"),
)

// Keep calculations lazy: absent inputs never become fabricated results.
#let render-report(data-file) = [
#let r = data-json(data-file("exp085/numbers.json"))
#let a = r.uncoupled.PING_A
#let b = r.uncoupled.PING_B
#let prc = r.phase_response
#let doublets = prc.responses.I.filter(row => row.i_volleys_before_next_e == 2)
#let pathways = r.pathway_comparison.conditions
#let mechanism = r.event_aligned_mechanism
#let no-coupling = pathways.filter(row => row.id == "none").first()
#let e-to-e = pathways.filter(row => row.id == "e_to_e").first()
#let e-to-i = pathways.filter(row => row.id == "e_to_i").first()
#let both = pathways.filter(row => row.id == "both").first()

#let body = [
  == Abstract

  Two gamma rhythms with different frequencies drift out of phase unless coupling corrects the mismatch. Lowet et al. showed that reciprocal excitation can synchronize PING networks through pathways targeting either excitatory or inhibitory neurons#cite(1). We recreate that setup and separate the two mechanisms. Both pathways alter volley timing, but only E-to-E coupling stops the drift at this operating point. Excitation advances the next E volley; inhibition follows rather than initiating the correction.

  == The question

  Lowet et al. connected two PING networks through reciprocal E-to-E and E-to-I projections and showed that locking depends on detuning and coupling strength#cite(1). The two pathways suggest different mechanisms. E-to-E input can advance the target rhythm directly. E-to-I input can recruit inhibition and delay its next volley. Which route actually stops phase drift?

  == The system

  Each network contains #r.network.populations_per_network.E excitatory and #r.network.populations_per_network.I inhibitory neurons in a local PING loop. Sparse reciprocal excitation targets both populations through separate pathways.

  #figure(
    data-image(data-file("exp085/network.svg"), width: 78%, alt: "Two matched PING networks with local E-to-I-to-E loops and reciprocal E-to-E and E-to-I coupling."),
    caption: [Two matched PING circuits. Long-range excitation targets either E or I with exact fan-in 8.],
  )

  Network A receives #r.network.detuning_input_rates_hz.PING_A Hz input and Network B receives #r.network.detuning_input_rates_hz.PING_B Hz. At the demonstration point, $K_(E E) = #r.network.weights.K_EE$, $K_(E I) = #r.network.weights.K_EI$, and delay $d = #r.network.delay_ms$ ms.

  == Evidence

  === 1. The uncoupled rhythms drift

  #figure(
    data-image(data-file("exp085/uncoupled.png"), width: 88%, alt: "Two clean PING rhythms above their continually wrapping relative phase."),
    caption: [Both networks sustain gamma rhythms, but their relative phase keeps moving.],
  )

  Network A oscillates at #calc.round(a.frequency_hz, digits: 1) Hz and Network B at #calc.round(b.frequency_hz, digits: 1) Hz. Their low interval variability confirms regular rhythms. Relative phase wraps #r.uncoupled.phase_wraps times, establishing the drift that coupling must stop.

  === 2. The pathways make different corrections

  We deliver one coupling-matched probe volley at different phases of Network A's cycle. E-targeted probes advance the next excitatory volley late in the cycle. I-targeted probes delay it only in a narrow window.

  #figure(
    data-image(data-file("exp085/phase_response_examples.png"), width: 88%, alt: "Examples of an E-targeted advance, an ineffective I-targeted probe, and an I-targeted doublet and delay."),
    caption: [Three representative responses: direct advance, no correction, and doublet-mediated delay.],
  )

  #figure(
    data-image(data-file("exp085/phase_response.png"), width: 92%, alt: "Phase responses to E-targeted and I-targeted probe volleys, including the conductance and voltage mechanism of an inhibitory doublet."),
    caption: [E input advances over a broad late-cycle range. I input delays only from phase #calc.round(doublets.first().pulse_phase_fraction, digits: 2) to #calc.round(doublets.last().pulse_phase_fraction, digits: 2); the lower panels show why.],
  )

  The I-targeted delay occurs when residual local excitation and the probe jointly push recovered I neurons back across threshold. Earlier probes meet refractory neurons; later probes arrive after the local excitation has faded. The inhibitory correction is therefore strong but difficult to engage.

  === 3. Only E-to-E coupling stops the drift

  We start four conditions from the same saved state: no coupling, E-to-E only, E-to-I only, and both pathways.

  #figure(
    data-image(data-file("exp085/pathway_comparison.png"), width: 88%, alt: "Relative-phase change after coupling onset for no coupling, E-to-E only, E-to-I only, and both pathways."),
    caption: [A slope means continued drift; a plateau means phase locking.],
  )

  No coupling drifts at #calc.round(no-coupling.final_drift_rate_cycles_per_s, digits: 2) cycles/s. E-to-I alone still drifts at #calc.round(e-to-i.final_drift_rate_cycles_per_s, digits: 2) cycles/s. E-to-E alone reduces drift to #calc.round(e-to-e.final_drift_rate_cycles_per_s, digits: 2) cycles/s and locks the rhythms. Both pathways also lock, at #calc.round(both.final_drift_rate_cycles_per_s, digits: 2) cycles/s.

  The doublet mechanism exists, but its narrow phase window is insufficient here. Direct excitation captures the drifting rhythms.

  === 4. The first correction begins in E

  #figure(
    data-image(data-file("exp085/event_aligned_mechanism.png"), width: 92%, alt: "Cross-network excitation followed by an advanced excitatory volley and advanced feedback inhibition."),
    caption: [The first correction runs from arriving excitation to an advanced E volley and then advanced feedback inhibition.],
  )

  Arriving E-to-E conductance advances Network B's next excitatory volley by #calc.round(mechanism.next_target_volley_advance_ms, digits: 1) ms. Feedback inhibition advances with it, so inhibition follows the correction rather than initiating it.

  == Methods

  + Run the networks uncoupled for 2 s. Discard 300 ms, detect excitatory volleys, and interpolate phase between them.

  + Measure phase response from one baseline cycle. Sample phases 0.02--0.30 in steps of 0.02 around the abrupt I response and use steps of 0.1 elsewhere. Keep probe strengths and the #prc.pulse.arrival_delay_ms ms arrival delay fixed. Change only target population and arrival phase.

  + Save the full uncoupled runtime state, then branch into the four coupling conditions. Classify the final #r.pathway_comparison.classification.final_window_ms ms as locked from phase drift and concentration.

  + Compare no-coupling and E-to-E branches around the first arriving cross-network volley. Measure target E and I activity, returned inhibition, and the next E-volley time.

  == Scope

  This is one selected operating point, not a claim that E-to-E coupling dominates across delays, detuning, noise, or network realizations. Broader claims require systematic sweeps and repeated runs.

  == References

  #reference-list((
    (
      text: [Lowet, E., Roberts, M., Hadjipapas, A., Peter, A., van der Eerden, J., and De Weerd, P. (2015). Input-dependent frequency modulation of cortical gamma oscillations shapes spatial synchronization and enables phase coding.],
      doi: "10.1371/journal.pcbi.1004072",
    ),
  ))
]
#body
]

#let body = if inputs-ready(data-file, inputs) {
  render-report(data-file)
} else {
  pending-report(
    data-file, inputs,
    [How do coupling and relative timing affect two PING rhythms? Compare uncoupled activity, phase-response probes, and distinct cross-network pathways in relation to Lowet 2015.],
    preview-figures, json-inputs: ("exp085",),
  )
}
