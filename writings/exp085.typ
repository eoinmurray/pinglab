#import "contents.typ": with-contents, with-result-sections
#import "/.demolab/lib.typ": data-json, data-image
#import "run-inputs.typ": data-file, inputs-ready, pending-report
#import "run-view.typ": with-datasets, run-view
#import "run-inputs.typ": input-assets
#import "/.demolab/lib.typ": cite, reference-list
#let data-file = data-file.with(article: "exp085")

#let meta = (
  status: "[▦ DATA]",
  title: "Lowet 2015",
  date: "2026-08-19",
  updated_at: "2026-08-31",
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


  Asked which inter-network pathway can lock detuned PING circuits whose gamma
  rhythms naturally drift apart. Compared excitation targeting the other
  circuit's excitatory population, inhibitory population, both populations or
  neither, and probed each phase response.

  Excitatory-to-excitatory coupling corrected phase drift and produced locking;
  inhibitory-targeted coupling alone did not in the selected regime.
  Distinguishes candidate pathways in the selected regime, not their dominance
  across delays, detuning, noise or network realizations.

  == Results

  #with-result-sections[

  === Two-circuit PING coupling schematic

  #figure(
    data-image(data-file("exp085/network.svg"), width: 78%, alt: "Two matched PING networks with local E-to-I-to-E loops and reciprocal E-to-E and E-to-I coupling."),
    caption: [Model schematic: two matched PING circuits. Long-range excitation targets either E or I with exact fan-in 8.],
  )

  === Uncoupled population rhythms and relative-phase drift

  Network A oscillated at #calc.round(a.frequency_hz, digits: 1) Hz and Network
  B at #calc.round(b.frequency_hz, digits: 1) Hz. Their low interval variability
  indicated regular rhythms, but relative phase wrapped #r.uncoupled.phase_wraps
  times.

  #figure(
    data-image(data-file("exp085/uncoupled.png"), width: 88%, alt: "Two clean PING rhythms above their continually wrapping relative phase."),
    caption: [Uncoupled population rhythms after the 300 ms burn-in. Rate
      excerpts are normalized to each trace's maximum. This is one seeded
      trajectory, not an across-seed estimate.],
  )

  === Example corrections from E- and I-targeted probe volleys

  The illustrative probes produced an E-targeted advance, no correction at the
  earlier I-targeted phase, and an I-targeted doublet and delay at the later
  phase.

  #figure(
    data-image(data-file("exp085/phase_response_examples.png"), width: 88%, alt: "Examples of an E-targeted advance, an ineffective I-targeted probe, and an I-targeted doublet and delay."),
    caption: [Three illustrative probes at nominal phases E/0.70, I/0.08 and
      I/0.12. Each trial added one coupling-matched volley to the same baseline
      input realization; these examples are reused in the phase-response curve.
      Each rate trace is normalized to its window maximum.],
  )

  === Phase-response curves for E- and I-targeted probes

  E input advanced the rhythm over a broad late-cycle range. I doublets occurred
  only from phase #calc.round(doublets.first().pulse_phase_fraction, digits: 2)
  to #calc.round(doublets.last().pulse_phase_fraction, digits: 2). Earlier probes
  met refractory neurons; later probes arrived after local excitation faded. The
  inhibitory correction was strong but confined to a narrow window.

  #figure(
    data-image(data-file("exp085/phase_response.png"), width: 92%, alt: "Phase responses to E-targeted and I-targeted probe volleys, including the conductance and voltage mechanism of an inhibitory doublet."),
    caption: [One probe per sampled phase and target population. Lower panels
      show residual local excitation and the probe jointly recruiting recovered
      I neurons; lower traces are means across I neurons.],
  )

  === Relative-phase change across four coupling conditions

  Final phase drift was
  #calc.round(no-coupling.final_drift_rate_cycles_per_s, digits: 2) cycles/s
  without coupling and #calc.round(e-to-i.final_drift_rate_cycles_per_s, digits: 2)
  cycles/s with E-to-I alone. E-to-E alone
  (#calc.round(e-to-e.final_drift_rate_cycles_per_s, digits: 2) cycles/s) and
  both pathways (#calc.round(both.final_drift_rate_cycles_per_s, digits: 2)
  cycles/s) met the locking criteria. The narrow doublet window did not stop
  drift at this operating point.

  #figure(
    data-image(data-file("exp085/pathway_comparison.png"), width: 88%, alt: "Relative-phase change after coupling onset for no coupling, E-to-E only, E-to-I only, and both pathways."),
    caption: [Relative-phase change after coupling onset for four conditions
      sharing the same initial dynamical state and subsequent drive.],
  )

  === Event-aligned excitation and feedback inhibition

  Arriving cross-network conductance advanced Network B's next E volley by
  #calc.round(mechanism.next_target_volley_advance_ms, digits: 1) ms. Feedback
  inhibition advanced with that volley, so this correction began in E, with
  inhibition following rather than initiating it.

  #figure(
    data-image(data-file("exp085/event_aligned_mechanism.png"), width: 92%, alt: "Cross-network excitation followed by an advanced excitatory volley and advanced feedback inhibition."),
    caption: [Illustrative event from the same no-coupling and E-to-E conditions.
      Conductances are means across target E neurons.],
  )

  ]

  == Methods

  Lowet et al. varied reciprocal excitatory-to-excitatory (E-to-E) and excitatory-to-inhibitory (E-to-I) coupling jointly between two pyramidal-interneuron gamma (PING) networks#cite(1). I separated these pathways in an adapted model, comparing isolated phase perturbations with sustained coupling under one initialization and fixed input realizations. No parameters were trained or selected across repetitions.

  + *Set the network and drive.* I constructed two local PING loops, each with #r.network.populations_per_network.E excitatory (E) and #r.network.populations_per_network.I inhibitory (I) conductance-based leaky integrate-and-fire neurons, and evolved them at 0.1 ms resolution. I drove their E populations through 128 independent spike channels at #r.network.detuning_input_rates_hz.PING_A and #r.network.detuning_input_rates_hz.PING_B Hz, using Bernoulli approximations to Poisson input. Local E-to-I excitation decayed in #r.network.local_e_to_i.ampa_tau_ms ms and I-to-E inhibition in 9 ms; external and cross-network excitation decayed in 2 ms. Reciprocal E-to-E and E-to-I projections had nominal weights $K_(E E) = #r.network.weights.K_EE$ and $K_(E I) = #r.network.weights.K_EI$, delay $d = #r.network.delay_ms$ ms, and eight afferents per target; each nominal strength was divided across them.

  + *Measure the uncoupled rhythms.* I simulated 2 s and discarded the first 300 ms for rhythm measurements. Per-neuron population rates were smoothed with a 1 ms Gaussian standard deviation; E volleys required 15 ms separation and prominence of 10% of the maximum rate. Frequency was the reciprocal mean inter-volley interval, variability its standard deviation divided by its mean, and phase advanced linearly between successive E volleys. I checked regular gamma activity, repeated phase wrapping and one I spike per neuron per cycle.

  + *Probe the phase response.* I used 900 ms trials and the baseline cycle spanning 700 ms in Network A. One synchronous volley arrived through the E or I probe pathway at phases 0.02--0.30 in steps of 0.02 and 0.40--0.90 in steps of 0.10, with fixed strengths, #prc.pulse.arrival_delay_ms ms delay and identical background drive. Responses were the baseline next-E-volley time minus the perturbed time, so positive values denote advances; arrival times were rounded to timesteps. Adjacent occupied I-spike timesteps were grouped into volleys to identify doublets.

  + *Compare coupling and classify locking.* At 500 ms I continued the same voltages, conductances, refractory timers and delayed spike histories under no coupling, E-to-E only, E-to-I only, or both, using identical subsequent input for 1.5 s. Over the final #r.pathway_comparison.classification.final_window_ms ms with valid phase, I fitted unwrapped A-minus-B phase in cycles against time and required absolute drift below #r.pathway_comparison.classification.maximum_absolute_drift_cycles_per_s cycles/s and concentration above #r.pathway_comparison.classification.minimum_phase_concentration:

    #set math.equation(numbering: "(1)")
    #show math.equation.where(block: true): it => context {
      if target() == "html" {
        html.elem("div", attrs: (class: "numbered-equation", style: "display:grid;grid-template-columns:1fr auto;align-items:center;gap:1em"))[
          #it
          #html.elem("span", counter(math.equation).display(it.numbering))
        ]
      } else { it }
    }
    $ R_"phase" = abs(1 / N_"phase" sum_(n=1)^(N_"phase") exp(i delta phi_n)) $ <phase-concentration>

    Here $R_"phase"$ is circular phase concentration, $N_"phase"$ the number of valid timesteps, $n$ their index, $delta phi_n$ the A-minus-B phase in radians, and $i$ the imaginary unit.

  + *Resolve the first correction.* I compared the no-coupling and E-to-E conditions around the first A-to-B arrival with a complete −5 to +17 ms window. I measured the next target E volley, target E/I rates and mean incoming and feedback conductances; the event and probe examples reuse the same trajectories, not independent repetitions.

  #run-view("exp085", inputs)

  #reference-list((
    (
      text: [Lowet, E., Roberts, M., Hadjipapas, A., Peter, A., van der Eerden, J., and De Weerd, P. (2015). Input-dependent frequency modulation of cortical gamma oscillations shapes spatial synchronization and enables phase coding. _PLOS Computational Biology_ 11(2), e1004072.],
      doi: "10.1371/journal.pcbi.1004072",
    ),
  ))
]
#body
]

#let report-body = if inputs-ready(data-file, inputs) {
  render-report(data-file)
} else {
  pending-report(
    data-file, inputs,
    [How do coupling and relative timing affect two PING rhythms? Compare uncoupled activity, phase-response probes, and distinct cross-network pathways in relation to Lowet 2015.],
    preview-figures, json-inputs: ("exp085",),
  )
}

#let meta = meta + (assets: input-assets("exp085", inputs))
#let body = with-datasets("exp085", inputs, report-body, placed: inputs-ready(data-file, inputs))
#let body = with-contents(body)
