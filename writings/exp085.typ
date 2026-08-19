#let meta = (
  title: "How does excitation synchronize two PING networks?",
  date: "2026-08-19",
  description: "Distinguish the pathways that phase-lock two cortical PING rhythms.",
  collection: "snnlang",
  status: "draft",
  order: 11,
)

#let r = json("/artifacts/data/exp085/numbers.json")
#let a = r.uncoupled.PING_A
#let b = r.uncoupled.PING_B
#let prc = r.phase_response
#let doublets = prc.responses.I.filter(row => row.i_volleys_before_next_e == 2)

#let body = [
  == Abstract

  Two frequency-detuned PING networks are connected by reciprocal excitatory projections. This experiment asks whether phase locking is caused mainly by direct excitation of pyramidal cells, recruitment of local inhibition, or interaction between both pathways.

  == Prior art

  Lowet et al. used two 80-excitatory, 20-inhibitory PING networks connected through E-to-E and E-to-I projections. They found that locking depends mainly on frequency detuning and coupling strength. Their topology provides the starting point for this experiment: #link("https://doi.org/10.1371/journal.pcbi.1004072")[Lowet et al. (2015)].

  Phase-response analysis provides a way to connect individual coupling events to network-level locking. It also warns that a population rhythm can respond differently from any one neuron: #link("https://doi.org/10.7554/eLife.26642")[Lowet et al. (2017)] and #link("https://doi.org/10.1371/journal.pcbi.1008575")[Xu and Riecke (2021)].

  == Methods

  #block(inset: 10pt, fill: rgb("f3f0e8"), radius: 3pt)[
    *Methods 1–3 are complete. Methods 4–5 are not yet run.*
  ]

  + *Define the networks.* Build two matched PING networks. Each contains #r.network.populations_per_network.E excitatory and #r.network.populations_per_network.I inhibitory neurons connected through local E-to-I and I-to-E synapses. Drive Network A at #r.network.detuning_input_rates_hz.PING_A Hz and Network B at #r.network.detuning_input_rates_hz.PING_B Hz; these values place both uncoupled rhythms in the gamma band. Use a #r.network.local_e_to_i.ampa_tau_ms ms local E-to-I AMPA decay so each inhibitory neuron fires once, rather than twice, after an excitatory volley. Connect the networks reciprocally through E-to-E and E-to-I projections. Each target receives exactly #r.network.exact_fan_in_per_target randomly selected excitatory afferents from the other network. Give the pathways separate weights $K_(E E)$ and $K_(E I)$ and a shared delay $d$.

    #figure(
      image("/artifacts/data/exp085/network.svg", width: 100%, alt: "SNNLANG circuit diagram of two matched PING networks with reciprocal E-to-E and E-to-I coupling."),
      caption: [SNNLANG circuit export. Each network contains a local E-to-I-to-E PING loop. Reciprocal cross-network excitation targets both populations with exact fan-in 8, separate $K_(E E)$ and $K_(E I)$ controls, and shared delay $d$.],
    )

    Generates #link("#result-1-network-definition")[Result 1].

  + *Confirm uncoupled phase drift.* Set $K_(E E) = K_(E I) = 0$ and run both networks for 2 s with independent input streams. Discard the first 300 ms. Detect excitatory population volleys, interpolate phase between successive volleys, and calculate the wrapped phase difference. Require both rhythms to remain in the gamma band with an inter-volley-interval coefficient of variation below 0.2. Generates #link("#result-2-uncoupled-phase-drift")[Result 2].

  + *Measure each pathway's phase response.* Use Network A and one baseline cycle. A preliminary scan at intervals of 0.1 cycle located an abrupt inhibitory response near phase 0.1. The reported run therefore samples phases 0.02--0.30 at intervals of 0.02 and retains intervals of 0.1 over the rest of the cycle.

    Keep $K_(E E) = #prc.pulse.e_target_strength$, $K_(E I) = #prc.pulse.i_target_strength$, and $d = #prc.pulse.arrival_delay_ms$ ms fixed. Only the probe target and arrival phase change. For desired phase $phi$, emit the source volley at $t_E + phi T - d$ so it reaches the target at $t_E + phi T$. Every condition uses the same network, initial state, input, and seed. Compare the next excitatory volley with the no-probe baseline. For I-targeted probes, also count inhibitory volleys and measure any second-volley latency. In a representative doublet, record local and probe excitatory conductance and every I-cell voltage. Positive phase shifts mean an advance; negative shifts mean a delay. Generates #link("#result-3-phase-response-curve")[Result 3].

  + *Test the coupling pathways.* Continue from the same saved state and input in four conditions: no coupling, E-to-E only, E-to-I only, and both pathways. Track the relative phase in each condition. Generates #link("#result-4-pathway-comparison")[Result 4].

  + *Trace the mechanism.* Align activity on incoming excitatory volleys. Measure target excitatory activity, target inhibitory activity, inhibition received by the target excitatory population, and the phase correction in the following cycle. Generates #link("#result-5-event-aligned-mechanism")[Result 5].

  == Results

  === Result 1: Network definition

  The authored graph contains two matched #(r.network.populations_per_network.E)-excitatory, #(r.network.populations_per_network.I)-inhibitory PING circuits. It keeps inhibition local and uses reciprocal long-range E-to-E and E-to-I projections. The two pathways have separate parameters, $K_(E E) = #r.network.weights.K_EE$ and $K_(E I) = #r.network.weights.K_EI$, with delay $d = #r.network.delay_ms$ ms. SNNLANG represents the sparse projections through exact-fan-in initialization.

  === Result 2: Uncoupled phase drift

  #figure(
    image("/artifacts/data/exp085/uncoupled.png", width: 100%, alt: "Excitatory and inhibitory PING rhythm excerpts above the wrapped phase difference between the uncoupled networks."),
    caption: [Uncoupled activity. The upper panels show normalized excitatory and inhibitory population rates over the same 250 ms interval. The lower panel shows the wrapped phase difference across the full post-burn recording. Vertical jumps are phase wrapping.],
  )

  Network A oscillates at #calc.round(a.frequency_hz, digits: 1) Hz with inter-volley-interval CV #calc.round(a.iei_cv, digits: 3). Network B oscillates at #calc.round(b.frequency_hz, digits: 1) Hz with CV #calc.round(b.iei_cv, digits: 3). Every inhibitory neuron fires exactly once in every measured cycle. Both networks therefore produce clean, regular gamma rhythms without baseline inhibitory doublets. Their relative phase wraps #r.uncoupled.phase_wraps times, establishing sustained phase drift before coupling.

  === Result 3: Phase-response curve

  #figure(
    image("/artifacts/data/exp085/phase_response_examples.png", width: 100%, alt: "Three event-aligned PING responses showing an E-targeted advance, an ineffective I-targeted probe, and an I-targeted doublet and delay."),
    caption: [Representative probe responses. Time zero is the reference excitatory volley. Black and red show perturbed E and I rates. Cyan marks probe arrival. The grey dashed trace and line show the baseline E response and baseline next-volley time; a black dotted line marks a shifted next volley.],
  )

  #figure(
    image("/artifacts/data/exp085/phase_response.png", width: 100%, alt: "Phase-response curves for coupling-matched excitation of the excitatory and inhibitory populations."),
    caption: [Response to coupling-matched probe volleys. The upper panels show the whole-cycle response and enlarged early-I window. Open points produce one inhibitory volley; filled red points produce doublets. The lower panels show local, probe, and total excitation onto I neurons and all I-cell voltages for the phase-0.12 doublet. Cyan marks probe arrival; the dashed voltage line marks threshold.],
  )

  E-targeted probes advance the rhythm late in the cycle. I-targeted probes delay it only from phase #calc.round(doublets.first().pulse_phase_fraction, digits: 2) to #calc.round(doublets.last().pulse_phase_fraction, digits: 2), where they trigger a second inhibitory volley.

  In the doublet case, local E-to-I conductance remains when the probe arrives. The probe raises total excitation as the recovered I-cell voltages approach threshold, producing the second volley. Earlier probes meet refractory neurons; later probes lack this remaining local excitation and are too weak alone.

  === Result 4: Pathway comparison

  *Axes.* Time against wrapped relative phase.  \
  *Traces.* No coupling, E-to-E only, E-to-I only, and both pathways.  \
  *Why.* Identify which pathway is sufficient for locking.  \
  *Expectation.* The conditions separate direct excitation, feedforward inhibition, and any interaction between them.

  === Result 5: Event-aligned mechanism

  *Axes.* Time from a source excitatory volley against normalized activity or conductance.  \
  *Traces.* Arriving excitation, target inhibitory activity, target inhibition, and target excitatory activity.  \
  *Why.* Establish the order of events that produces each phase correction.  \
  *Expectation.* The pathway responsible for locking changes before the target network's next corrected volley.
]
