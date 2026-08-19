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

#let body = [
  == Abstract

  Two frequency-detuned PING networks are connected by reciprocal excitatory projections. This experiment asks whether phase locking is caused mainly by direct excitation of pyramidal cells, recruitment of local inhibition, or interaction between both pathways.

  == Prior art

  Lowet et al. used two 80-excitatory, 20-inhibitory PING networks connected through E-to-E and E-to-I projections. They found that locking depends mainly on frequency detuning and coupling strength. Their topology provides the starting point for this experiment: #link("https://doi.org/10.1371/journal.pcbi.1004072")[Lowet et al. (2015)].

  Phase-response analysis provides a way to connect individual coupling events to network-level locking. It also warns that a population rhythm can respond differently from any one neuron: #link("https://doi.org/10.7554/eLife.26642")[Lowet et al. (2017)] and #link("https://doi.org/10.1371/journal.pcbi.1008575")[Xu and Riecke (2021)].

  == Methods

  #block(inset: 10pt, fill: rgb("f3f0e8"), radius: 3pt)[
    *Methods 1–2 are complete. Methods 3–5 are not yet run.*
  ]

  + *Define the networks.* Build two matched PING networks. Each contains #r.network.populations_per_network.E excitatory and #r.network.populations_per_network.I inhibitory neurons connected through local E-to-I and I-to-E synapses. Drive Network A at #r.network.detuning_input_rates_hz.PING_A Hz and Network B at #r.network.detuning_input_rates_hz.PING_B Hz; these values place both uncoupled rhythms in the gamma band. Use a #r.network.local_e_to_i.ampa_tau_ms ms local E-to-I AMPA decay so each inhibitory neuron fires once, rather than twice, after an excitatory volley. Connect the networks reciprocally through E-to-E and E-to-I projections. Each target receives exactly #r.network.exact_fan_in_per_target randomly selected excitatory afferents from the other network. Give the pathways separate weights $K_(E E)$ and $K_(E I)$ and a shared delay $d$.

    #figure(
      image("/artifacts/data/exp085/network.svg", width: 100%, alt: "SNNLANG circuit diagram of two matched PING networks with reciprocal E-to-E and E-to-I coupling."),
      caption: [SNNLANG circuit export. Each network contains a local E-to-I-to-E PING loop. Reciprocal cross-network excitation targets both populations with exact fan-in 8, separate $K_(E E)$ and $K_(E I)$ controls, and shared delay $d$.],
    )

    Generates #link("#result-1-network-definition")[Result 1].

  + *Confirm uncoupled phase drift.* Set $K_(E E) = K_(E I) = 0$ and run both networks for 2 s with independent input streams. Discard the first 300 ms. Detect excitatory population volleys, interpolate phase between successive volleys, and calculate the wrapped phase difference. Require both rhythms to remain in the gamma band with an inter-volley-interval coefficient of variation below 0.2. Generates #link("#result-2-uncoupled-phase-drift")[Result 2].

  + *Measure each pathway's phase response.* At different phases of one network's cycle, deliver a brief excitatory pulse either to its excitatory population or to its inhibitory population. Match these pulses to the E-to-E and E-to-I coupling events used in the paired network. Measure how each pulse changes the timing of the next excitatory volley. This shows whether each pathway advances or delays the rhythm and predicts the phases at which coupled networks could lock. Generates #link("#result-3-phase-response-curve")[Result 3].

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

  *Axes.* Pulse phase against the change in next-volley phase.  \
  *Traces.* Phase responses to excitation of the excitatory population and excitation of the inhibitory population.  \
  *Why.* Explain how the E-to-E and E-to-I pathways correct phase, and predict the pathway comparison in Method 4.  \
  *Expectation.* Each response changes with pulse timing. Their zero crossings and slopes identify possible stable locking phases.

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
