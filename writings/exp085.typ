#let meta = (
  title: "How does excitation synchronize two PING networks?",
  date: "2026-08-19",
  description: "Distinguish the pathways that phase-lock two cortical PING rhythms.",
  collection: "snnlang",
  status: "draft",
  order: 11,
)

#let body = [
  == Abstract

  Two frequency-detuned PING networks are connected by reciprocal excitatory projections. This experiment asks whether phase locking is caused mainly by direct excitation of pyramidal cells, recruitment of local inhibition, or interaction between both pathways.

  == Prior art

  Lowet et al. used two 80-excitatory, 20-inhibitory PING networks connected through E-to-E and E-to-I projections. They found that locking depends mainly on frequency detuning and coupling strength. Their topology provides the starting point for this experiment: #link("https://doi.org/10.1371/journal.pcbi.1004072")[Lowet et al. (2015)].

  Phase-response analysis provides a way to connect individual coupling events to network-level locking. It also warns that a population rhythm can respond differently from any one neuron: #link("https://doi.org/10.7554/eLife.26642")[Lowet et al. (2017)] and #link("https://doi.org/10.1371/journal.pcbi.1008575")[Xu and Riecke (2021)].

  == Methods

  #block(inset: 10pt, fill: rgb("f3f0e8"), radius: 3pt)[
    *Method 1 is complete. Methods 2–5 are not yet run.*
  ]

  + *Define the networks.* Build two matched PING networks. Each contains 80 excitatory and 20 inhibitory neurons connected through local E-to-I and I-to-E synapses. Drive Network A at 110 Hz and Network B at 90 Hz; Method 2 will test whether this produces the required frequency detuning. Connect the networks reciprocally through E-to-E and E-to-I projections. Each target receives exactly eight randomly selected excitatory afferents from the other network. Give the pathways separate weights $K_(E E)$ and $K_(E I)$ and a shared delay $d$.

    #figure(
      image("/artifacts/data/exp085/network.svg", width: 100%, alt: "SNNLANG circuit diagram of two matched PING networks with reciprocal E-to-E and E-to-I coupling."),
      caption: [SNNLANG circuit export. Each network contains a local E-to-I-to-E PING loop. Reciprocal cross-network excitation targets both populations with exact fan-in 8, separate $K_(E E)$ and $K_(E I)$ controls, and shared delay $d$.],
    )

    Generates #link("#result-1-network-definition")[Result 1].

  + *Confirm uncoupled phase drift.* Run the networks without cross-network coupling. Confirm that each produces a stable PING rhythm and that their relative phase drifts. Generates #link("#result-2-uncoupled-phase-drift")[Result 2].

  + *Measure each pathway's phase response.* At different phases of one network's cycle, deliver a brief excitatory pulse either to its excitatory population or to its inhibitory population. Match these pulses to the E-to-E and E-to-I coupling events used in the paired network. Measure how each pulse changes the timing of the next excitatory volley. This shows whether each pathway advances or delays the rhythm and predicts the phases at which coupled networks could lock. Generates #link("#result-3-phase-response-curve")[Result 3].

  + *Test the coupling pathways.* Continue from the same saved state and input in four conditions: no coupling, E-to-E only, E-to-I only, and both pathways. Track the relative phase in each condition. Generates #link("#result-4-pathway-comparison")[Result 4].

  + *Trace the mechanism.* Align activity on incoming excitatory volleys. Measure target excitatory activity, target inhibitory activity, inhibition received by the target excitatory population, and the phase correction in the following cycle. Generates #link("#result-5-event-aligned-mechanism")[Result 5].

  == Results

  === Result 1: Network definition

  The authored graph contains two matched 80-excitatory, 20-inhibitory PING circuits. It keeps inhibition local and uses reciprocal long-range E-to-E and E-to-I projections. The two pathways have separate parameters, $K_(E E) = 0.08$ and $K_(E I) = 0.08$, with delay $d = 0.5$ ms. SNNLANG represents the sparse projections through exact-fan-in initialization. No network activity has been simulated yet.

  === Result 2: Uncoupled phase drift

  *Axes.* Time against wrapped relative phase.  \
  *Trace.* Phase of Network A relative to Network B.  \
  *Why.* Establish the frequency mismatch before coupling.  \
  *Expectation.* The phase repeatedly crosses the full wrapped range.

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
