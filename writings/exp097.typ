#let meta = (
  title: "Can a PING cycle be seen as a running engine?",
  date: "2026-08-23",
  description: "Tests whether recurrent conductances form a useful two-variable portrait of a stochastic PING cycle and where that reduction fails.",
  collection: "snnlang",
  status: "draft",
  order: 11,
)

#let body = [
  == Abstract

  Pyramidal–interneuron gamma (PING) is often described as alternating excitatory and inhibitory volleys. That description hides the local state changes that make the rhythm work. This scout asks whether the recurrent excitatory and inhibitory conductances form a useful two-variable portrait of the cycle, how stochastic structure changes from individual cells to population means, and what information is lost when membrane voltage is omitted. A synchronized mechanical animation will expose the measured trajectory, the spikes that produce each conductance increment, and the voltage changes that lead to the next volley.

  The aim is to identify the smallest honest state portrait of this simulated PING mechanism, not to establish that the same reduction holds for biological gamma rhythms.

  == Shared design

  Nothing below has been run. Use the established 80-E, 20-I conductance-based PING network at its active gamma operating point. Freeze the network, timestep, synaptic parameters, and homogeneous Poisson drive. Observe five paired 500 ms trials and exclude the first 100 ms as settling time.

  Define the two main displayed state variables as

  $ g_E(t) = "mean conductance of the recurrent E→I projection", $

  $ g_I(t) = "mean conductance of the recurrent I→E projection". $

  These are separate target-local conductances. Neither is transferred into the other. E and I spike arrivals trigger conductance increments; each conductance then decays according to its own synaptic dynamics. Mean E and I membrane voltage and population spike count provide causal context but do not replace the two-variable state.

  #figure(
    image(
      "/artifacts/data/exp097/ping_engine_storyboard.svg",
      width: 100%,
      alt: "Design schematic with a mechanical E and I conductance engine, synchronized spike and voltage traces, a conductance phase portrait, and a comparison of two-variable and four-variable state descriptions.",
    ),
    caption: [
      Design schematic, not data. The planned animation keeps four views synchronized: a mechanical cutaway driven by $g_E$ and $g_I$; spike and voltage traces that identify each trigger; a moving point in the $(g_E, g_I)$ plane; and a direct comparison between the two-conductance portrait and the fuller $(g_E, g_I, V_E, V_I)$ state. The measured animation will use the same layout but a distinct measured-evidence colour treatment.
    ],
  )

  == 1. Do two recurrent conductances trace a coherent PING cycle?

  The first investigation tests whether the population means $g_E(t)$ and $g_I(t)$ produce a repeatable oriented trajectory rather than an unstructured cloud. For every complete post-transient cycle, align time to the E-population volley and trace the path through the $(g_E, g_I)$ plane. Measure trajectory orientation, enclosed area, cycle duration, E-to-I volley lag, and the phase of each conductance peak.

  Select the illustrative trial by a frozen rule: choose the trial whose post-transient rhythm frequency is closest to the five-trial median. Select four complete cycles beginning at its first complete E volley after the transient. The animation will show these cycles continuously, with measured biological time and a clearly stated playback slowdown.

  *Expected patterns.* Under a coherent PING cycle, $g_E$ rises after the E volley, the I volley follows, and $g_I$ rises before E activity is suppressed. Repeated cycles should traverse the plane in the same direction with bounded variation in area and timing. A collapsed line, inconsistent orientation, or broad phase cloud would show that the two-conductance portrait is visually convenient but dynamically weak.

  *Planned visual evidence.* A measured mechanical cycle paired with the $(g_E, g_I)$ trajectory, population spike ticks, and direct timing labels. The main motion uses physical conductance values rather than a generic activity scale.

  == 2. Where does stochasticity live in the cycle?

  The second investigation compares three resolutions of the same cycles: individual target-cell conductances, their distribution across cells, and the population mean. Measure the size and timing of discrete conductance increments, across-cell dispersion, cycle-to-cycle peak variation, and how much variance remains after population averaging.

  The visual mapping will preserve this hierarchy. The main plungers show population means. A narrow moving distribution shows the cells beneath each mean. Spike-trigger indicators flash at presynaptic arrivals, making each local jump and subsequent decay visible. No pipe, shared reservoir, or moving fluid will connect the conductances.

  *Expected patterns.* Individual-cell traces should show irregular spike-driven increments and exponential decay. Population averaging should reduce amplitude noise while preserving the ordered E-then-I rhythm. If averaging removes the discrete structure almost completely, the animation must show both levels rather than falsely making the mean look like a single synapse.

  *Planned visual evidence.* One synchronized passage will move from individual kicks, through the across-cell distribution, to the population-scale engine. The schematic uses the prospective blue-grey grammar. Measured marks will use the registered measured-evidence grammar with separate E and I shape coding and direct labels.

  == 3. What does the two-variable portrait leave out?

  The third investigation tests the boundary of the two-conductance reduction. Compare cycle phase and next-volley prediction from $(g_E, g_I)$ against the fuller state $(g_E, g_I, V_E, V_I)$, where $V_E$ and $V_I$ are mean population membrane voltages. Use leave-one-cycle-out prediction within each trial and leave-one-trial-out prediction across trials. Report circular phase error and next-volley timing error rather than classification accuracy alone.

  This comparison does not ask whether four variables reconstruct the complete network. It asks whether voltage contains cycle-relevant state that the conductance plane aliases. Refractory state, cell-to-cell dispersion, and synaptic history remain outside both reductions.

  *Expected patterns.* If $(g_E, g_I)$ is sufficient for the visible cycle, adding mean voltage should give little improvement in phase or next-volley timing. A substantial improvement would show that the two-conductance engine is an explanatory projection, not a minimal dynamical state. Similar errors from both descriptions would leave omitted refractory or distributional state as the stronger limitation.

  *Planned visual evidence.* The animation will pair the conductance-only engine with voltage needles and mark moments when identical or nearby conductance states lead to different subsequent motion. A small error comparison will state whether voltage materially improves phase and timing prediction.

  == Methods

  === 4.1 Frozen model and sampling

  + Use 80 excitatory and 20 inhibitory cells with 128 homogeneous Poisson input channels at 100 Hz per channel, a 0.1 ms timestep, and a 2 ms inhibitory decay.
  + Freeze one network realization and use five predeclared input realizations. Observe 500 ms per trial and exclude the first 100 ms.
  + Measure population spikes, membrane voltages, recurrent E→I AMPA conductance, and recurrent I→E GABA conductance at the native timestep.

  === 4.2 Operational definitions

  + Compute $g_E(t)$ from the recurrent E→I AMPA projection and $g_I(t)$ from the recurrent I→E GABA projection. Average over batch and target cells only after retaining the full per-cell traces.
  + Compute population spike counts at the native timestep. Any display-rate envelope is descriptive and does not drive conductance motion.
  + Detect E volleys with the registered population-count rhythm policy. Require each selected cycle to contain an E volley followed by an I volley.
  + Normalize only screen position. Display the physical minimum, maximum, and units beside each mapped variable.

  === 4.3 Analysis and visual mapping

  + Freeze cycle detection and trial selection before inspecting the final animation.
  + Show measured values at their native temporal resolution. Permit downsampling only when it preserves extrema, spike events, and cycle boundaries.
  + Keep E and I identity stable across pistons, traces, spikes, and the phase portrait. Pair colour with labels and shape.
  + State biological time and playback slowdown together. Mark the animation as one simulated trial and distinguish population means from individual-cell values.

  === 4.4 Scientific decision gates and budget

  + *Stop* if complete cycles do not produce a consistently oriented conductance trajectory.
  + *Revise the visual reduction* if population means hide the stochastic increments that generate them.
  + *Retain the two-variable engine* if it preserves cycle phase and next-volley timing nearly as well as the four-variable comparison.
  + *Label it an explanatory projection* if adding mean voltage materially reduces either error.
  + Keep the scout to one frozen network, five short trials, and the three registered analyses. Its evidence remains specific to this simulated operating point.
]
