#import "/.demolab/lib.typ": cite, reference-list, video

#let meta = (
  title: "Demonstrating Diesmann 1999",
  date: "2026-08-20",
  description: "Test whether a feedforward spiking network preserves some pulse packets, loses others, and draws surviving packets toward a stable form.",
  collection: "snnlang",
  status: "draft",
  order: 13,
)

#let r = json("/artifacts/data/exp087/numbers.json")
#let op = r.operating_point
#let bg = op.background
#let weak = r.representative_packets.filter(packet => packet.id == "weak").first()
#let broad = r.representative_packets.filter(packet => packet.id == "broad").first()
#let oversized = r.representative_packets.filter(packet => packet.id == "oversized").first()

#let body = [
  == Abstract

  A synfire chain is a sequence of neuron pools connected in one direction. A brief packet of nearly synchronous spikes enters the first pool, which may generate a new packet in the next pool. Diesmann et al. (1999) showed that weak or diffuse packets disappear, while sufficiently strong packets can converge toward a stable size and temporal width#cite(1). Here, each network runs for #r.t_ms ms against balanced background activity averaging #calc.round(bg.mean_settled_rate_hz, digits: 2) Hz per neuron. A weak diffuse input produces no coherent volley. A broad packet grows from #broad.layers.first().alpha to #broad.layers.last().alpha neurons and sharpens, while an oversized narrow packet settles near the same state. This qualitatively demonstrates stable packet propagation within substantial ongoing activity.

  == Prior work

  Diesmann et al. identified two regimes separated by a boundary in packet state space. Below the boundary, activity dies. Above it, different input packets approach the same stable packet#cite(1). Our neuron model, background drive, and finite chain differ from the original model, so this is a qualitative demonstration rather than an exact reproduction.

  == Methods

  #block(inset: 10pt, fill: rgb("f3f0e8"), radius: 3pt)[
    *Methods 1--4 are complete. Every simulation uses one network seed and one fixed background realization.*
  ]

  + *Define the chain and the pulse packet.* Build #r.layers feedforward pools of #r.neurons_per_layer conductance-based leaky integrate-and-fire neurons. Each neuron receives exactly #r.feedforward_fan_in excitatory connections from the preceding pool, with total incoming strength #r.feedforward_total_strength_us µS and delay #r.feedforward_delay_ms ms. Independent excitatory and inhibitory background inputs produce ongoing irregular spikes. Adjust inhibitory background strength by pool so asynchronous feedforward activity does not raise the mean rate downstream. The first pool receives an artificial pulse packet through the same feedforward connection rule.

    #figure(
      image(
        "/artifacts/data/exp087/network.svg",
        width: 100%,
        alt: "SNNLANG diagram of a six-pool feedforward synfire chain. A pulse packet enters Pool 1, every pool projects to the next, and balanced excitatory and inhibitory background inputs reach all pools.",
      ),
      caption: [SNNLANG circuit export. A pulse packet enters Pool 1 and activity can travel through six feedforward pools. Independent excitatory and inhibitory background inputs reach every pool.],
    )

    Describe each packet by its size $alpha$, the number of input spikes, and its temporal spread $sigma$, the standard deviation of their spike times. A narrow packet concentrates excitation and is more likely to make the target pool fire. A broad packet distributes the same input over time and may fail to cross threshold.

    #figure(
      image(
        "/artifacts/data/exp087/packet_definition.svg",
        width: 100%,
        alt: "Design schematic defining pulse packet size alpha and temporal spread sigma, then comparing how narrow and broad packets sum in a target pool.",
      ),
      caption: [*Pulse packet definition.* Packet size $alpha$ counts spikes. Packet spread $sigma$ measures their timing dispersion. The curves illustrate why equal-sized packets need not have the same effect. They are not simulated data.],
    )

    Generates #link("#result-1-reference-propagation")[Result 1].

  + *Compare packets with different starting states.* Use one fixed background realization and discard the first 30 ms as settling time. Tune the background so every pool fires between 8 and 12 Hz per neuron. Test feedforward strengths 0.10, 0.12, and 0.14 µS. Select the weakest point at which an 80-spike, 3 ms reference packet reaches every pool while background-only coincidences remain below the packet threshold. Freeze those settings. Then test a weak diffuse packet, a broad strong packet, and a narrow oversized packet. Detect a packet when its 1 ms peak exceeds that pool's largest background-only peak by at least five spikes. Measure $alpha$ as the number of neurons participating within 3 ms of that peak and calculate their temporal spread $sigma$.

    #figure(
      image(
        "/artifacts/data/exp087/packet_fates.svg",
        width: 100%,
        alt: "Design schematic comparing a weak diffuse pulse packet that dies with two different pulse packets that converge toward the same stable packet across six pools.",
      ),
      caption: [*Expected packet fates.* A weak or diffuse packet should die. Two different packets inside the propagation regime should approach the same stable size and spread. These trajectories explain the comparison and are not simulated data. Result 2 reproduces this layout with measured spikes and packet statistics.],
    )

    Generates #link("#result-2-packet-fates")[Result 2].

  + *Map the packet state transformation.* Keep the tuned network and background input fixed. Test 12 packet sizes from 20 to 100 spikes and seven temporal spreads from 0.2 to 5 ms. At each pool, plot the measured packet state $(alpha_n, sigma_n)$ and draw an arrow to the state produced in the next pool, $(alpha_(n+1), sigma_(n+1))$. Classify a packet as extinct when no later pool produces a response volley. Locate the boundary between extinction and propagation, and test whether surviving trajectories approach one common state.

    #figure(
      image(
        "/artifacts/data/exp087/synfire_state_space.svg",
        width: 100%,
        alt: "Design schematic of pulse packet state space with packet size alpha, temporal spread sigma, an extinction boundary, and trajectories converging toward a stable packet.",
      ),
      caption: [*Expected packet state space.* The dashed line separates initial packets that disappear from those that propagate. Layer-to-layer arrows in the propagation basin should approach one stable packet. This is a design schematic, not simulated data. Result 3 shows the measured transformations.],
    )

    Generates #link("#result-3-packet-state-space")[Result 3].

  + *Render a looping raster hero.* Use the complete measured broad-packet run from Method 2. After the 30 ms settling period, scroll a 40 ms window through the remaining activity. Draw every measured spike as a conventional raster tick on a white field. Show background spikes in black, spikes inside the measured packet windows in cyan, and packet-input time as a red dashed line. Repeat the measured 30--100 ms segment to make an eight-second seamless loop. The repeated boundary is an animation device; the spikes within the segment come from one simulation.

    #figure(
      image(
        "/artifacts/data/exp087/raster_hero_storyboard.svg",
        width: 100%,
        alt: "Design frame for a white scientific raster animation with six pool bands, black background spikes, a cyan propagating packet, and a red input-time marker.",
      ),
      caption: [*Design frame for the hero animation.* Result 4 replaces these illustrative marks with every measured spike from the broad-packet run.],
    )

    Generates #link("#result-4-looping-raster-hero")[Result 4].

  == Results

  + <result-1-reference-propagation> *Reference packet propagation.*

    #figure(
      image("/artifacts/data/exp087/reference_propagation.png", width: 100%, alt: "Measured spike raster and packet statistics for a broad reference packet propagating through six synfire pools."),
      caption: [Reference-packet propagation at total feedforward strength #op.feedforward_strength_us µS. External background channels fire at #op.background_rate_hz Hz, producing #calc.round(bg.mean_settled_rate_hz, digits: 2) Hz mean neural output.],
    )

    The broad reference packet recruits #broad.layers.at(0).alpha neurons in Pool 1 and #broad.layers.at(1).alpha in Pool 2. It reaches #broad.layers.last().alpha neurons in Pool 6 while narrowing from #calc.round(broad.layers.at(0).sigma_ms, digits: 2) ms to #calc.round(broad.layers.last().sigma_ms, digits: 2) ms. Each pool regenerates one coherent volley above its background activity.

  + <result-2-packet-fates> *Three packet trajectories.*

    #figure(
      image("/artifacts/data/exp087/packet_fates_measured.png", width: 100%, alt: "Measured rasters and packet statistics showing one packet dying and two different packets converging to the same stable volley."),
      caption: [Measured packet fates. The upper panels show spikes grouped by pool. The lower panels show packet size $alpha$ in black and spread $sigma$ in red.],
    )

    The weak input produces no volley distinguishable from background. The broad and oversized packets reach #broad.layers.last().alpha and #oversized.layers.last().alpha participating neurons, with late-chain spreads of #calc.round(broad.layers.last().sigma_ms, digits: 2) and #calc.round(oversized.layers.last().sigma_ms, digits: 2) ms. Their final states are close despite very different initial temporal spreads.

  + <result-3-packet-state-space> *Packet state space.*

    #figure(
      image("/artifacts/data/exp087/packet_state_space.png", width: 100%, alt: "Measured packet-state map separating extinction from propagation beside three layer-to-layer trajectories."),
      caption: [Measured packet state space. Tiles classify #r.state_space.len() initial size-width combinations. Arrows follow the three representative packets from their inputs through successive pools.],
    )

    #r.state_space.filter(packet => packet.survives).len() of #r.state_space.len() tested packets reach Pool 6. The boundary shifts toward larger $alpha$ as $sigma$ increases: temporally diffuse packets require more spikes. Surviving trajectories approach a high-participation packet, while the weak trajectory moves directly to extinction.

  + <result-4-looping-raster-hero> *Looping raster hero.*

    #video("synfire_raster_hero.mp4", caption: [One measured broad-packet run shown as an eight-second seamless scrolling raster.])

    Black marks are genuine background-driven spikes from the measured run. Cyan marks are the same measured spikes when they fall inside a detected packet volley. The red dashed line marks input time. The raster repeats the settled 70 ms segment only to form a seamless loop; it does not splice different simulations together.

  == References

  #reference-list((
    (
      text: [Diesmann, M., Gewaltig, M.-O., and Aertsen, A. (1999). Stable propagation of synchronous spiking in cortical neural networks. Nature 402, 529--533.],
      doi: "10.1038/990101",
    ),
  ))
]
