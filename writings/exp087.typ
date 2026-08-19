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
#let weak = r.representative_packets.filter(packet => packet.id == "weak").first()
#let broad = r.representative_packets.filter(packet => packet.id == "broad").first()
#let oversized = r.representative_packets.filter(packet => packet.id == "oversized").first()

#let body = [
  == Abstract

  A synfire chain is a sequence of neuron pools connected in one direction. A brief packet of nearly synchronous spikes enters the first pool, which may generate a new packet in the next pool. Diesmann et al. (1999) showed that weak or diffuse packets disappear, while sufficiently strong packets can converge toward a stable size and temporal width#cite(1). At the selected SNNLANG operating point, a weak packet disappears after two pools. A broad packet grows and sharpens, while an oversized narrow packet relaxes immediately. Both surviving packets reach the same 100-spike, approximately 0.06 ms state. A grid of initial packet sizes and widths separates extinction from propagation. This qualitatively demonstrates the stable-packet attractor described by Diesmann et al.

  == Prior work

  Diesmann et al. identified two regimes separated by a boundary in packet state space. Below the boundary, activity dies. Above it, different input packets approach the same stable packet#cite(1). Our neuron model, background drive, and finite chain differ from the original model, so this is a qualitative demonstration rather than an exact reproduction.

  == Methods

  #block(inset: 10pt, fill: rgb("f3f0e8"), radius: 3pt)[
    *Methods 1--4 are complete. Every simulation uses one network seed and one fixed background realization.*
  ]

  + *Define the chain and the pulse packet.* Build #r.layers feedforward pools of #r.neurons_per_layer conductance-based leaky integrate-and-fire neurons. Each neuron receives exactly #r.feedforward_fan_in excitatory connections from the preceding pool, with total incoming strength #r.feedforward_total_strength_us µS and delay #r.feedforward_delay_ms ms. Independent background spikes keep neurons near firing threshold without producing population volleys by themselves. The first pool receives an artificial pulse packet through the same feedforward connection rule.

    #figure(
      image(
        "/artifacts/data/exp087/network.svg",
        width: 100%,
        alt: "SNNLANG diagram of a six-pool feedforward synfire chain. A pulse packet enters Pool 1, every pool projects to the next, and independent background input reaches all pools.",
      ),
      caption: [SNNLANG circuit export. A pulse packet enters Pool 1 and activity can travel through six feedforward pools. Independent background input reaches every pool.],
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

  + *Compare packets with different starting states.* Use one fixed realization of the background input. Test feedforward strengths 0.33, 0.35, 0.37, and 0.40 µS at background rates 2, 5, and 10 Hz. Select the weakest point at which a 50-spike, 5 ms reference packet reaches every pool without secondary volleys. Freeze those network settings. Then test a weak diffuse packet, a broad strong packet, and a narrow oversized packet. For every pool, count the spikes in its response volley to obtain $alpha$, and calculate their spike-time standard deviation to obtain $sigma$.

    #figure(
      image(
        "/artifacts/data/exp087/packet_fates.svg",
        width: 100%,
        alt: "Design schematic comparing a weak diffuse pulse packet that dies with two different pulse packets that converge toward the same stable packet across six pools.",
      ),
      caption: [*Expected packet fates.* A weak or diffuse packet should die. Two different packets inside the propagation regime should approach the same stable size and spread. These trajectories explain the comparison and are not simulated data. Result 2 reproduces this layout with measured spikes and packet statistics.],
    )

    Generates #link("#result-2-packet-fates")[Result 2].

  + *Map the packet state transformation.* Keep the tuned network and background input fixed. Test 11 packet sizes from 20 to 90 spikes and seven temporal spreads from 0.2 to 5 ms. At each pool, plot the measured packet state $(alpha_n, sigma_n)$ and draw an arrow to the state produced in the next pool, $(alpha_(n+1), sigma_(n+1))$. Classify a packet as extinct when no later pool produces a response volley. Locate the boundary between extinction and propagation, and test whether surviving trajectories approach one common state.

    #figure(
      image(
        "/artifacts/data/exp087/synfire_state_space.svg",
        width: 100%,
        alt: "Design schematic of pulse packet state space with packet size alpha, temporal spread sigma, an extinction boundary, and trajectories converging toward a stable packet.",
      ),
      caption: [*Expected packet state space.* The dashed line separates initial packets that disappear from those that propagate. Layer-to-layer arrows in the propagation basin should approach one stable packet. This is a design schematic, not simulated data. Result 3 shows the measured transformations.],
    )

    Generates #link("#result-3-packet-state-space")[Result 3].

  + *Render a looping raster hero.* Use the measured spike times from the three packet trajectories in Method 2. Place neurons from Pools 1--6 in separate horizontal bands, with time moving horizontally. Scroll the raster at a constant speed. Show the weak packet fading out and the two surviving packets approaching the same compact volley. Incoming spikes must stop at each pool and a newly generated volley must appear in the next pool. Repeat the complete sequence at an exact interval and match the first and last frames to produce an eight-second seamless MP4 loop.

    #figure(
      image(
        "/artifacts/data/exp087/raster_hero_storyboard.svg",
        width: 100%,
        alt: "Approximate frame from a dark scrolling raster animation with six pool bands. A weak packet disappears while two other packets produce matching compact volleys.",
      ),
      caption: [*Approximate frame from the hero animation.* Result 4 replaces these illustrative marks with measured spikes. Packet size appears as spike density and packet spread appears as horizontal volley width.],
    )

    Generates #link("#result-4-looping-raster-hero")[Result 4].

  == Results

  + <result-1-reference-propagation> *Reference packet propagation.*

    #figure(
      image("/artifacts/data/exp087/reference_propagation.png", width: 100%, alt: "Measured spike raster and packet statistics for a broad reference packet propagating through six synfire pools."),
      caption: [Reference-packet propagation at total feedforward strength #op.feedforward_strength_us µS and background rate #op.background_rate_hz Hz.],
    )

    The broad reference packet produces #broad.layers.at(0).alpha spikes in Pool 1 and #broad.layers.at(1).alpha in Pool 2. It then reaches all #r.neurons_per_layer neurons while narrowing from #calc.round(broad.layers.at(0).sigma_ms, digits: 2) ms to approximately #calc.round(broad.layers.last().sigma_ms, digits: 2) ms. Each pool emits one regenerated volley.

  + <result-2-packet-fates> *Three packet trajectories.*

    #figure(
      image("/artifacts/data/exp087/packet_fates_measured.png", width: 100%, alt: "Measured rasters and packet statistics showing one packet dying and two different packets converging to the same stable volley."),
      caption: [Measured packet fates. The upper panels show spikes grouped by pool. The lower panels show packet size $alpha$ in black and spread $sigma$ in red.],
    )

    The weak packet falls from #weak.layers.at(0).alpha spikes to #weak.layers.at(1).alpha, then disappears. The broad and oversized packets both reach #broad.layers.last().alpha spikes with late-chain spreads of #calc.round(broad.layers.last().sigma_ms, digits: 2) and #calc.round(oversized.layers.last().sigma_ms, digits: 2) ms. Different initial packets therefore converge on the same measured state.

  + <result-3-packet-state-space> *Packet state space.*

    #figure(
      image("/artifacts/data/exp087/packet_state_space.png", width: 100%, alt: "Measured packet-state map separating extinction from propagation beside three layer-to-layer trajectories."),
      caption: [Measured packet state space. Tiles classify 77 initial size-width combinations. Arrows follow the three representative packets from their inputs through successive pools.],
    )

    #r.state_space.filter(packet => packet.survives).len() of #r.state_space.len() tested packets reach Pool 6. The boundary shifts toward larger $alpha$ as $sigma$ increases: temporally diffuse packets require more spikes. Surviving trajectories collapse toward the saturated 100-spike state, while the weak trajectory moves toward extinction.

  + <result-4-looping-raster-hero> *Looping raster hero.*

    #video("synfire_raster_hero.mp4", caption: [Measured spikes from the three packet trajectories, arranged as an eight-second seamless scrolling raster.])

    Red shows the packet that dies. Cyan shows the broad packet sharpening as each pool regenerates it. Amber begins as the oversized packet and becomes cyan after it reaches the same stable form. Spike marks fade at the horizontal boundaries so the periodic time window loops without a visible cut.

  == References

  #reference-list((
    (
      text: [Diesmann, M., Gewaltig, M.-O., and Aertsen, A. (1999). Stable propagation of synchronous spiking in cortical neural networks. Nature 402, 529--533.],
      doi: "10.1038/990101",
    ),
  ))
]
