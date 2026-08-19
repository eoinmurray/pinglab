#import "/.demolab/lib.typ": cite, reference-list

#let meta = (
  title: "Demonstrating Diesmann 1999",
  date: "2026-08-20",
  description: "Test whether a feedforward spiking network preserves some pulse packets, loses others, and draws surviving packets toward a stable form.",
  collection: "snnlang",
  status: "draft",
  order: 13,
)

#let r = json("/artifacts/data/exp087/numbers.json")

#let body = [
  == Abstract

  A synfire chain is a sequence of neuron pools connected in one direction. A brief packet of nearly synchronous spikes enters the first pool, which may generate a new packet in the next pool. Diesmann et al. (1999) showed that weak or diffuse packets disappear, while sufficiently strong packets can converge toward a stable size and temporal width#cite(1). This experiment asks whether a conductance-based SNNLANG chain can qualitatively demonstrate that result. It treats packet size $alpha$ and temporal spread $sigma$ as the packet's state, then follows that state through #r.layers pools. The network graph has been compiled, but no simulation has been run.

  == Prior work

  Diesmann et al. identified two regimes separated by a boundary in packet state space. Below the boundary, activity dies. Above it, different input packets approach the same stable packet#cite(1). Our neuron model, background drive, and finite chain differ from the original model, so this is a qualitative demonstration rather than an exact reproduction.

  == Methods

  #block(inset: 10pt, fill: rgb("f3f0e8"), radius: 3pt)[
    *Method 1 is compiled. Methods 2 and 3 have not been run. All Results are planned outputs.*
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

  + *Compare packets with different starting states.* Use one fixed realization of the background input. Tune only feedforward strength and background rate until one reference packet produces one volley in every pool, without secondary volleys. Freeze those network settings. Then test a weak, diffuse packet and two clearly different packets expected to propagate. For every pool, count the spikes in its response volley to obtain $alpha$, and calculate their spike-time standard deviation to obtain $sigma$. The weak packet should disappear. The other two should change from pool to pool but approach similar late-chain values.

    #figure(
      image(
        "/artifacts/data/exp087/packet_fates.svg",
        width: 100%,
        alt: "Design schematic comparing a weak diffuse pulse packet that dies with two different pulse packets that converge toward the same stable packet across six pools.",
      ),
      caption: [*Expected packet fates.* A weak or diffuse packet should die. Two different packets inside the propagation regime should approach the same stable size and spread. These trajectories explain the comparison and are not simulated data. Result 2 will reproduce this layout with measured spikes and packet statistics.],
    )

    Generates #link("#result-2-packet-fates")[Result 2].

  + *Map the packet state transformation.* Keep the tuned network and background input fixed. Test a compact grid of starting values for $alpha$ and $sigma$. At each pool, plot the measured packet state $(alpha_n, sigma_n)$ and draw an arrow to the state produced in the next pool, $(alpha_(n+1), sigma_(n+1))$. Classify a packet as extinct when no later pool produces a response volley. Look for a boundary between extinction and propagation, and for surviving trajectories that approach one common state.

    #figure(
      image(
        "/artifacts/data/exp087/synfire_state_space.svg",
        width: 100%,
        alt: "Design schematic of pulse packet state space with packet size alpha, temporal spread sigma, an extinction boundary, and trajectories converging toward a stable packet.",
      ),
      caption: [*Expected packet state space.* The dashed line separates initial packets that disappear from those that propagate. Layer-to-layer arrows in the propagation basin should approach one stable packet. This is a design schematic, not simulated data. Result 3 will replace it with measured transformations.],
    )

    Generates #link("#result-3-packet-state-space")[Result 3].

  + *Render a looping raster hero.* Use the measured spike times from the three packet trajectories in Method 2. Place neurons from Pools 1--6 in separate horizontal bands, with time moving horizontally. Scroll the raster at a constant speed. Show the weak packet fading out and the two surviving packets approaching the same compact volley. Incoming spikes must stop at each pool and a newly generated volley must appear in the next pool. Repeat the complete sequence at an exact interval and match the first and last frames to produce an eight-second seamless MP4 loop.

    #figure(
      image(
        "/artifacts/data/exp087/raster_hero_storyboard.svg",
        width: 100%,
        alt: "Approximate frame from a dark scrolling raster animation with six pool bands. A weak packet disappears while two other packets produce matching compact volleys.",
      ),
      caption: [*Approximate frame from the hero animation.* The final MP4 will replace these illustrative marks with measured spikes. Packet size appears as spike density and packet spread appears as horizontal volley width.],
    )

    Generates #link("#result-4-looping-raster-hero")[Result 4].

  == Results

  + <result-1-reference-propagation> *Reference packet propagation.*
    - *Axes.* In the upper plot, time on the horizontal axis and neuron index grouped by pool on the vertical axis. In the lower plots, pool number on the horizontal axis and either packet size $alpha$ or spread $sigma$ on the vertical axis.
    - *Traces.* Spike marks from all six pools, followed by the measured $alpha$ and $sigma$ of each response volley.
    - *Why.* Confirm that the selected operating point carries one clean packet through the complete chain.
    - *Expectation.* One volley appears in each pool. Its size and spread settle rather than vanishing or splitting into secondary volleys.

  + <result-2-packet-fates> *Three packet trajectories.*
    - *Axes.* Reproduce the three-column layout of the Method 2 schematic using measured data. Upper panels show time against pool number. Lower panels show pool number against packet size $alpha$ and spread $sigma$.
    - *Traces.* The weak diffuse packet, a broad strong packet, and a narrow oversized packet.
    - *Why.* Demonstrate both packet extinction and convergence from different starting states.
    - *Expectation.* The weak packet dies. The two surviving packets approach similar $alpha$ and $sigma$ values in later pools.

  + <result-3-packet-state-space> *Packet state space.*
    - *Axes.* Packet size $alpha$ on the horizontal axis and temporal spread $sigma$ on the vertical axis.
    - *Marks.* Each arrow connects a packet measured in one pool to the packet it produces in the next. Background regions distinguish extinction from propagation. Mark the inferred boundary and stable packet.
    - *Why.* Test the central Diesmann result: packet propagation behaves as an attraction process with an extinction boundary.
    - *Expectation.* Initial states separate into extinction and propagation regions. Surviving trajectories from different starting points converge toward one stable packet state.

  + <result-4-looping-raster-hero> *Looping raster hero.*
    - *Axes.* Time on the horizontal axis and neuron index, grouped into Pools 1--6, on the vertical axis. The visible time window scrolls continuously.
    - *Marks.* Short luminous marks show measured spikes. Muted red identifies the packet that dies. Cyan identifies propagating activity. A faint envelope shows each packet's temporal width without obscuring individual spikes.
    - *Why.* Present extinction, regeneration, and convergence in one immediately readable visual while retaining the natural raster representation of spiking activity.
    - *Expectation.* The weak packet fades before the final pool. The other packets begin with different sizes and widths but produce similar volleys by Pool 6. The eight-second MP4 loops without a visible cut or false impression that individual spikes pass through the pools.

  == References

  #reference-list((
    (
      text: [Diesmann, M., Gewaltig, M.-O., and Aertsen, A. (1999). Stable propagation of synchronous spiking in cortical neural networks. Nature 402, 529--533.],
      doi: "10.1038/990101",
    ),
  ))
]
