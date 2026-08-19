#import "/.demolab/lib.typ": cite, reference-list

#let meta = (
  title: "Can weaker coupling produce intermittent gamma synchronization?",
  date: "2026-08-19",
  description: "Reduce coupling at fixed detuning and test whether two PING networks develop cortical-like intermittent phase attraction.",
  collection: "snnlang",
  status: "draft",
  order: 12,
)

#let start = (
  input-a-hz: 300,
  input-b-hz: 260,
  mean-frequency-hz: 48.7,
  detuning-hz: 4.0,
  relative-detuning: 0.082,
  k-ee-us: 0.08,
  k-ei-us: 0.08,
  delay-ms: 2,
  phase-concentration: 0.997,
)

#let body = [
  == Abstract

  In macaque V1, nearby gamma rhythms did not hold one fixed relative-phase position. Their relative phase continued to move, but its velocity slowed near positions where coupling made the two instantaneous frequencies more similar. This phase-dependent slowing concentrated the observed phase differences around those positions, producing a peaked distribution despite continuing phase slips#cite(1). The effect could create recurring windows for effective communication between cortical populations without forcing them into permanent synchronization. Starting from two rigidly phase-locked PING networks, this experiment reduces coupling and asks whether their relative phase begins to circulate while still slowing near a preferred position. Network structure, input, frequency difference, delay, and noise remain fixed. Broader parameter maps and oscillator modelling are left for later experiments.

  == Prior work

  Lowet et al. (2015) mapped synchronization across detuning and coupling in two 80-E, 20-I PING networks with reciprocal E-to-E and E-to-I fan-in eight#cite(2). Lowet et al. (2017) found the corresponding signature in macaque V1: coupling reduced instantaneous frequency differences near preferred phases, producing imperfect rather than permanent synchronization#cite(1).

  The shared parameter-space axes are uncoupled frequency detuning $Delta f$ and measured effective interaction strength $epsilon$. Mean frequency $bar(f)$ sets the timescale; phase noise controls how sharply the rhythms lock.

  #table(
    columns: (1.05fr, 1.55fr, 1.55fr, 1.45fr),
    [*Quantity*], [*Lowet regime*], [*Our starting point*], [*First move*],
    [Mean frequency $bar(f)$], [Approximately 30--40 Hz in the illustrated model and macaque conditions], [#start.mean-frequency-hz Hz], [Hold fixed initially],
    [Detuning $Delta f$], [Illustrated macaque examples span approximately 2.8--4.8 Hz], [#start.detuning-hz Hz, or #start.relative-detuning of $bar(f)$], [Hold near the starting value],
    [Effective interaction $epsilon$], [Phase-dependent frequency modulation is approximately 1.6--1.8 Hz in representative macaque conditions], [Not measured; the interaction cancels #start.detuning-hz Hz at the locked phase], [Describe the correction; fit $epsilon$ later],
    [Synchronization], [Representative phase-locking values approximately 0.3--0.5 with phase slips], [Phase concentration #start.phase-concentration with no late slips], [Weaken coupling first],
  )

  *Coupling units.* Both models divide total pathway strength across eight afferents. Lowet reports summed conductance density in mS/cm²; SNNLANG reports summed absolute conductance in µS. Normalizing each by target leak conductance places our E-to-E and E-to-I strengths at $kappa = 1.6$ and $0.8$, within Lowet's broad sweep but more E-to-E dominated#cite(2). Here $kappa$ is cross-network conductance divided by target leak conductance.

  == Methods

  #block(inset: 10pt, fill: rgb("f3f0e8"), radius: 3pt)[
    *This experiment has not been run. Its starting values and topology were sourced from exp085 and are fixed here as design inputs.*
  ]

  + *Define the starting rhythms.* Each network contains an excitatory population, an inhibitory population, and a local E-to-I-to-E feedback loop. Reciprocal excitation targets both E and I populations with $K_(E E) = #start.k-ee-us$ µS, $K_(E I) = #start.k-ei-us$ µS, and delay $d = #start.delay-ms$ ms. Drive Networks A and B at #start.input-a-hz and #start.input-b-hz Hz. Run them without cross-network coupling and confirm regular rhythms near mean frequency #start.mean-frequency-hz Hz and detuning #start.detuning-hz Hz. Interpolate phase between consecutive excitatory volleys and calculate their wrapped and unwrapped relative phase. Generates #link("#result-1-uncoupled-rhythms")[Result 1].

    #figure(
      image(
        "/artifacts/data/exp085/network.svg",
        width: 100%,
        alt: "Planned topology of two PING networks with local excitatory-inhibitory loops and reciprocal excitatory coupling.",
      ),
      caption: [Planned starting topology. Networks A and B each contain a local E-to-I-to-E PING loop. Reciprocal projections target E and I populations with independently controlled strengths.],
    )

  + *Reduce coupling while everything else stays the same.* Save the network immediately before coupling begins. Replay from that point several times with the same neuron states and the same pre-generated input spike trains after that point. Change only the coupling strength $K$: begin at #start.k-ee-us µS, then use progressively weaker values down to zero. E-to-E and E-to-I coupling always use the same value. Run one trajectory at each value of $K$; repeated seeds and trials are outside this experiment. For each coupling strength, track the phase gap between the two rhythms. Strong coupling may keep that gap fixed. Weaker coupling may allow one rhythm to repeatedly gain a full cycle on the other, producing phase slips. We are looking for the intermediate behaviour reported by Lowet et al. (2017): phase slips continue, but the phase gap repeatedly slows near one preferred value, making that value more common#cite(1). Select one trajectory that shows this pattern. This demonstrates the behaviour once; it does not establish reliability across seeds. Generates #link("#result-2-coupling-boundary")[Result 2].

  + *Relate phase velocity to phase position.* For the selected condition, estimate each network's instantaneous frequency from consecutive excitatory volleys. Relative-phase velocity is $v_theta = 2 pi (f_A - f_B)$, where $f_A$ and $f_B$ are the instantaneous frequencies of Networks A and B. Group $v_theta$ by wrapped relative-phase position $theta$, and compare the resulting velocity curve with the distribution of $theta$. Intermittent attraction requires continued phase slips, a non-uniform phase distribution, and lower absolute velocity near the distribution's preferred position. Generates #link("#result-3-intermittent-attraction")[Result 3].

  == Results

  + <result-1-uncoupled-rhythms> *Uncoupled rhythms.*
    - *Axes.* Time on the horizontal axis; normalized population activity on the vertical axis.
    - *Traces.* Excitatory and inhibitory activity from Networks A and B, followed by their wrapped relative-phase position.
    - *Why.* Confirm that both networks generate regular gamma rhythms with a controlled intrinsic frequency difference before coupling.
    - *Expectation.* Each network maintains a clean PING rhythm while their uncoupled relative phase repeatedly wraps.

  + <result-2-coupling-boundary> *Coupling boundary.*
    - *Axes.* Coupling scale $K$ on the horizontal axis. The upper panel shows phase concentration; the lower panel shows phase slips per second.
    - *Traces.* One trace in each panel across the fixed-detuning coupling sweep, with the starting point, uncoupled baseline, and selected intermittent condition marked.
    - *Why.* Test one controlled move from permanent locking toward the intermittent synchronization seen in cortex.
    - *Expectation.* Reducing coupling first introduces phase slips while phase concentration remains above the uncoupled baseline. Further reduction approaches unconstrained drift.

  + <result-3-intermittent-attraction> *Intermittent phase attraction.*
    - *Axes.* The first two panels show time against relative-phase position $theta$ and velocity $v_theta$. The third shows position $theta$ against mean velocity $v_theta$. The fourth shows the distribution of $theta$.
    - *Traces.* Wrapped phase position and phase velocity through time, followed by the position-binned velocity curve and phase-position distribution for the same condition.
    - *Why.* Test the specific Lowet 2017 signature without fitting a full oscillator model or mapping a second parameter.
    - *Expectation.* Position continues to wrap through phase slips, but absolute velocity is smallest near the position at which the phase distribution peaks.

  == References

  #reference-list((
    (
      text: [Lowet, E., Roberts, M. J., Peter, A., Gips, B., and De Weerd, P. (2017). A quantitative theory of gamma synchronization in macaque V1.],
      doi: "10.7554/eLife.26642",
    ),
    (
      text: [Lowet, E., Roberts, M., Hadjipapas, A., Peter, A., van der Eerden, J., and De Weerd, P. (2015). Input-dependent frequency modulation of cortical gamma oscillations shapes spatial synchronization and enables phase coding.],
      doi: "10.1371/journal.pcbi.1004072",
    ),
  ))
]
