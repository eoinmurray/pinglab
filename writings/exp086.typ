#import "/.demolab/lib.typ": data-json, data-image
#import "run-inputs.typ": data-file, inputs-ready, pending-report
#import "/.demolab/lib.typ": cite, reference-list
#let data-file = data-file.with(article: "exp086")

#let meta = (
  status: "Implemented",
  title: "Lowet 2017",
  date: "2026-08-19",
  description: "Reduce coupling at fixed detuning and test whether two PING networks develop cortical-like intermittent phase attraction.",
  collection: "demo",
  order: 2,
)

#let inputs = ("exp086",)
#let preview-figures = (
  (path: "exp086/network.svg", label: "network"),
  (path: "exp086/uncoupled.png", label: "uncoupled"),
  (path: "exp086/coupling_regimes.svg", label: "coupling regimes"),
  (path: "exp086/coupling_regimes_measured.png", label: "coupling regimes measured"),
  (path: "exp086/intermittent_attraction.svg", label: "intermittent attraction"),
  (path: "exp086/intermittent_attraction_measured.png", label: "intermittent attraction measured"),
)

// Keep calculations lazy: absent inputs never become fabricated results.
#let render-report(data-file) = [
#let run = data-json(data-file("exp086/numbers.json"))
#let trajectories = run.trajectories
#let selected = run.selected_intermediate
#let strong = trajectories.filter(row => calc.abs(row.k - 0.08) < 0.0001).first()
#let uncoupled = trajectories.filter(row => calc.abs(row.k) < 0.0001).first()
#let selected-k = calc.round(selected.k, digits: 2)
#let selected-concentration = calc.round(selected.phase_concentration, digits: 3)
#let selected-density-ratio = calc.round(selected.density_peak_to_mean, digits: 1)
#let selected-preferred = calc.round(selected.preferred_phase_rad, digits: 2)
#let selected-slow = calc.round(selected.slow_phase_rad, digits: 2)
#let selected-alignment = calc.round(selected.phase_alignment_error_rad, digits: 2)
#let strong-concentration = calc.round(strong.phase_concentration, digits: 3)
#let uncoupled-concentration = calc.round(uncoupled.phase_concentration, digits: 3)
#let uncoupled-frequency-a = calc.round(uncoupled.network_a.frequency_hz, digits: 1)
#let uncoupled-frequency-b = calc.round(uncoupled.network_b.frequency_hz, digits: 1)


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

  In macaque V1, nearby gamma rhythms did not hold one fixed relative-phase position. Their relative phase continued to move, but its velocity slowed near positions where coupling made the two instantaneous frequencies more similar. This phase-dependent slowing concentrated the observed phase differences around those positions despite continuing phase slips#cite(1). The effect could create recurring windows for communication without permanent synchronization. We reproduced that qualitative regime in one fixed-input PING trajectory by reducing equal reciprocal coupling from 0.08 to #selected-k µS. The networks completed #selected.phase_slips phase slips, but their relative phase remained concentrated around a preferred position. Removing coupling produced #uncoupled.phase_slips slips and an almost uniform phase distribution. This is a controlled demonstration, not a claim about reliability across inputs or seeds.

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
    *All three methods were run once. The starting values and topology were sourced from exp085. Every coupling condition reused the same saved network state and pre-generated input spike trains.*
  ]

  + *Define the starting rhythms.* Each network contains an excitatory population, an inhibitory population, and a local E-to-I-to-E feedback loop. Reciprocal excitation targets both E and I populations with $K_(E E) = #start.k-ee-us$ µS, $K_(E I) = #start.k-ei-us$ µS, and delay $d = #start.delay-ms$ ms. Drive Networks A and B at #start.input-a-hz and #start.input-b-hz Hz. Run them without cross-network coupling and confirm regular rhythms near mean frequency #start.mean-frequency-hz Hz and detuning #start.detuning-hz Hz. Interpolate phase between consecutive excitatory volleys and calculate their wrapped and unwrapped relative phase. Generates #link("#result-1-uncoupled-rhythms")[Result 1].

    #figure(
      data-image(data-file("exp086/network.svg"),
        width: 100%,
        alt: "Implemented topology of two PING networks with local excitatory-inhibitory loops and reciprocal excitatory coupling.",
      ),
      caption: [Implemented topology. Networks A and B each contain a local E-to-I-to-E PING loop. Reciprocal projections target E and I populations with independently controlled strengths.],
    )

  + *Reduce coupling while everything else stays the same.* Save the network immediately before coupling begins. Replay from that point several times with the same neuron states and the same pre-generated input spike trains after that point. Change only the coupling strength $K$: begin at #start.k-ee-us µS, then use progressively weaker values down to zero. E-to-E and E-to-I coupling always use the same value. Run one trajectory at each value of $K$; repeated seeds and trials are outside this experiment. For each coupling strength, track the phase gap between the two rhythms. Strong coupling may keep that gap fixed. Weaker coupling may allow one rhythm to repeatedly gain a full cycle on the other, producing phase slips. We are looking for the intermediate behaviour reported by Lowet et al. (2017): phase slips continue, but the phase gap repeatedly slows near one preferred value, making that value more common#cite(1). Select one trajectory that shows this pattern. This demonstrates the behaviour once; it does not establish reliability across seeds. Generates #link("#result-2-coupling-boundary")[Result 2].

    The expected and measured coupling regimes are placed together in #link("#result-2-coupling-boundary")[Result 2].

  + *Relate phase velocity to phase position.* For the selected condition, estimate each network's instantaneous frequency from consecutive excitatory volleys. Relative-phase velocity is $v_theta = 2 pi (f_A - f_B)$, where $f_A$ and $f_B$ are the instantaneous frequencies of Networks A and B. Group $v_theta$ by wrapped relative-phase position $theta$, and compare the resulting velocity curve with the distribution of $theta$. Intermittent attraction requires continued phase slips, a non-uniform phase distribution, and lower absolute velocity near the distribution's preferred position. Generates #link("#result-3-intermittent-attraction")[Result 3].

    The expected signature and its measured counterpart are placed together in #link("#result-3-intermittent-attraction")[Result 3].

  == Results

  + <result-1-uncoupled-rhythms> *Uncoupled rhythms.*

    #figure(
      data-image(data-file("exp086/uncoupled.png"), width: 100%),
      caption: [Both uncoupled networks maintain regular PING rhythms, but Network A runs at #uncoupled-frequency-a Hz and Network B at #uncoupled-frequency-b Hz. Their relative phase therefore circulates, completing #uncoupled.phase_slips slips with phase concentration #uncoupled-concentration.],
    )

  + <result-2-coupling-boundary> *Coupling boundary.*

    #figure(
      [
        #align(center)[*Expected — schematic*]
        #data-image(data-file("exp086/coupling_regimes.svg"),
          width: 100%,
          alt: "Schematic comparison of strong, intermediate, and absent coupling.",
        )
        #v(8pt)
        #align(center)[*Observed — measured*]
        #data-image(data-file("exp086/coupling_regimes_measured.png"),
          width: 100%,
          alt: "Measured comparison of strong, intermediate, and absent coupling.",
        )
      ],
      caption: [Expected coupling regimes above and measured trajectories below. The schematic is qualitative; aligned columns connect each proposed regime to its observation. At 0.08 µS, relative phase remains fixed with concentration #strong-concentration. At #selected-k µS, it completes #selected.phase_slips slips but repeatedly slows near one phase region. With no coupling, it completes #uncoupled.phase_slips slips and approaches uniform drift.],
    )

  + <result-3-intermittent-attraction> *Intermittent phase attraction.*

    #figure(
      [
        #align(center)[*Expected — schematic*]
        #data-image(data-file("exp086/intermittent_attraction.svg"),
          width: 100%,
          alt: "Schematic four-panel signature of intermittent phase attraction.",
        )
        #v(8pt)
        #align(center)[*Observed — measured*]
        #data-image(data-file("exp086/intermittent_attraction_measured.png"),
          width: 100%,
          alt: "Measured four-panel signature of intermittent phase attraction.",
        )
      ],
      caption: [Expected intermittent-attraction signature above and measured result below. The schematic is qualitative; matching panel positions connect each proposed relationship to its observation. At $K = #selected-k$ µS, relative phase continues to slip but has concentration #selected-concentration. Its distribution peaks at #selected-preferred rad and reaches #selected-density-ratio times the mean density. The smallest absolute mean velocity occurs at #selected-slow rad, #selected-alignment rad away—one analysis bin—from the density peak. This single trajectory shows the Lowet-style signature, but does not establish its reliability across trials.],
    )

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
#body
]

#let body = if inputs-ready(data-file, inputs) {
  render-report(data-file)
} else {
  pending-report(
    data-file, inputs,
    [Can weaker coupling produce intermittent rather than permanent phase attraction? Compare phase slips, phase concentration, and relative-phase velocity in relation to Lowet 2017.],
    preview-figures, json-inputs: ("exp086",),
  )
}
