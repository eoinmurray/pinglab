#import "contents.typ": with-contents
#import "/.demolab/lib.typ": data-json, data-image
#import "run-inputs.typ": data-file, inputs-ready, pending-report
#import "run-view.typ": with-datasets, run-view
#import "run-inputs.typ": input-assets
#import "/.demolab/lib.typ": cite, reference-list
#let data-file = data-file.with(article: "exp086")
#let report-image(path, ..args) = context {
  let graphic = data-image(data-file(path), ..args)
  if target() == "html" {
    html.elem("div", attrs: (class: "exp086-figure"), graphic)
  } else { graphic }
}

#let meta = (
  status: "[▦ DATA]",
  title: "Lowet 2017",
  date: "2026-08-19",
  updated_at: "2026-08-29",
  description: "Reduce coupling at fixed detuning and test whether two PING networks develop cortical-like intermittent phase attraction.",
  collection: "demo",
  order: 2,
)

#let inputs = ("exp086",)
#let preview-figures = (
  (path: "exp086/network.svg", label: "network"),
  (path: "exp086/uncoupled.png", label: "uncoupled"),
  (path: "exp086/coupling_regimes_measured.png", label: "coupling regimes measured"),
  (path: "exp086/intermittent_attraction_measured.png", label: "intermittent attraction measured"),
)

// Keep calculations lazy: absent inputs never become fabricated results.
#let render-report(data-file) = [
#context if target() == "html" {
  html.elem("style", ".exp086-figure img {height:auto;max-width:100%;}")
}
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
  mean-frequency-hz: calc.round((uncoupled.network_a.frequency_hz + uncoupled.network_b.frequency_hz) / 2, digits: 1),
  detuning-hz: calc.round(uncoupled.network_a.frequency_hz - uncoupled.network_b.frequency_hz, digits: 1),
  relative-detuning: calc.round(2 * (uncoupled.network_a.frequency_hz - uncoupled.network_b.frequency_hz) / (uncoupled.network_a.frequency_hz + uncoupled.network_b.frequency_hz), digits: 3),
  k-ee-us: 0.08,
  k-ei-us: 0.08,
  delay-ms: 2,
  phase-concentration: strong-concentration,
  residual-hz: calc.round(strong.network_a.frequency_hz - strong.network_b.frequency_hz, digits: 2),
)

#let body = [
  == Abstract

  In macaque V1, nearby gamma rhythms did not hold one fixed relative-phase position. Their relative phase continued to move, but its velocity slowed near positions where coupling made the two instantaneous frequencies more similar. This phase-dependent slowing concentrated the observed phase differences around those positions despite continuing phase slips#cite(1). The effect could create recurring windows for communication without permanent synchronization. I reproduced that qualitative regime in one fixed-input PING trajectory by reducing equal reciprocal coupling from 0.08 to #selected-k µS. The networks accumulated #selected.phase_slips whole net phase windings, but their relative phase remained concentrated around a preferred position. Removing coupling produced #uncoupled.phase_slips whole net windings and an almost uniform phase distribution. This is a controlled demonstration, not a claim about reliability across inputs or seeds.

  #run-view("exp086", inputs)

  == Prior work

  Lowet et al. (2015) mapped synchronization across detuning and coupling in two 80-E, 20-I PING networks with reciprocal E-to-E and E-to-I fan-in eight#cite(2). Lowet et al. (2017) found the corresponding signature in macaque V1: coupling reduced instantaneous frequency differences near preferred phases, producing imperfect rather than permanent synchronization#cite(1).

  The shared parameter-space axes are uncoupled frequency detuning $Delta f_"detune"$ and measured effective interaction strength $epsilon$. Mean frequency $accent(f, macron)$ sets the timescale; phase noise controls how sharply the rhythms lock.

  #table(
    columns: (1.05fr, 1.55fr, 1.55fr, 1.45fr),
    [*Quantity*], [*Lowet regime*], [*Starting point*], [*First move*],
    [Mean frequency $accent(f, macron)$], [Approximately 30--40 Hz in the illustrated model and macaque conditions], [#start.mean-frequency-hz Hz], [Hold fixed initially],
    [Detuning $Delta f_"detune"$], [Illustrated macaque examples span approximately 2.8--4.8 Hz], [#start.detuning-hz Hz, or #start.relative-detuning of $accent(f, macron)$], [Hold near the starting value],
    [Effective interaction $epsilon$], [Phase-dependent frequency modulation is approximately 1.6--1.8 Hz in representative macaque conditions], [Not fitted; strong coupling leaves a mean frequency difference of #start.residual-hz Hz], [Describe the correction; fit $epsilon$ later],
    [Synchronization], [Representative phase-locking values approximately 0.3--0.5 with phase slips], [Phase concentration #start.phase-concentration with no whole net windings], [Weaken coupling first],
  )

  *Coupling units.* Both models divide total pathway strength across eight afferents. Lowet reports summed conductance density in mS/cm²; the present model uses summed absolute conductance in µS. Normalizing each by target leak conductance places this model's E-to-E and E-to-I strengths at $kappa = 1.6$ and $0.8$, within Lowet's broad sweep but more E-to-E dominated#cite(2). Here $kappa$ is cross-network conductance divided by target leak conductance. The present neurons are conductance-based leaky integrate-and-fire cells; Lowet used Hodgkin–Huxley cells. Leak normalization compares scales, not dynamical equivalence.

  == Results

  === Uncoupled rhythms <result-1-uncoupled-rhythms>

    #figure(
      report-image("exp086/uncoupled.png", width: 100%, alt: "Uncoupled population rhythms and circulating relative phase."),
      caption: [Both uncoupled networks maintained regular PING rhythms, but Network A ran at #uncoupled-frequency-a Hz and Network B at #uncoupled-frequency-b Hz. Their relative phase therefore circulated, accumulating #uncoupled.phase_slips whole net windings with phase concentration #uncoupled-concentration.],
    )

  === Coupling boundary <result-2-coupling-boundary>

    #figure(
      report-image("exp086/coupling_regimes_measured.png",
        width: 100%,
        alt: "Measured comparison of strong, intermediate, and absent coupling.",
      ),
      caption: [One fixed-input trajectory per condition. At 0.08 µS, relative phase concentrated near one position with concentration #strong-concentration. At #selected-k µS, it accumulated #selected.phase_slips whole net phase windings but repeatedly slowed near one phase region. With no coupling, it accumulated #uncoupled.phase_slips whole net windings and approached uniform drift.],
    )

  === Intermittent phase attraction <result-3-intermittent-attraction>

    #figure(
      report-image("exp086/intermittent_attraction_measured.png",
        width: 100%,
        alt: "Measured four-panel signature of intermittent phase attraction.",
      ),
      caption: [At equal reciprocal coupling $K = #selected-k$ µS, relative phase continued to slip but had concentration #selected-concentration. Its distribution peaked at #selected-preferred rad and reached #selected-density-ratio times the mean density. Across 24 phase bins, the smallest absolute mean velocity occurred at #selected-slow rad, #selected-alignment rad away from the density peak. The displayed velocity trace uses 8 ms Gaussian smoothing. This single trajectory shows the Lowet-style signature, but does not establish its reliability across trials.],
    )

  == Methods

  #block(inset: 10pt, fill: rgb("f3f0e8"), radius: 3pt)[
    *One fixed-input trajectory was simulated per coupling condition. Every condition reused the same saved network state and pre-generated input spike trains.*
  ]

  + *Define the starting rhythms.* Each network contains an excitatory population, an inhibitory population, and a local E-to-I-to-E feedback loop. Reciprocal excitation targets both E and I populations with $K_(E E) = #start.k-ee-us$ µS, $K_(E I) = #start.k-ei-us$ µS, and delay $d = #start.delay-ms$ ms. I drove Networks A and B at #start.input-a-hz and #start.input-b-hz Hz. I ran them without cross-network coupling and confirmed regular rhythms near mean frequency #start.mean-frequency-hz Hz and detuning #start.detuning-hz Hz. I interpolated phase between consecutive excitatory volleys and calculated their wrapped and unwrapped relative phase. See #link(<result-1-uncoupled-rhythms>)[Uncoupled rhythms].

    #figure(
      report-image("exp086/network.svg",
        width: 100%,
        alt: "Implemented topology of two PING networks with local excitatory-inhibitory loops and reciprocal excitatory coupling.",
      ),
      caption: [Implemented topology. Networks A and B each contain a local E-to-I-to-E PING loop. Reciprocal projections target E and I populations with independently controlled strengths.],
    )

  + *Reduce coupling while everything else stays the same.* I saved the network immediately before coupling began. I replayed from that point several times with the same neuron states and the same pre-generated input spike trains after that point. I changed only the coupling strength $K$: I began at #start.k-ee-us µS, then used progressively weaker values down to zero. E-to-E and E-to-I coupling always used the same value. I used nine values from 0.08 to 0.00 µS in 0.01 µS steps, each with a 500 ms uncoupled prefix and a 4,500 ms coupled suffix. I excluded the first 300 ms after coupling from phase measurements. I ran one trajectory at each value of $K$; repeated seeds and trials are outside this experiment. For each coupling strength, I tracked the phase gap between the two rhythms. Strong coupling may keep that gap fixed. Weaker coupling may allow one rhythm to repeatedly gain a full cycle on the other, producing phase slips. I looked for the intermediate behaviour reported by Lowet et al. (2017): phase slips continue, but the phase gap repeatedly slows near one preferred value, making that value more common#cite(1). Among nonzero, nonmaximal coupling conditions with at least two whole net windings, I selected the largest product of phase concentration, peak-to-mean density, nonnegative slowing fraction and the exponential of negative angular alignment error. I counted whole net windings by taking the floor of the absolute unwrapped phase change divided by $2 pi$; this does not count every reversing slip event. This demonstrates the behaviour once; it does not establish reliability across seeds. See #link(<result-2-coupling-boundary>)[Coupling boundary].


  + *Relate phase velocity to phase position.* For the selected condition, I estimated each network's instantaneous frequency from consecutive excitatory volleys. Relative-phase velocity is $v_phi = 2 pi (f_A - f_B)$, where $f_A$ and $f_B$ are the instantaneous frequencies of Networks A and B in Hz, and $v_phi$ is in rad/s. I grouped $v_phi$ by wrapped relative phase $phi$ in radians, and compared the resulting velocity curve with the distribution of $phi$. Intermittent attraction requires continued phase slips, a non-uniform phase distribution, and lower absolute velocity near the distribution's preferred position. See #link(<result-3-intermittent-attraction>)[Intermittent phase attraction].

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

#let report-body = if inputs-ready(data-file, inputs) {
  render-report(data-file)
} else {
  pending-report(
    data-file, inputs,
    [Can weaker coupling produce intermittent rather than permanent phase attraction? Compare phase slips, phase concentration, and relative-phase velocity in relation to Lowet 2017.],
    preview-figures, json-inputs: ("exp086",),
  )
}

#let meta = meta + (assets: input-assets("exp086", inputs))
#let body = with-datasets("exp086", inputs, report-body, placed: inputs-ready(data-file, inputs))
#let body = with-contents(body)
