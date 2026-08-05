#let meta = (
  title: "Reciprocal inhibition entrains detuned PING circuits",
  date: "2026-08-05",
  description: "A registered graph-native sweep shows a contiguous transition from uncoupled phase drift to active frequency/phase locking and then delay-dependent suppression in two independently driven PING circuits.",
  collection: "snnlang",
  status: "draft",
  order: 5,
)

#let r = json("/artifacts/data/exp078/numbers.json")
#let baseline = r.sweep.rows.at(0)
#let short = r.sweep.rows.at(1)
#let intermediate = r.sweep.rows.at(2)
#let half = r.sweep.rows.at(6)
#let strong-short = r.sweep.rows.at(10)
#let selected = r.calibration.selected
#let n(x, digits: 3) = calc.round(x, digits: digits)
#let hz(x) = str(n(x)) + " Hz"

#let body = [
  == Abstract

  Reciprocal inhibition entrained two independently driven, slightly detuned
  PING circuits without a shared input or simulator change. The uncoupled pair
  oscillated at #hz(selected.metrics.populations.a.dominant_frequency_hz) and
  #hz(selected.metrics.populations.b.dominant_frequency_hz), a
  #n(100 * selected.detuning_fraction, digits: 2)% separation, with phase-locking
  value #n(baseline.metrics.synchrony.plv) and #hz(baseline.metrics.synchrony.frequency_difference_hz)
  residual frequency difference. Reciprocal GABA coupling at strength
  #short.variant.strength and #short.variant.delay_ms ms reduced that difference
  to #hz(short.metrics.synchrony.frequency_difference_hz), raised phase locking
  to #n(short.metrics.synchrony.plv), and raised mean 30–80 Hz coherence from
  #n(baseline.metrics.synchrony.gamma_coherence) to
  #n(short.metrics.synchrony.gamma_coherence). Four registered cells satisfy all
  locking gates with an adjacent locked neighbour. Strong short/intermediate
  coupling instead silenced circuit B. This one-seed result demonstrates the
  mechanism and its suppression boundary; it does not estimate generality.

  == Registered goal

  #raw(read("/artifacts/data/exp078/goal.txt"), block: true, lang: "text")

  == Design registered before coupling

  The hypothesis was that moderate reciprocal I→E GABA coupling would entrain
  two active, independently driven PING circuits while zero coupling permitted
  drift and excessive coupling could suppress activity. Both circuits contain
  #r.registration.populations.excitatory_per_circuit excitatory and
  #r.registration.populations.inhibitory_per_circuit inhibitory conductance
  neurons. Independent Poisson generators drive
  #r.registration.populations.input_channels_per_circuit channels per circuit;
  their seeds are archived separately, so no common-input synchrony is possible.
  Every condition uses seed #r.registration.seed, a #r.registration.simulation.dt_ms
  ms timestep, #r.registration.simulation.duration_ms ms duration, and discards
  the first #r.registration.simulation.transient_ms ms.

  The bounded calibration changed input rate and input weight only. The reusable
  snnlang PING component fixed all within-circuit E↔I settings. A candidate had
  to retain finite, non-silent, non-saturated E/I activity, a prominent 30–80 Hz
  rate peak in each circuit, and 5–15% detuning. The deterministic score selected
  the valid candidate nearest 10% detuning, with peak prominence and grid order
  breaking ties. Cross-coupling was not evaluated during calibration.

  Locking required both circuits to remain active, frequency difference no more
  than #n(100 * r.registration.locking_thresholds.max_frequency_difference_fraction_of_baseline)%
  of baseline, phase-locking gain at least
  #r.registration.locking_thresholds.min_plv_gain, mean-band coherence gain at
  least #r.registration.locking_thresholds.min_coherence_gain, phase-locking
  value at least #r.registration.locking_thresholds.min_half_window_plv in each
  half-window, and phase-offset drift no more than
  #r.registration.locking_thresholds.max_half_window_phase_offset_difference_rad
  rad. A passing cell also needed an orthogonally adjacent locked cell. Silent
  conditions were never eligible.

  == Calibration produced active detuned gamma

  #table(
    columns: (1.5fr, 1fr, 1fr),
    [Diagnostic], [Circuit A], [Circuit B],
    [Dominant frequency], [#hz(selected.metrics.populations.a.dominant_frequency_hz)], [#hz(selected.metrics.populations.b.dominant_frequency_hz)],
    [Peak prominence], [#n(selected.metrics.populations.a.peak_prominence, digits: 1)], [#n(selected.metrics.populations.b.peak_prominence, digits: 1)],
    [Excitatory rate], [#hz(selected.metrics.populations.a.e_rate_hz)], [#hz(selected.metrics.populations.b.e_rate_hz)],
    [Inhibitory rate], [#hz(selected.metrics.populations.a.i_rate_hz)], [#hz(selected.metrics.populations.b.i_rate_hz)],
    [E active-cell fraction], [#n(selected.metrics.populations.a.e_active_fraction)], [#n(selected.metrics.populations.b.e_active_fraction)],
    [I active-cell fraction], [#n(selected.metrics.populations.a.i_active_fraction)], [#n(selected.metrics.populations.b.i_active_fraction)],
  )

  Candidate #selected.index used input weight #selected.settings.input_weight
  and independent per-channel drive rates #hz(selected.settings.rate_a_hz) and
  #hz(selected.settings.rate_b_hz). All cells participated. The first analysis
  implementation took the maximum coherence anywhere in the gamma band; its
  uncoupled value was 0.992, making a +0.10 gain impossible. Before any coupling
  condition ran, the declared coherence statistic was frozen as the arithmetic
  mean of magnitude-squared coherence bins from 30 through 80 Hz. The corrected
  uncoupled baseline is #n(baseline.metrics.synchrony.gamma_coherence).

  == A contiguous locking region emerged

  #figure(
    image("/artifacts/data/exp078/coupling_heatmaps.png", width: 100%),
    caption: [The complete registered strength × delay sweep. Low-to-moderate
      coupling removes the frequency difference and increases phase locking and
      coherence. Delay changes the stable phase offset. Missing spectral values
      correspond to explicitly flagged suppression, not synchronization.],
  )

  #table(
    columns: (1.25fr, .75fr, .75fr, .75fr, .75fr, .8fr),
    [Condition], [Δf], [PLV], [Coherence], [Phase], [Lag],
    [Uncoupled], [#hz(baseline.metrics.synchrony.frequency_difference_hz)], [#n(baseline.metrics.synchrony.plv)], [#n(baseline.metrics.synchrony.gamma_coherence)], [#n(baseline.metrics.synchrony.mean_phase_difference_rad)], [#n(baseline.metrics.synchrony.cross_correlation_lag_ms, digits: 1) ms],
    [0.2, short], [#hz(short.metrics.synchrony.frequency_difference_hz)], [#n(short.metrics.synchrony.plv)], [#n(short.metrics.synchrony.gamma_coherence)], [#n(short.metrics.synchrony.mean_phase_difference_rad)], [#n(short.metrics.synchrony.cross_correlation_lag_ms, digits: 1) ms],
    [0.2, intermediate], [#hz(intermediate.metrics.synchrony.frequency_difference_hz)], [#n(intermediate.metrics.synchrony.plv)], [#n(intermediate.metrics.synchrony.gamma_coherence)], [#n(intermediate.metrics.synchrony.mean_phase_difference_rad)], [#n(intermediate.metrics.synchrony.cross_correlation_lag_ms, digits: 1) ms],
    [0.5, half period], [#hz(half.metrics.synchrony.frequency_difference_hz)], [#n(half.metrics.synchrony.plv)], [#n(half.metrics.synchrony.gamma_coherence)], [#n(half.metrics.synchrony.mean_phase_difference_rad)], [#n(half.metrics.synchrony.cross_correlation_lag_ms, digits: 1) ms],
  )

  The four locked cells are 0.2/short, 0.2/intermediate, 0.5/half-period,
  and 1.0/half-period. Each has an orthogonally adjacent locked neighbour, so
  the result is not an isolated best cell. Short and intermediate delay at
  strength 0.2 settle near an anti-phase offset of
  #n(short.metrics.synchrony.mean_phase_difference_rad) and
  #n(intermediate.metrics.synchrony.mean_phase_difference_rad) rad. Half-period
  delay changes the locked offset to #n(half.metrics.synchrony.mean_phase_difference_rad)
  rad and the cross-correlation lag to
  #n(half.metrics.synchrony.cross_correlation_lag_ms, digits: 1) ms. The sign and
  size of phase/lag therefore change systematically with the delay regime.

  == Rasters and rates distinguish locking from silence

  #figure(
    image("/artifacts/data/exp078/matched_rasters.png", width: 100%),
    caption: [Matched excitatory rasters after the transient for the uncoupled
      baseline, the first registered locked condition, and the strongest
      half-period condition. Inputs, seed, initial state, and within-circuit
      parameters are identical across rows.],
  )

  #figure(
    image("/artifacts/data/exp078/population_rates.png", width: 100%),
    caption: [Smoothed excitatory population rates for the same conditions.
      The coupled pair shares a rhythm with a stable offset; the baseline drifts.],
  )

  At strength #strong-short.variant.strength and short delay, circuit B has
  #hz(strong-short.metrics.populations.b.e_rate_hz) excitatory and
  #hz(strong-short.metrics.populations.b.i_rate_hz) inhibitory activity, while
  circuit A remains at #hz(strong-short.metrics.populations.a.e_rate_hz) and
  #hz(strong-short.metrics.populations.a.i_rate_hz). The condition is explicitly
  inactive and spectrally invalid. Strength 2/intermediate and strengths 4 at
  short/intermediate delay show the same one-sided suppression boundary. The
  half-period delay avoids complete silence even at the largest strengths, but
  those cells fail the full locking gates.

  == Graph-native execution and evidence

  #figure(
    image("/artifacts/data/exp078/representative_graph.svg", width: 100%),
    caption: [Representative reciprocal graph. Each inhibitory population sends
      a delayed GABA projection to the other circuit's excitatory population;
      the two input nodes are independent.],
  )

  All #r.sweep.condition_count conditions differ only through graph data. Each
  archives its graph, authenticated manifest, canonical diagrams, graph digest,
  raw named E/I, voltage, and conductance recordings, firing-rate and spectral
  traces, runtime, and peak traced Python memory. The complete registered sweep
  took #r.duration locally. Simulator edits were #r.exit.simulator_edits and paid
  compute cost was \$#r.exit.paid_compute_usd.

  The first complete sweep attempt computed all conditions but failed atomically
  before publication: derived rate arrays had overwritten raw spike keys in the
  figure map. Commit `a781c7f` records that killed attempt and an experiment-side
  key fix. The unchanged rerun published the evidence in commit `03bfbe9`.
  `tools/snn`, exp077, the registered grid, input realization, metrics, and gates
  were unchanged.

  == Conclusion and limitations

  The registered success criterion passes. Reciprocal inhibition alone moves
  this pair from independent phase drift into a contiguous active locking region,
  and excessive short/intermediate coupling produces suppression rather than a
  misleading synchrony score. This supports graph-native Milestone 4 as a
  mechanism demonstration and shows that scientifically meaningful variants can
  be authored in graph data without simulator edits.

  The evidence has one exploratory seed, one calibrated oscillator pair, dense
  connectivity, and one set of intrinsic PING parameters. It does not estimate
  robustness across seeds, circuit sizes, detuning ranges, input statistics, or
  biological preparations. The run manifest is worktree-dirty only because a
  concurrent Demolab preview regenerated tracked PDFs; its code-specific dirty
  flag is false and no patch affected execution.

  == Timestamped activity log

  #for event in r.activity [
    *#event.timestamp* \
    #event.event

  ]
]
