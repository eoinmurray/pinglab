#let meta = (
  title: "Reciprocal E→I coupling captures phase in large PING circuits",
  date: "2026-08-06",
  description: "Complete runtime-state continuation shows two independently driven 800 E / 200 I PING circuits converge from a wandering phase relation to a stable lag.",
  collection: "miscellaneous",
  status: "draft",
)

#let r = json("/artifacts/data/exp078/numbers.json")
#let refinement = json("/artifacts/data/exp078/refinement/results.json")
#let best = refinement.best
#let acquisition = json("/artifacts/data/exp078/acquisition/results.json")
#let acquired = acquisition.selection
#let acquisition-metrics = acquisition.metrics
#let baseline = r.sweep.rows.at(0)
#let short = r.sweep.rows.at(1)
#let intermediate = r.sweep.rows.at(2)
#let low-half = r.sweep.rows.at(3)
#let medium-intermediate = r.sweep.rows.at(5)
#let strong-short = r.sweep.rows.at(10)
#let selected = r.calibration.selected
#let n(x, digits: 3) = calc.round(x, digits: digits)
#let hz(x) = str(n(x)) + " Hz"

#let body = [
  == Abstract

  Reciprocal long-range excitation can capture the phase of two independently driven PING
  circuits containing 800 excitatory and 200 inhibitory neurons each in a narrow
  strength--delay region. Excitatory neurons in each circuit project to the
  other circuit's inhibitory population, which then inhibits its local
  excitatory population. The uncoupled pair oscillates at
  #hz(selected.metrics.populations.a.dominant_frequency_hz) and
  #hz(selected.metrics.populations.b.dominant_frequency_hz), a
  #n(100 * selected.detuning_fraction, digits: 2)% separation, with phase-locking
  value #n(baseline.metrics.synchrony.plv) and #hz(baseline.metrics.synchrony.frequency_difference_hz)
  residual frequency difference. A registered coarse sweep found no condition
  passing every joint gate. A declared exploratory refinement then found a
  contiguous neighborhood with zero frequency difference and high phase
  locking. Its strongest condition, E→I weight #best.strength and delay
  #best.delay_ms ms, reaches PLV #n(best.metrics.synchrony.plv), retains
  half-window PLVs #n(best.metrics.synchrony.half_1_plv) and
  #n(best.metrics.synchrony.half_2_plv), and limits half-window phase drift to
  #n(best.metrics.synchrony.half_phase_offset_difference_rad) rad. A stronger
  test continues a mature saved state once with static weight #acquired.strength
  coupling. It reaches the
  sustained-250-ms capture criterion after
  #n(acquisition-metrics.capture_time_ms, digits: 1) ms. The coupled phase is
  initially #n(acquisition-metrics.early_phase_error_rad) rad from its eventual
  lag, but its final-quarter median error falls to
  #n(acquisition-metrics.quartile_phase_errors_rad.at(3)) rad. During the final
  250 ms, 95% of the smoothed instantaneous frequency differences are below
  #hz(acquisition-metrics.late_frequency_difference_95_hz). This trajectory
  therefore shows acquisition across the observation window rather than beginning
  already synchronized.
  Strong coupling instead suppresses one circuit. This one-seed result is a
  reduced mechanism demonstration, not an estimate of biological generality.

  == Methods

  #figure(
    image(
      "/artifacts/data/exp078/representative_graph.svg",
      width: 100%,
      alt: "Two independently driven PING components connected by reciprocal delayed excitatory projections from each excitatory population to the other inhibitory population.",
    ),
    caption: [Experimental topology. Independent spike inputs drive circuits A
      and B, each containing #r.registration.populations.excitatory_per_circuit
      excitatory and #r.registration.populations.inhibitory_per_circuit
      inhibitory neurons. Dashed AMPA projections run from each E population to
      the other circuit's I population; local I→E projections close each PING
      loop. There is no shared input or direct cross-circuit inhibition.],
  )

  + *Construct the oscillator pair.* The hypothesis was that moderate reciprocal
    excitatory-to-inhibitory (E→I) AMPA coupling would entrain two active,
    independently driven PING circuits by recruiting inhibition in the other
    circuit. Zero coupling would permit drift, whereas excessive coupling could
    suppress activity. Each circuit contains
    #r.registration.populations.excitatory_per_circuit excitatory and
    #r.registration.populations.inhibitory_per_circuit inhibitory conductance
    neurons. Independent Poisson generators drive
    #r.registration.populations.input_channels_per_circuit channels per circuit.
    Their seeds are archived separately, so common-input synchrony is impossible.
    Every condition uses seed #r.registration.seed, a
    #r.registration.simulation.dt_ms ms timestep, and a
    #r.registration.simulation.duration_ms ms duration. Analysis excludes the
    first #r.registration.simulation.transient_ms ms.

    The superseded run connected each inhibitory population directly to the
    other circuit's excitatory population. That shortcut did not represent the
    intended long-range anatomy, so its results were discarded. Before this
    replacement sweep, the cross-circuit pathway was amended to E→I while the
    uncoupled calibration, input realization, strength and delay grids, initial
    conditions, analysis, and acceptance thresholds remained fixed.

    A subsequent scale amendment replaced the preliminary 40 E / 10 I circuits
    with the 800 E / 200 I circuits reported here. Calibration and the complete
    coupling sweep were rerun at the larger scale; none of the preliminary
    small-circuit measurements enters this report.

  + *Calibrate detuned gamma activity.* The bounded calibration changed input
    rate and input weight only; all within-circuit excitatory--inhibitory settings
    remained fixed. A candidate had to retain finite, non-silent,
    non-saturated activity, a prominent 30--80 Hz rate peak in each circuit, and
    5--15% detuning. The deterministic score selected the valid candidate nearest
    10% detuning, with peak prominence and grid order breaking ties.
    Cross-coupling was not evaluated during calibration.

  + *Apply the registered locking gate.* Locking required both circuits to remain
    active, frequency difference no more than
    #n(100 * r.registration.locking_thresholds.max_frequency_difference_fraction_of_baseline)%
    of baseline, phase-locking-value (PLV) gain at least
    #r.registration.locking_thresholds.min_plv_gain, mean-band coherence gain at
    least #r.registration.locking_thresholds.min_coherence_gain, PLV at least
    #r.registration.locking_thresholds.min_half_window_plv in each half-window,
    and phase-offset drift no more than
    #r.registration.locking_thresholds.max_half_window_phase_offset_difference_rad
    rad. PLV ranges from zero for dispersed phase differences to one for a fixed
    phase relationship. A passing condition also needed an orthogonally adjacent
    locked condition. Silent conditions were never eligible.

  + *Refine the observed boundary.* After the registered sweep failed, an
    explicitly exploratory grid sampled six weights from 0.10 through 0.35 and
    six delays from 8 through 15 ms. Conditions were ranked, without altering
    the registered result, by normalized frequency difference, full-window PLV,
    minimum half-window PLV, and half-window phase drift. The best condition was
    selected mechanically before the comparison figure was drawn.

  + *Continue a mature phase-separated state.* Cross-circuit projections were
    present at weight zero during burn-in, making the zero-weight and coupled
    graphs structurally state-compatible. Independent Poisson inputs evolved both
    circuits to timestep #acquisition.checkpoint.step
    (#n(acquisition.checkpoint.time_ms, digits: 1) ms). The zero-coupling run was
    repeated exactly to that timestep and complete membrane, refractory,
    conductance, recurrent-delay, and delayed-input state was exported. The same
    saved state and future input realization then initialized one static reciprocal
    E→I coupling continuation. No uncoupled continuation was simulated. After the
    broader weight--delay refinement, a bounded 2 s search compared mature
    checkpoints and weights 0.20--0.30 on one fixed archived input stream at delay
    10 ms. The declared gate required early PLV at most 0.90, displaced early phase, delayed
    acquisition, late PLV at least 0.92, 95% of final-250-ms circular phase errors
    below 0.65 rad, and 95% of final-250-ms instantaneous frequency differences
    below 3 Hz; the passing case with the smallest late phase error was selected.
    Weights were not part of runtime state. A 250 ms
    pre-checkpoint recording context was used only to prevent band-pass/Hilbert
    edge bias; no spikes were shifted or replayed into the model.

  == Calibration produced active detuned gamma

  #block(breakable: false)[
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
  ]

  Candidate #selected.index used input weight #selected.settings.input_weight
  and independent per-channel drive rates #hz(selected.settings.rate_a_hz) and
  #hz(selected.settings.rate_b_hz). All cells participated. The first analysis
  implementation took the maximum coherence anywhere in the gamma band. The
  baseline saturated near one, making the registered gain impossible. Before
  any coupling condition ran, the coherence statistic was frozen as the arithmetic
  mean of magnitude-squared coherence bins from 30 through 80 Hz. The corrected
  uncoupled baseline is #n(baseline.metrics.synchrony.gamma_coherence).

  #block(breakable: false)[
  == Mature-state continuation reveals phase capture

  #figure(
    image(
      "/artifacts/data/exp078/acquisition/phase_acquisition.png",
      width: 88%,
      alt: "Separate spike rasters and population-rate panels for circuits A and B, followed by A-minus-B phase delay and instantaneous frequency difference over time.",
    ),
    caption: [One continuous coupled trajectory from an already mature checkpoint.
      Reciprocal E→I weight #acquired.strength and delay #acquired.delay_ms ms are
      enabled at 0 ms; nothing else is reset. The first two panels show
      fixed samples of 100 E and 50 I neurons from circuits A and B. The next
      two show their full-population E rates on a common scale without overlap.
      The final two panels show 150 ms circularly smoothed A-minus-B phase delay and
      the corresponding instantaneous frequency difference, additionally smoothed
      over 75 ms.
      Phase wanders over the early and middle window while frequency difference
      changes sign. After #n(acquisition-metrics.capture_time_ms, digits: 1) ms,
      phase remains bounded around a preferred lag and slow frequency difference
      remains centred near zero.],
  )
  ]

  #block(breakable: false)[
    #table(
      columns: (1.25fr, .75fr, .75fr, .75fr, .75fr, .8fr),
      [Condition], [Frequency difference], [PLV], [30--80 Hz coherence], [Phase (rad)], [Lag (ms)],
      [Uncoupled], [#hz(baseline.metrics.synchrony.frequency_difference_hz)], [#n(baseline.metrics.synchrony.plv)], [#n(baseline.metrics.synchrony.gamma_coherence)], [#n(baseline.metrics.synchrony.mean_phase_difference_rad)], [#n(baseline.metrics.synchrony.cross_correlation_lag_ms, digits: 1) ms],
      [0.2, short], [#hz(short.metrics.synchrony.frequency_difference_hz)], [#n(short.metrics.synchrony.plv)], [#n(short.metrics.synchrony.gamma_coherence)], [#n(short.metrics.synchrony.mean_phase_difference_rad)], [#n(short.metrics.synchrony.cross_correlation_lag_ms, digits: 1) ms],
      [0.2, half period], [#hz(low-half.metrics.synchrony.frequency_difference_hz)], [#n(low-half.metrics.synchrony.plv)], [#n(low-half.metrics.synchrony.gamma_coherence)], [#n(low-half.metrics.synchrony.mean_phase_difference_rad)], [#n(low-half.metrics.synchrony.cross_correlation_lag_ms, digits: 1) ms],
      [0.5, intermediate], [#hz(medium-intermediate.metrics.synchrony.frequency_difference_hz)], [#n(medium-intermediate.metrics.synchrony.plv)], [#n(medium-intermediate.metrics.synchrony.gamma_coherence)], [#n(medium-intermediate.metrics.synchrony.mean_phase_difference_rad)], [#n(medium-intermediate.metrics.synchrony.cross_correlation_lag_ms, digits: 1) ms],
      [0.25, 10 ms exploratory], [#hz(best.metrics.synchrony.frequency_difference_hz)], [#n(best.metrics.synchrony.plv)], [#n(best.metrics.synchrony.gamma_coherence)], [#n(best.metrics.synchrony.mean_phase_difference_rad)], [#n(best.metrics.synchrony.cross_correlation_lag_ms, digits: 1) ms],
    )
  ]

  The coarse registered sweep remains a negative result: none of its 15 coupled
  cells passes every joint gate. The subsequent 36-cell exploratory refinement
  samples weights 0.10--0.35 and delays 8--15 ms. Seven cells have zero
  frequency difference and PLV above 0.8, including adjacent cells at weights
  0.25 and 0.30 and delays 8--11.5 ms. The selected condition has equal
  #hz(best.metrics.populations.a.dominant_frequency_hz) spectral peaks, PLV
  #n(best.metrics.synchrony.plv), and only
  #n(best.metrics.synchrony.half_phase_offset_difference_rad) rad phase-offset
  change between analysis halves. Its mean broad-band coherence does not meet
  the original +0.10 gain gate. That gate averages the whole 30--80 Hz band and
  is poorly matched to a pair that collapses onto a narrow 30.5 Hz rhythm; it is
  retained as provenance, but does not negate the direct frequency and phase
  evidence for synchronization.

  Complete-state continuation changes the mechanistic interpretation. The
  checkpoint begins with two healthy gamma oscillators rather than two freshly
  reset populations, and it contains no imposed spike-stream offset. Under weight
  #acquired.strength coupling, median phase error across the four successive
  analysis quarters is
  #n(acquisition-metrics.quartile_phase_errors_rad.at(0)),
  #n(acquisition-metrics.quartile_phase_errors_rad.at(1)),
  #n(acquisition-metrics.quartile_phase_errors_rad.at(2)), and
  #n(acquisition-metrics.quartile_phase_errors_rad.at(3)) rad. The smoothed
  late-phase span is
  #n(acquisition-metrics.smooth_late_span_rad) rad around an eventual lag of
  #n(acquisition-metrics.fixed_phase_delay_rad) rad, while the final-window 95th
  percentile absolute frequency difference is
  #hz(acquisition-metrics.late_frequency_difference_95_hz). Coupling therefore changes
  the phase dynamics over repeated cycles rather than merely selecting an already
  locked initial condition. The longer window also prevents an overclaim:
  post-capture PLV is #n(acquisition-metrics.post_capture_plv), and 95% of
  post-capture phase errors lie below
  #n(acquisition-metrics.post_capture_phase_error_95_rad) rad. The relationship is
  bounded and frequency-synchronized on the displayed slow timescale, but it is
  not an absorbing fixed-phase state.

  == Strong coupling suppresses one circuit

  At strength #strong-short.variant.strength and short delay, circuit B has
  #hz(strong-short.metrics.populations.b.e_rate_hz) excitatory and
  #hz(strong-short.metrics.populations.b.i_rate_hz) inhibitory activity, while
  circuit A remains at #hz(strong-short.metrics.populations.a.e_rate_hz) and
  #hz(strong-short.metrics.populations.a.i_rate_hz). The condition is explicitly
  inactive and spectrally invalid. Other strong short- and intermediate-delay
  conditions show the same one-sided suppression boundary. Half-period delay
  avoids complete silence at the largest registered strengths, but those cells
  fail the full locking gates.

  == Conclusion and limitations

  The registered coarse-grid success criterion fails, but the explicitly
  exploratory refinement reveals the missed phase-locking region. Reciprocal
  E→I excitation can reshape relative phase over many cycles and produce a bounded
  phase relationship with near-zero slow frequency difference. The 2 s extension
  retains noisy phase corrections rather than a permanent fixed lag. Excessive coupling recruits
  enough inhibition to suppress circuit B rather than synchronize it. This is
  the intended reduced mechanism: long-range excitation can synchronize PING
  circuits by recruiting the other circuit's local inhibition.

  The evidence has one exploratory seed, one calibrated oscillator pair, dense
  connectivity, and one set of intrinsic PING parameters. It does not estimate
  robustness across seeds, circuit sizes, detuning ranges, input statistics, or
  biological preparations.
]
