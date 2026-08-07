#import "/.demolab/lib.typ": cite, reference-list

#let meta = (
  title: "Arnold tongue of two coupled PING circuits",
  date: "2026-08-07",
  description: "An 80 E / 20 I graph-native reproduction recovers the widening Arnold tongue; a focused 800 E / 200 I confirmation resolves its lone phase-sign failure.",
  collection: "miscellaneous",
  status: "final",
)

#let r = json("/artifacts/data/exp078/numbers.json")
#let fs = json("/artifacts/data/exp078/finite_size_followup.json")
#let verdict = if r.conclusion.passed { "passed" } else { "failed" }

#let body = [
  == Abstract

  Two independently driven PING circuits were crossed over measured natural-
  frequency detuning and reciprocal coupling to reproduce the synchronization
  result of Lowet et al.#cite(1). The locking map forms a contiguous region
  around zero detuning and widens across
  #r.conclusion.longest_successive_increase_run successive coupling steps. All
  #r.benchmark.completed_trials primary trials are valid. The faster circuit
  leads in #calc.round(r.conclusion.phase_lead_fraction * 100, digits: 1)% of
  qualifying locked cells, but the registered rule requires the correct sign
  in every such cell. The reproduction therefore #verdict: its Arnold-tongue
  geometry succeeds, while one near-resolution phase cell prevents an overall
  pass at 80 E / 20 I. A preregistered finite-size confirmation at 800 E / 200 I
  subsequently passes all #fs.conclusion.coupled_trials coupled trials.

  == Methods

  #enum(
    [*Author the fixed graph.* Each circuit contains
    #r.config.n_e_per_circuit conductance-based excitatory neurons and
    #r.config.n_i_per_circuit inhibitory neurons. Independent
    #(r.config.n_input_per_circuit)-channel Poisson populations drive the two
    excitatory populations. Local E-to-I AMPA and I-to-E GABA-A projections
    generate PING activity. Four reciprocal AMPA projections couple
    $E_A$ to $E_B$ and $I_B$, and $E_B$ to $E_A$ and $I_A$. Their common weight
    is $K$. The graph and network seed remain fixed across the sweep.],

    [*Calibrate uncoupled frequency.* Equal-rate, zero-coupling trials determine
    a monotonic operating interval from
    #r.registration.calibrated_rate_interval_hz.at(0) to
    #r.registration.calibrated_rate_interval_hz.at(1) Hz per input channel.
    Input-rate pairs target thirteen signed detunings. For each seed, the
    measured zero-coupling frequencies define

    $ Delta f_0 = f_A^0 - f_B^0. $

    Here $f_A^0$ and $f_B^0$ are the uncoupled gamma peak frequencies of
    circuits A and B. $Delta f_0$, not input-rate difference or coupled
    frequency difference, is the horizontal coordinate.],

    [*Register locking before the primary sweep.* Repeated uncoupled equal-drive
    trials fix the admissible emergent frequency difference, absolute relative-
    phase slope, and phase-slip count. A valid trial is locked only when all
    three estimators fall within their frozen tolerances. Phase-locking value is
    descriptive and does not determine the label. A bounded pilot at the two
    extreme detunings and zero detuning selects the coupling ceiling
    $K=#r.registration.pilot_maximum_coupling$ before the primary map is viewed.],

    [*Execute the frozen grid.* The primary design crosses thirteen target
    detunings with eleven coupling levels and #r.config.trials_per_cell
    independently generated input pairs per cell. Each trajectory lasts
    #r.config.t_ms ms at a #r.config.dt_ms ms timestep; the first
    #r.config.burn_ms ms are discarded. A trial is invalid if any recorded state
    is non-finite, any E or I firing rate is below 1 Hz, or either E population
    lacks a resolved 25--80 Hz spectral peak.],

    [*Measure frequency and phase.* Excitatory spikes are converted to population
    rates with a fixed 5 ms Gaussian kernel. Gamma frequency comes from the
    post-transient spectrum. A zero-phase 25--90 Hz band-pass filter followed by
    the analytic signal gives relative phase

    $ phi(t) = "unwrap"(theta_A(t) - theta_B(t)). $

    $theta_A$ and $theta_B$ are the instantaneous phases of circuits A and B,
    and $phi$ is their unwrapped difference. Each trial reports both frequencies,
    absolute frequency difference, the linear slope of $phi$, complete phase
    slips, phase-locking value, and circular mean phase.],

    [*Retain reproducible evidence without dense-state bloat.* Exact input and
    population-spike tensors are bit-packed for every primary cell. Population-
    mean voltage and projection conductance are retained at 1 ms resolution,
    together with the compiled bundles, frozen grid, input hashes, and seed
    ledger. The realized runtime parameter tensors are retained once for each of
    the #r.registration.realized_parameter_tensor_sets coupling levels, with
    hashes verifying that all non-coupling tensors remain identical across
    levels. The simulator records dense neuron state for analysis, but the
    archive keeps the compact sufficient evidence rather than tens of gigabytes
    of redundant dense zeros.],

    [*Apply the registered verdict.* The geometric criterion requires a
    contiguous locked region centred near zero whose width increases across at
    least three successive nonzero coupling levels. Every grid cell must contain
    a valid trial. In every locked nonzero-detuning cell, the circuit with the
    greater $f^0$ must lead in phase. The final criterion is deliberately strict:
    a high fraction is not substituted for the word “every”.],
  )

  #figure(
    image(
      "/artifacts/data/exp078/network.svg",
      width: 100%,
      alt: "Two independently driven PING circuits with four reciprocal cross-circuit excitatory projections.",
    ),
    caption: [Authored two-circuit network. Each private Poisson input drives a
    local E-to-I-to-E PING loop. Dashed arrows are the four reciprocal AMPA
    pathways varied together by $K$; red bar-headed arrows are local GABA-A
    inhibition. The same compiled graph topology is used at every condition.],
  )

  == Calibration and registration

  #figure(
    image(
      "/artifacts/data/exp078/calibration.png",
      width: 100%,
      alt: "Gamma peak frequency versus Poisson input rate with the frozen monotonic interval shaded.",
    ),
    caption: [Uncoupled input-rate calibration. Input rate in hertz per channel is
    on the horizontal axis and median gamma peak frequency in hertz is on the
    vertical axis; bars show half the within-rate interquartile range. The shaded
    #r.registration.calibrated_rate_interval_hz.at(0)--#r.registration.calibrated_rate_interval_hz.at(1)
    Hz interval is frozen before coupled trials are inspected. The 60 Hz point
    is monotonic but exceeds the registered 0.8 Hz within-rate IQR ceiling; the
    140 Hz point is excluded because the frequency estimator hops between
    approximately 25 and 48 Hz spectral modes across seeds, producing the large
    error bar.],
  )

  The zero-coupling calibration registers a frequency-difference tolerance of
  #calc.round(r.registration.locking_tolerances.frequency_difference_hz, digits: 2)
  Hz, an absolute phase-slope tolerance of
  #calc.round(r.registration.locking_tolerances.absolute_phase_slope_rad_s, digits: 2)
  rad/s, and at most #r.registration.locking_tolerances.phase_slips complete
  slips. The primary-grid checksum is
  #(r.registration.primary_grid_sha256).

  == Results

  #figure(
    image(
      "/artifacts/data/exp078/locking_map.png",
      width: 100%,
      alt: "Heatmap of locked-trial fraction over measured natural-frequency detuning and coupling.",
    ),
    caption: [Primary Arnold-tongue map. Measured uncoupled detuning
    $Delta f_0=f_A^0-f_B^0$ in hertz is on the horizontal axis and reciprocal
    coupling $K$ is on the vertical axis. Colour gives the fraction of valid
    trials satisfying the registered frequency, phase-drift, and phase-slip
    tolerances. The centred locked width reaches
    #calc.round(r.conclusion.centred_locked_widths_hz.last(), digits: 2) Hz at
    the largest registered coupling and widens across
    #r.conclusion.longest_successive_increase_run successive steps.],
  )

  #figure(
    image(
      "/artifacts/data/exp078/supporting_maps.png",
      width: 100%,
      alt: "Six supporting heatmaps for frequency difference, phase slope, slips, phase-locking value, circular phase, and validity.",
    ),
    caption: [Supporting maps on the same measured-detuning by coupling grid.
    Panels show emergent absolute frequency difference in hertz, relative-phase
    slope in radians per second, complete phase slips, phase-locking value,
    circular mean phase in radians, and valid-trial fraction. Validity is
    #calc.round(r.valid_trial_fraction * 100, digits: 1)% across the primary
    trials, so the tongue is not produced by silent or numerically invalid
    conditions.],
  )

  #figure(
    image(
      "/artifacts/data/exp078/representative_traces.png",
      width: 100%,
      alt: "Representative rasters, population rates, relative phase, and instantaneous frequency difference at zero detuning and either side of the plus-four-hertz locking boundary.",
    ),
    caption: [Predetermined representative traces. Rows show the target-zero
    condition at maximum coupling, the first locked +4 Hz target condition, and
    the same detuning at the immediately preceding coupling. Columns show both E
    rasters, 5 ms-smoothed E rates, unwrapped relative phase, and instantaneous
    frequency difference. The paired locations illustrate stable phase at the
    registered inside cell and drift immediately outside it.],
  )

  == Verdict and scale decision

  The Arnold-tongue geometry passes: the locked region is centred near zero and
  its width increases for #r.conclusion.longest_successive_increase_run
  successive coupling transitions. Trial validity also passes. The phase-lead
  clause fails in one qualifying cell, leaving a lead-sign agreement of
  #calc.round(r.conclusion.phase_lead_fraction * 100, digits: 1)%. Because the
  protocol requires agreement in every cell, the overall 80 E / 20 I
  reproduction #verdict.

  The narrow failure occurs where measured detuning is close to the estimator's
  registered resolution, while the remaining qualifying cells show the expected
  sign. The reduced 800 E / 200 I confirmation therefore tests only target
  detunings $-1$ and $+1$ Hz at $K=0$, $0.016$, and $0.024$, with ten seeds and
  a 5 s post-transient window. All #fs.conclusion.valid_coupled_trials coupled
  trials are valid and locked, and all #fs.conclusion.phase_sign_correct_trials
  have the expected phase sign. At the previously disputed negative-detuning
  cell, all #fs.conclusion.disputed_negative_cell.phase_sign_correct_trials
  seeds have the correct sign with mean phase
  #calc.round(fs.conclusion.disputed_negative_cell.mean_phase_rad, digits: 3) rad;
  the mirrored positive condition gives
  #calc.round(fs.conclusion.mirrored_positive_cell.mean_phase_rad, digits: 3) rad.
  The focused confirmation therefore passes and supports finite-size/resolution
  noise, rather than a systematic contradiction, as the cause of the lone
  80 E / 20 I failure.

  #figure(
    image(
      "/artifacts/data/exp078/finite_size_followup.png",
      width: 100%,
      alt: "Focused 800 E / 200 I confirmation showing locking, phase, phase-locking value, mirrored phase distributions, and synchronized population-rate traces.",
    ),
    caption: [Focused 800 E / 200 I finite-size confirmation. Top panels show
    locked fraction, circular mean relative phase, and phase-locking value over
    coupling for mirrored natural-frequency detunings; error bars are one
    standard deviation across ten paired seeds. The lower-left panel isolates
    the phase distributions at the formerly disputed $K=0.016$ condition. The
    remaining panels show one second of representative post-burn excitatory
    population rates for the negative and positive detuning conditions (circuit
    A black, circuit B red). Both signs lock tightly and exhibit the expected
    mirrored phase ordering.],
  )

  Dense retention would occupy
  #calc.round(r.benchmark.projected_dense_recording_bytes / 1e9, digits: 2) GB.
  The published exact-event and state-summary archive occupies
  #calc.round(r.benchmark.published_compact_archive_bytes / 1e6, digits: 1) MB,
  a #calc.round(r.benchmark.compression_ratio, digits: 1)-fold reduction. The
  #r.benchmark.completed_cells primary cells required a median
  #calc.round(r.benchmark.median_cell_runtime_s, digits: 2) s each locally.

  #reference-list((
    (
      text: [Lowet, Roberts, Hadjipapas, Peter, van der Eerden & De Weerd (2015): _Input-Dependent Frequency Modulation of Cortical Gamma Oscillations Shapes Spatial Synchronization and Enables Phase Coding_. PLOS Computational Biology.],
      doi: "10.1371/journal.pcbi.1004072",
    ),
  ))
]
