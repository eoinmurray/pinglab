#import "/.demolab/lib.typ": cite, reference-list

#let meta = (
  title: "Arnold tongue of two coupled PING circuits",
  date: "2026-08-07",
  description: "An 80 E / 20 I graph-native reproduction recovers the widening Arnold tongue; a focused 800 E / 200 I confirmation resolves its lone phase-sign failure.",
  collection: "exploratory",
  status: "final",
)

#let r = json("/artifacts/data/exp078/numbers.json")
#let cal = json("/artifacts/data/exp078/calibration.json")
#let fs = json("/artifacts/data/exp078/finite_size_followup.json")
#let verdict = if r.conclusion.passed { "passed" } else { "failed" }
#let pct(x) = calc.round(x * 100, digits: 1)

#let body = [
  == Abstract

  Reciprocal excitation should synchronize two PING circuits over a wider
  natural-frequency mismatch as coupling increases, while preserving the rule
  that the intrinsically faster circuit leads in phase#cite(1). We test that
  claim in a graph-native sweep over measured detuning and coupling. The
  80 E / 20 I network recovers a centred Arnold tongue that widens across
  #r.conclusion.longest_successive_increase_run successive coupling steps, and
  all #r.benchmark.completed_trials primary trials are valid. One
  near-resolution cell violates the strict phase-sign criterion, so the 80 E /
  20 I reproduction #verdict despite #pct(r.conclusion.phase_lead_fraction)%
  sign agreement. A focused 800 E / 200 I confirmation resolves that exception:
  all #fs.conclusion.coupled_trials coupled trials lock with the expected phase
  sign.

  == Methods

  #enum(
    [*Fix the graph and coupling intervention.* Each circuit contains
    #r.config.n_e_per_circuit conductance-based excitatory neurons and
    #r.config.n_i_per_circuit inhibitory neurons. A private
    #(r.config.n_input_per_circuit)-channel Poisson population drives each
    excitatory population. Local E-to-I AMPA and I-to-E GABA-A projections
    generate PING activity. Four reciprocal AMPA projections share coupling
    weight $K$: $E_A$ projects to $E_B$ and $I_B$, and $E_B$ projects to $E_A$
    and $I_A$. The graph and network seed remain fixed. Figure 1
    shows the intervention; the evidence-scale result verifies the realized
    parameter tensors. <method-network>],

    [*Construct and measure the natural-detuning axis before coupling.* First,
    both circuits receive the same Poisson input rate with $K=0$. Five trials at
    each rate measure how uncoupled gamma frequency changes with drive. The
    longest interval that is fully valid, monotonic, and has a within-rate
    frequency interquartile range (IQR) no greater than 0.8 Hz is retained.

    The retained curve maps an input rate to its median uncoupled gamma
    frequency. Second, this mapping is used in reverse to construct the drive
    for each of 13 requested signed frequency differences $d$. The midpoint of
    the retained output-frequency range is $f_c=40.8$ Hz. For each $d$, circuit
    A is assigned the desired frequency $f_A^*=f_c+d/2$, while circuit B is
    assigned $f_B^*=f_c-d/2$. Linear interpolation on the measured curve then
    gives the input rates $r_A$ and $r_B$ expected to produce those two
    frequencies. For example, $d=+4$ Hz requests $f_A^*=42.8$ Hz and
    $f_B^*=38.8$ Hz, which map to $r_A=106.0$ and $r_B=86.36$ Hz per channel.
    Those two rates form one A/B input-rate pair.

    Third, each pair is run with $K=0$ using five matched seeds. For each seed,
    the measured natural detuning is

    $ Delta f_0 = f_A^0 - f_B^0, quad "(1)" $ <eq-detuning>

    where $f_A^0$ and $f_B^0$ are the gamma peak frequencies measured under the
    unequal drives while the circuits remain uncoupled. This seed-specific
    measured value, not $r_A-r_B$ and not the requested target, is the natural-
    detuning coordinate assigned to the corresponding coupled trial. Figure 2
    shows both the equal-drive calibration and the resulting unequal-drive
    detuning measurements. <method-calibration>],

    [*Measure phase and concentration.* Excitatory spikes are converted to
    population rates with a fixed 5 ms Gaussian kernel. Gamma frequency comes
    from the post-transient spectrum. A zero-phase 25--90 Hz band-pass filter
    and analytic signal define relative phase

    $ phi(t) = "unwrap"(theta_A(t) - theta_B(t)), quad "(2)" $ <eq-phase>

    where $phi(t)$ is unwrapped relative phase at time $t$, and $theta_A(t)$
    and $theta_B(t)$ are instantaneous phases. Phase concentration is

    $ R = abs(1 / T sum_(t=1)^T exp(i phi(t))), quad "(3)" $ <eq-plv>

    where $R$ is phase-locking value (PLV), $T$ is the number of analysed time
    samples, and $i$ is the imaginary unit. Figure 3 shows their time-domain
    behaviour; Figure 4 maps them across the primary grid.
    <method-measurement>],

    [*Freeze the locking rule.* Repeated uncoupled equal-drive trials set the
    admissible emergent frequency difference, relative-phase slope, and
    phase-slip count. A valid trial is locked when

    $ L = bold(1)[abs(f_A - f_B) <= epsilon_f]
        bold(1)[abs(a_phi) <= epsilon_phi]
        bold(1)[N_"slip" <= epsilon_"slip"], quad "(4)" $ <eq-locking>

    where $L$ is the binary locking label, $bold(1)[dot]$ is an indicator,
    $f_A$ and $f_B$ are coupled gamma frequencies, $epsilon_f$ is the frozen
    frequency-difference tolerance, $a_phi$ is the fitted slope of
    Equation 2, $epsilon_phi$ is its tolerance,
    $N_"slip"$ is the complete phase-slip count, and $epsilon_"slip"$ is its
    tolerance. PLV from Equation 3 is descriptive, not part
    of $L$. Figure 4 reports the component estimators, and Figure 5 applies
    Equation 4 without post hoc threshold changes.
    <method-locking>],

    [*Execute the frozen primary grid.* Thirteen target detunings cross eleven
    coupling levels with #r.config.trials_per_cell paired input seeds per cell.
    Each trajectory lasts #r.config.t_ms ms at a #r.config.dt_ms ms timestep;
    the first #r.config.burn_ms ms are discarded. A trial is invalid if recorded
    state is non-finite, any E or I firing rate is below 1 Hz, or either E
    population lacks a resolved 25--80 Hz peak. Figures 3--5 report this frozen
    grid. <method-grid>],

    [*Apply the registered verdict.* Geometry passes only if the locked region
    is contiguous, centred near zero detuning, and widens across at least three
    successive nonzero coupling levels. Every grid cell must contain a valid
    trial. In every locked nonzero-detuning cell, the circuit with greater
    natural frequency in Equation 1 must lead in phase under Equation 2. The verdict
    subsection applies these clauses
    literally. <method-verdict>],

    [*Run a focused finite-size confirmation.* Because only one phase-sign cell
    fails at 80 E / 20 I, the follow-up increases each population tenfold and
    tests mirrored target detunings at the uncoupled, disputed, and stronger
    coupling levels. It retains E-population spikes and summary observables,
    uses ten paired seeds per cell, and lengthens the post-transient window to
    5 s. Figure 6 reports the result without reopening the primary
    grid. <method-followup>],
  )

  The evidence archive bit-packs exact input and population spikes for every
  primary and follow-up cell. The primary archive also retains population-mean
  voltage and projection conductance at 1 ms resolution, compiled bundles,
  input hashes, seed ledger, and one realized parameter set per coupling level.

  == Results

  === The authored graph isolates reciprocal coupling

  Method 1 varies only the four cross-circuit AMPA projections.
  The local PING loops, private drives, graph topology, and seed are fixed, so a
  change across $K$ is attributable to the registered coupling intervention.

  #figure(
    image(
      "/artifacts/data/exp078/network.svg",
      width: 100%,
      alt: "Two independently driven PING circuits with four reciprocal cross-circuit excitatory projections.",
    ),
    caption: [Two-circuit coupling intervention. Each private Poisson input
    drives a local E-to-I-to-E PING loop. Dashed pathways are the four reciprocal
    AMPA projections varied together by coupling weight $K$; red bar-headed
    pathways are local GABA-A inhibition. Only the dashed pathways vary across
    the primary grid.],
  ) <fig-network>

  The realized parameter archive confirms that all non-coupling tensors remain
  invariant across #r.registration.realized_parameter_tensor_sets coupling
  levels, as required by Method 1.

  #pagebreak()

  === Calibration defines a stable detuning axis

  Method 2 yields a stable monotonic interval from
  #r.registration.calibrated_rate_interval_hz.at(0) to
  #r.registration.calibrated_rate_interval_hz.at(1) Hz per input channel. This
  interval supplies 13 paired A/B input-rate conditions. Measuring those
  conditions without coupling produces the natural-detuning coordinate used in
  the primary map.

  #figure(
    image(
      "/artifacts/data/exp078/calibration.png",
      width: 100%,
      alt: "Two-panel calibration showing combined uncoupled gamma frequency versus equal A/B input rate and measured natural detuning versus A-minus-B input-rate difference.",
    ),
    caption: [Construction of the natural-detuning axis before coupling. (A)
    Both circuits receive the same input rate with $K=0$. Each point is the
    median of 10 gamma-frequency measurements: circuits A and B in each of five
    trials. Bars show half their interquartile range. The shaded
    #r.registration.calibrated_rate_interval_hz.at(0)--#r.registration.calibrated_rate_interval_hz.at(1)
    Hz interval is retained because its frequency response is valid, stable,
    and monotonic. The 60 Hz condition is rejected for excessive within-rate
    spread; the 140 Hz condition is rejected because the frequency estimator
    switches between two spectral modes across seeds. (B) For each requested
    frequency difference, two desired frequencies are placed symmetrically
    around the 40.8 Hz centre of the retained range. Interpolation on Panel A
    converts each desired frequency into a drive rate, producing 13 A/B
    input-rate pairs. These pairs are then run with $K=0$. The horizontal axis
    shows the applied input-rate difference
    $r_A-r_B$. The vertical axis shows the median measured natural detuning
    $Delta f_0=f_A^0-f_B^0$ from Equation 1; bars span the interquartile range
    across five matched seeds. These measured values, rather than the input-rate
    differences or requested targets, become the detuning coordinates in
    Figures 4 and 5. Dashed lines mark zero.],
  ) <fig-calibration>

  The frozen thresholds used in Equation 4 are
  #calc.round(r.registration.locking_tolerances.frequency_difference_hz, digits: 2)
  Hz for frequency difference,
  #calc.round(r.registration.locking_tolerances.absolute_phase_slope_rad_s, digits: 2)
  rad/s for absolute phase slope, and
  #r.registration.locking_tolerances.phase_slips complete slips. The grid is
  identified by checksum #(r.registration.primary_grid_sha256).

  === Representative trajectories distinguish locking from drift

  Conditions selected before inspection show the transition in time: relative
  phase remains bounded after locking and drifts at the immediately preceding
  coupling.

  #figure(
    image(
      "/artifacts/data/exp078/representative_traces.png",
      width: 100%,
      alt: "Representative rasters, population rates, relative phase, and instantaneous frequency difference at zero detuning and either side of the plus-four-hertz locking boundary.",
    ),
    caption: [Predetermined time-domain checks. Rows show zero detuning at
    maximum coupling, the first locked positive-detuning condition, and the
    same detuning at the immediately preceding coupling. Columns show E rasters,
    5 ms-smoothed E rates in hertz, unwrapped relative phase $phi(t)$ from
    Equation 2 in radians, and instantaneous frequency difference in hertz.
    Relative phase is bounded in the locked conditions and drifts immediately
    outside the boundary.],
  ) <fig-traces>

  === Component estimators localize the locking transition

  All #r.benchmark.completed_trials primary trials are valid. Across the grid,
  low coupled-frequency difference, low relative-phase drift, few phase slips,
  and high phase concentration emerge together as coupling increases.

  #figure(
    image(
      "/artifacts/data/exp078/supporting_maps.png",
      width: 100%,
      alt: "Six supporting heatmaps for frequency difference, phase slope, slips, phase-locking value, circular phase, and validity.",
    ),
    caption: [Component estimators on the primary grid. All panels share
    measured natural detuning in hertz horizontally and coupling $K$ vertically.
    Panels report coupled frequency difference in hertz, fitted slope of
    Equation 2 in radians per second, complete phase slips, PLV $R$ from Equation 3,
    circular mean phase in radians,
    and valid-trial fraction. Validity
    is #pct(r.valid_trial_fraction)% across the grid, while low frequency
    difference, low phase drift, few slips, and high PLV coincide in the region
    subsequently classified as locked in Figure 5.],
  ) <fig-supporting>

  === Coupling produces a widening Arnold tongue

  Applying Equation 4 to the component estimators produces a contiguous locked
  region centred near zero detuning. Its width reaches
  #calc.round(r.conclusion.centred_locked_widths_hz.last(), digits: 2) Hz at the
  largest coupling and increases across
  #r.conclusion.longest_successive_increase_run successive coupling steps.

  #figure(
    image(
      "/artifacts/data/exp078/locking_map.png",
      width: 100%,
      alt: "Heatmap of locked-trial fraction over measured natural-frequency detuning and coupling.",
    ),
    caption: [Primary Arnold tongue. Measured natural detuning $Delta f_0$ from
    Equation 1 is on the horizontal axis in hertz; reciprocal coupling
    $K$ is on the vertical axis; colour is the fraction of valid trials that
    satisfy Equation 4. The centred locked region widens with coupling,
    satisfying the registered geometric clause.],
  ) <fig-tongue>

  === The strict 80 E / 20 I verdict fails one phase-sign cell <result-verdict>

  Method 6 gives a split result. Geometry and validity pass, but
  one qualifying cell has the wrong phase sign. Agreement is
  #pct(r.conclusion.phase_lead_fraction)%, while the registered clause requires
  every qualifying cell to agree. The overall 80 E / 20 I reproduction therefore
  #verdict. The exception lies near the estimator's registered resolution, so
  Method 7 tests that boundary rather than repeating the full
  grid.

  === The 800 E / 200 I confirmation resolves the exception

  The focused confirmation passes: all #fs.conclusion.valid_coupled_trials
  coupled trials are valid and locked under Equation 4, and all
  #fs.conclusion.phase_sign_correct_trials have the phase sign required by
  Equation 1 and Method 6. At the disputed negative-detuning
  cell, all #fs.conclusion.disputed_negative_cell.phase_sign_correct_trials
  seeds have the correct sign with mean phase
  #calc.round(fs.conclusion.disputed_negative_cell.mean_phase_rad, digits: 3)
  rad. The mirrored positive condition gives
  #calc.round(fs.conclusion.mirrored_positive_cell.mean_phase_rad, digits: 3)
  rad.

  #figure(
    image(
      "/artifacts/data/exp078/finite_size_followup.png",
      width: 100%,
      alt: "Focused 800 E / 200 I confirmation showing locking, phase, phase-locking value, mirrored phase distributions, and synchronized population-rate traces.",
    ),
    caption: [Focused 800 E / 200 I finite-size confirmation. Top panels show
    locked fraction, circular mean relative phase in radians, and PLV $R$ from
    Equation 3 over coupling $K$ for mirrored natural detunings; error bars are
    one standard deviation across #fs.config.trials_per_cell paired seeds. The
    lower-left panel isolates phase at the disputed coupling. The remaining
    panels show post-burn E-population rates in hertz (circuit A black, circuit B
    red). Both detuning signs lock with mirrored phase ordering, resolving the
    lone 80 E / 20 I exception.],
  ) <fig-followup>

  The follow-up supports finite-size or estimator-resolution noise, rather than
  a systematic phase-ordering contradiction, as the cause of the original
  exception.

  === Evidence scale

  Dense primary retention would occupy
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
