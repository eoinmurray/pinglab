#let meta = (
  title: "What happens when coupling switches on?",
  date: "2026-08-17",
  description: "Two equilibrated SNNLANG PING circuits branch from an identical runtime state to reveal how reciprocal excitation changes their relative phase.",
  collection: "snnlang",
  status: "draft",
  order: 11,
)

#let r = json("/artifacts/data/exp085/numbers.json")
#let uncoupled = r.conditions.first()
#let strongest = r.conditions.last()
#let sync-example = r.synchrony_example
#let convergence-counts = r.conditions.map(row =>
  "K = " + str(row.coupling) + ": " + str(row.clean_convergence_count) + "/" + str(r.config.trials)
).join("; ")

#let body = [
  == Abstract

  Two slightly detuned PING circuits begin from the same mature uncoupled trajectory before reciprocal excitation switches on. Across #(r.config.couplings.len()) coupling strengths, the unwrapped relative phase changes from a persistent drift of #calc.round(uncoupled.phase_slope_median_rad_s, digits: 2) rad/s to #calc.round(strongest.phase_slope_median_rad_s, digits: 2) rad/s. The experiment uses runtime-state branching to isolate coupling onset from network activation and shows the transition directly without imposing a binary synchronization threshold.

  == Methods

  + *Equilibrate two detuned circuits.* Independent #r.config.input_rate_hz Hz/channel Poisson streams drive two #r.config.n_e_per_circuit E / #r.config.n_i_per_circuit I PING components. Circuit A uses #r.config.tau_a_ms ms inhibitory decay and circuit B uses #r.config.tau_b_ms ms. Both run uncoupled for #r.config.equilibration_ms ms.

  + *Branch one mature state.* The executor captures voltages, refractory counters, conductances, population histories, and input-delay histories at coupling onset. That identical state branches across #(r.config.couplings.len()) reciprocal coupling strengths. Private future inputs remain paired across K and independent between circuits.

  + *Measure relative phase.* E-population rates are smoothed with a #r.phase_analysis.smoothing_sigma_ms ms Gaussian kernel and filtered over #r.phase_analysis.band_hz.at(0)--#r.phase_analysis.band_hz.at(1) Hz. Hilbert phases define the unwrapped A-minus-B relative phase, expressed as its change from coupling onset. The final #r.phase_analysis.terminal_edge_trim_ms ms are excluded to prevent the finite-record filter boundary from appearing as a phase jump. No locking threshold or latency is inferred.

  + *Measure supporting responses.* The registered #raw(r.frequency_analysis.name) policy reports each circuit's dominant rhythm from the final 1.5 s. Frequency difference, descriptive phase-locking value, and population firing rates support the phase trajectories. Figure 3 measures every trial at one illustrative K by its circular phase error from its own mean terminal offset, defined over the final #r.phase_analysis.terminal_phase_window_ms ms and smoothed over #r.phase_analysis.phase_error_smooth_ms ms. The displayed K is the condition containing the stable trial with the largest clean onset-to-settled reduction; all samples at that K are then shown. This is descriptive, not a registered latency estimator.

  #figure(
    image("/artifacts/data/exp085/network.svg", width: 88%, alt: "Two independently driven PING circuits connected by reciprocal excitatory projections."),
    caption: [Two-circuit coupling-onset graph. Independent spike inputs drive circuits A and B. Four reciprocal AMPA projections share coupling strength K; their weights are zero during equilibration and vary only after the common runtime-state branch. Local PING topology and every non-coupling parameter remain fixed.],
  )

  == Results

  The coupling scan shows frequency convergence and rising phase concentration without reducing either to a binary label. At K = #uncoupled.coupling the circuits repeatedly lap one another; the strongest condition ends with a median phase-locking value of #calc.round(strongest.phase_locking_value_median, digits: 2).

  #figure(
    image("/artifacts/data/exp085/response.svg", width: 100%, alt: "Dominant frequencies, their absolute difference, and phase-locking value across reciprocal coupling."),
    caption: [Supporting coupling response. First: median dominant E-population frequencies of circuits A and B in hertz. Second: their absolute difference in hertz. Third: median descriptive phase-locking value across #r.config.trials paired trials. Fourth: mean E and I firing rates in hertz for both circuits. K is the shared weight of the four reciprocal AMPA projections.],
  )

  #figure(
    image("/artifacts/data/exp085/synchrony_over_time.svg", width: 100%, alt: "Five network pairs at one coupling strength converging toward their terminal phase relationships over time."),
    caption: [All #r.config.trials paired trials at the selected K = #sync-example.coupling. Each trace is that trial's circular error from its own mean phase offset during the final #r.phase_analysis.terminal_phase_window_ms ms, with #r.phase_analysis.phase_error_smooth_ms ms smoothing; lower values indicate closer approach to the terminal relationship. The rule-selected seed #sync-example.seed is black and the other seeds are grey. Retrospective terminal references make these illustrations of convergence, not synchronization-latency estimates.],
  )

  Under the same illustrative clean-convergence rule, the passing counts are #convergence-counts. A trial passes only when its terminal-error 95th percentile is at most #r.phase_analysis.clean_convergence_terminal_p95_max_rad rad and it has no post-settling excursion above #r.phase_analysis.clean_convergence_post_settling_max_rad rad. This post-hoc descriptive rule is not a formal synchronization boundary.
]
