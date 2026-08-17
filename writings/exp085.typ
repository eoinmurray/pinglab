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

#let body = [
  == Abstract

  Two slightly detuned PING circuits begin from the same mature uncoupled trajectory before reciprocal excitation switches on. Across #(r.config.couplings.len()) coupling strengths, the unwrapped relative phase changes from a persistent drift of #calc.round(uncoupled.phase_slope_median_rad_s, digits: 2) rad/s to #calc.round(strongest.phase_slope_median_rad_s, digits: 2) rad/s. The experiment uses runtime-state branching to isolate coupling onset from network activation and shows the transition directly without imposing a binary synchronization threshold.

  == Methods

  + *Equilibrate two detuned circuits.* Independent #r.config.input_rate_hz Hz/channel Poisson streams drive two #r.config.n_e_per_circuit E / #r.config.n_i_per_circuit I PING components. Circuit A uses #r.config.tau_a_ms ms inhibitory decay and circuit B uses #r.config.tau_b_ms ms. Both run uncoupled for #r.config.equilibration_ms ms.

  + *Branch one mature state.* The executor captures voltages, refractory counters, conductances, population histories, and input-delay histories at coupling onset. That identical state branches across #(r.config.couplings.len()) reciprocal coupling strengths. Private future inputs remain paired across K and independent between circuits.

  + *Measure relative phase.* E-population rates are smoothed with a #r.phase_analysis.smoothing_sigma_ms ms Gaussian kernel and filtered over #r.phase_analysis.band_hz.at(0)--#r.phase_analysis.band_hz.at(1) Hz. Hilbert phases define the unwrapped A-minus-B relative phase, expressed as its change from coupling onset. Figures divide this phase change by $2 pi$ to report relative cycles gained. The final #r.phase_analysis.terminal_edge_trim_ms ms are excluded to prevent the finite-record filter boundary from appearing as a phase jump. This continuous trajectory is the primary result; no locking threshold or latency is inferred.

  + *Measure supporting responses.* The registered #raw(r.frequency_analysis.name) policy reports each circuit's dominant rhythm from the final 1.5 s. Frequency difference, descriptive phase-locking value, and population firing rates support the phase trajectories. Figure 5 measures settling as each trial's circular phase error from its own mean terminal offset, defined over the final #r.phase_analysis.terminal_phase_window_ms ms and smoothed over #r.phase_analysis.phase_error_smooth_ms ms. It is a descriptive convergence view, not a registered latency estimator.

  #figure(
    image("/artifacts/data/exp085/network.svg", width: 88%, alt: "Two independently driven PING circuits connected by reciprocal excitatory projections."),
    caption: [Two-circuit coupling-onset graph. Independent spike inputs drive circuits A and B. Four reciprocal AMPA projections share coupling strength K; their weights are zero during equilibration and vary only after the common runtime-state branch. Local PING topology and every non-coupling parameter remain fixed.],
  )

  == Results

  The relative-phase trajectories expose the response to coupling without reducing it to a label. At K = #uncoupled.coupling the circuits repeatedly lap one another. Increasing K changes the slope and structure of that drift; the strongest condition ends with a median phase-locking value of #calc.round(strongest.phase_locking_value_median, digits: 2).

  #figure(
    image("/artifacts/data/exp085/relative_phase.svg", width: 100%, alt: "Small multiples of unwrapped relative phase after coupling onset at eleven reciprocal coupling strengths."),
    caption: [Cumulative A-minus-B relative cycles gained after coupling onset. One vertical unit means that A has completed one additional cycle relative to B. Panels increase in reciprocal coupling K from left to right and top to bottom. Every trace is zeroed at onset; the final #r.phase_analysis.terminal_edge_trim_ms ms are excluded as a registered filter-edge guard. The black trace is one predeclared paired trial; grey traces are the other #(r.config.trials - 1) trials from the same mature-state and input protocol. Persistent slope indicates repeated lapping, plateaus indicate transient capture, and a bounded trace indicates a stable phase relationship.],
  )

  #figure(
    image("/artifacts/data/exp085/response.svg", width: 100%, alt: "Dominant frequencies, their absolute difference, and phase-locking value across reciprocal coupling."),
    caption: [Supporting coupling response. First: median dominant E-population frequencies of circuits A and B in hertz. Second: their absolute difference in hertz. Third: median descriptive phase-locking value across #r.config.trials paired trials. Fourth: mean E and I firing rates in hertz for both circuits. K is the shared weight of the four reciprocal AMPA projections.],
  )

  #figure(
    image("/artifacts/data/exp085/representative_rates.svg", width: 100%, alt: "Excitatory population-rate traces for circuits A and B at four predeclared coupling strengths."),
    caption: [Excitatory population activity after coupling onset at the four predeclared K values. Time is in milliseconds and rate is in hertz after the registered #r.phase_analysis.smoothing_sigma_ms ms smoothing. Circuit A is black and circuit B is dashed red; all panels use the same paired trial and axes. The sequence makes repeated lapping, transient alignment, and shared volley timing directly comparable.],
  )

  #figure(
    image("/artifacts/data/exp085/synchrony_over_time.svg", width: 100%, alt: "Circular phase error from the terminal phase relationship over time at four coupling strengths."),
    caption: [Convergence toward the terminal phase relationship after coupling onset for K = 0.06--0.10. Each line is the median circular error from the trial's own mean phase offset during the final #r.phase_analysis.terminal_phase_window_ms ms; bands span the interquartile range across #r.config.trials paired trials, and trajectories use #r.phase_analysis.phase_error_smooth_ms ms smoothing. Lower values indicate closer approach to the terminal relationship. The terminal reference is descriptive and retrospective, so this figure does not estimate synchronization latency.],
  )
]
