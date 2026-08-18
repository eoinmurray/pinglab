#let meta = (
  title: "Can controlled phase offsets synchronize?",
  date: "2026-08-18",
  description: "Mature SNNLANG PING circuits start from prescribed relative phases to test whether reciprocal excitation drives convergence.",
  collection: "snnlang",
  status: "draft",
  order: 11,
)

#let r = json("/artifacts/data/exp085/numbers.json")

#let body = [
  == Abstract

  Two mature, slightly detuned PING circuits begin at four prescribed relative phases before reciprocal excitation switches on. The maximum onset-phase error is #calc.round(r.max_target_error_rad, digits: 4) rad. With K = #r.config.coupling, terminal phase error falls to a median #calc.round(r.coupled_terminal_error_median_rad, digits: 2) rad; matched uncoupled controls retain #calc.round(r.control_terminal_error_median_rad, digits: 2) rad error. Controlling onset phase turns an accidental representative trajectory into a direct test of convergence.

  == Methods

  + *Scout mature uncoupled dynamics.* Independent #r.config.input_rate_hz Hz/channel Poisson streams drive two #r.config.n_e_per_circuit E / #r.config.n_i_per_circuit I PING circuits. Circuit A uses #r.config.tau_a_ms ms inhibitory decay and circuit B uses #r.config.tau_b_ms ms. Each seed runs uncoupled for #r.config.scout_ms ms.

  + *Select valid phase-controlled states.* Smoothed E-population rates are filtered over #r.phase_analysis.band_hz.at(0)--#r.phase_analysis.band_hz.at(1) Hz. The runner locates naturally occurring joint states nearest the four prescribed phases. It then deterministically replays the uncoupled prefix and captures the complete graph runtime state at each onset. No voltage, conductance, refractory counter, population history, or delay history is edited or recombined.

  + *Apply paired futures.* Each captured state branches into K = #r.config.coupling coupling-on and K = 0 controls. Within a seed, every prescribed phase receives the same future private A/B input streams. Exact replay assertions compare every population spike against the scout prefix before any branch is accepted.

  + *Measure convergence.* Each trajectory is compared with its own mean relative phase over the final #r.phase_analysis.terminal_phase_window_ms ms. Circular terminal-phase error is smoothed over #r.phase_analysis.phase_error_smooth_ms ms. The final #r.phase_analysis.terminal_edge_trim_ms ms are excluded uniformly as a filter-edge guard.

  #figure(
    image("/artifacts/data/exp085/network.svg", width: 88%, alt: "Two independently driven PING circuits connected by reciprocal excitatory projections."),
    caption: [Controlled coupling-onset graph. Independent spike inputs drive circuits A and B. Four reciprocal AMPA projections switch from zero during scout and replay to K = #r.config.coupling after each captured onset.],
  )

  == Results

  The prescribed offsets are recovered to sub-milliradian accuracy across all #(r.config.phase_targets_rad.len() * r.config.trials) onsets.

  #figure(
    image("/artifacts/data/exp085/phase_control.svg", width: 72%, alt: "Prescribed relative phase against achieved relative phase at coupling onset."),
    caption: [Phase-control accuracy. Each point is one seed and prescribed onset phase; the diagonal is exact agreement. Maximum circular target error is #calc.round(r.max_target_error_rad, digits: 5) rad.],
  )

  #figure(
    image("/artifacts/data/exp085/synchrony_over_time.svg", width: 100%, alt: "Terminal phase error after onset for coupled circuits and matched uncoupled controls."),
    caption: [Convergence from controlled relative phases. Top: coupling switches to K = #r.config.coupling. Bottom: matched states continue uncoupled. Thin traces are all #r.config.trials seeds at each prescribed phase; thick traces are phase-group medians. Coupled trajectories collapse toward low terminal error, whereas uncoupled controls continue to traverse relative phase.],
  )

  Controlled onset resolves the ambiguity in the earlier accidental-phase experiment: convergence is a repeatable consequence of reciprocal coupling here, not a fortunate single seed. The terminal reference remains retrospective, so the figure demonstrates convergence without registering a synchronization-latency estimator.
]
