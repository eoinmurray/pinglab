# Exp097 ScientificCollectionState

## Registration

- Writing: `writings/exp097.typ`
- Collection: `snnlang`
- Status: `ExpScout`; execution complete
- Scout execution: `r006`
- Simulation results: `numbers.json`, `measured_cycle.svg`, `measured_engine.mp4`,
  `measured_engine_poster.png`, `animation_state.json`
- Simulation-result web animation: `ping_engine_state.js`, `ping_engine.js`, `ping_engine.css`

## Execution

- Local execution used the frozen 80-E, 20-I network, network seed 83, input
  seeds 8300--8304, 100 Hz/channel drive, 0.1 ms timestep, 2 ms inhibitory
  decay, and five 500 ms trials with a 100 ms transient exclusion.
- Full recordings preserve E/I spikes, E/I voltages, E-to-I AMPA conductance,
  and I-to-E GABA conductance in the run scratch artifact.
- The animation state contains five simulated cycles from the trial selected by
  the frozen median-frequency rule. It is downsampled for display; analyses use
  native-resolution recordings.

## Scientific disposition

- Revise: the conductance plane is coherent but mean voltage improves held-out
  phase and next-volley prediction.
- The scout is specific to one network realization and one operating point.
- Promotion to `ExpStudyPlan` requires a new prospective plan and user review.
