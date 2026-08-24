# Exp097 ScientificCollectionState

## Registration

- Writing: `writings/exp097.typ`
- Collection: `snnlang`
- Scientific role: exploratory recurrent-state visualization scout
- Lifecycle status: `ExpScout`; retrospectively reconstructed plan and completed
  `ScoutExecution`
- Implementation: `experiments/exp097.py`; tests: `experiments/tests/test_exp097.py`
- Hard dependencies: `experiments/exp083.py`, `tools/snn`, and `tools/snnlang`
- Scout execution: `r008`
- Writing metadata: title `Can a PING cycle be seen as a running engine?`,
  status `complete`, order 11
- Simulation results: `numbers.json`, `measured_cycle.svg`, `measured_engine.mp4`,
  `measured_engine_poster.png`, `animation_state.json`, `input_ramp_engine.mp4`,
  `input_ramp_engine_poster.png`, `input_ramp_engine_state.json`
- Simulation-result web animation: `ping_engine_state.js`, `ping_engine.js`, `ping_engine.css`

## PublicationView

- Current local view uses ad-hoc run `r008` and the two video results.
- No campaign evidence has been accepted or activated as a gold-star view.

## Execution

- Local execution used the frozen 800-E, 200-I network, network seed 83, input
  seeds 8300--8304, 50 Hz/channel drive, 0.1 ms timestep, 2 ms inhibitory
  decay, and five 500 ms trials with a 100 ms transient exclusion.
- Full recordings preserve E/I spikes, E/I voltages, E-to-I AMPA conductance,
  and I-to-E GABA conductance in the run scratch artifact.
- The animation state contains five simulated cycles from the trial selected by
  the frozen median-frequency rule. It is downsampled for display; analyses use
  native-resolution recordings.
- A second simulation uses the same network and input realization with a linear
  0--50--0 Hz/channel drive between two 200 ms silent periods.
## Scientific disposition

- Revise: the conductance plane is coherent but mean voltage improves held-out
  phase and next-volley prediction.
- The constant-drive rhythm is below gamma, and the scout is specific to one
  network realization and its tested operating points.

## Campaign readiness and blockers

- Not campaign-ready: no prospectively frozen plan predates this execution.
- A new scout must first locate a gamma-frequency operating point before a
  multi-realization `ExpStudyPlan` would be scientifically warranted.
- Campaign construction, evidence acceptance, activation, and publication each
  remain separate user review gates.
