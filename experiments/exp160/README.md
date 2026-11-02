# exp160 — Spiking one-way entrainment

Generate Figure 1 (matched population rates, phase and conductance timing) and
Figure 2 (all-neuron spike rasters) for the accepted condition. Settings live in
`recipe.configuration`: 400 E / 100 I cells per circuit, private sensory fan-in
128, local E→I fan-in 80, target gain CVs 0.1 / 0.2, E→I / I→I weight multipliers
0.8 / 0.1, A/B inputs 26 / 24 Hz/channel, and 0.3 µS nominal A→B E strength.
The retained A→B I projection has zero gain. Coupling starts at 500 ms in a
1,500 ms trial. Control and coupled trials share weights and complete inputs.

## Stages

1. `uv run python experiments/exp160/compute.py --parameter-source exp159-r001-compute`
   runs the fixed pair on CPU. It expands the pinned native reference weights,
   normalizes incoming totals, and applies the recorded sparse masks and
   unit-mean lognormal target gains. It retains native bundles, execution-start
   snapshots, inputs, spikes and conductance recordings. It prints its compute ID.
2. `uv run python experiments/exp160/analyse.py --source COMPUTE_ID` validates
   the selected inputs and transformed execution weights, then measures rates,
   phase locking and conductance/output timing. It prints its analysis ID.
3. `uv run python experiments/exp160/present.py --source ANALYSE_ID` renders
   Figures 1–2 and the required appendix network diagram. Scientific figures
   use saved spikes and conductances; no smoothing or jitter is applied to rasters.
   All stages accept optional `--run-id`; no stage launches an upstream stage.

To regenerate the accepted figures without simulations, use
`present.py --source exp160-r077-analyse`, or run analysis on
`exp160-r068-compute` and pass the resulting analysis ID to presentation.

## Evidence

The accepted condition is `exp160-r068-compute`. Coupled PLV is 0.9849 and
phase drift 0.0434 cycles/s over 800–1,500 ms; control drift is 1.9219 cycles/s.
E participation averages 59.8% / 73.8% per complete population cycle; I remains
strongly cycle-locked. These are one exploratory realization, not seed validation
or evidence of biological realism. The earlier illustrative timing criterion
fails: 90% of input peaks fall in low inhibition, but only 75% precede raw B E
output by the required 0.2–10 ms. The threshold remains 80% for both fractions.

Historical exploratory runs and their selection provenance remain immutable in
Pingstore. Sweep controls, later unselected conditions, network-only redraw mode
and the third raster close-up are not part of the maintained generation code.

## Checks

`uv run pytest experiments/exp160/test.py -q -k 'not native_pair'` checks phase,
timing, imports, matched inputs and exact agreement with the accepted native
configuration and execution weights without simulations. The `native_pair` test
also simulates a short pair and requires execution authorization.
