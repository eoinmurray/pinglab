# exp164 — Untrained digit routing

Two constant-cue views of a saved fixed A/B → C spiking circuit. Rates, phase,
reconstructed conductances and raw 8×8 A/B/C activity maps are retained. There
is no classifier, trained decoder, sweep or probe stage in this experiment.

- `compute.py --source PHASE_ID` imports an explicit authenticated historical
  `exp163` digit-routing phase, preserving native start arrays, bundle, diagram,
  inputs, spikes, images and original configuration. It does not simulate.
  `--preserve-scratch DIRECTORY` optionally preserves migration history after
  authenticating recovered run copies against canonical Pingstore evidence.
- `recipe.py` owns observation settings and validates the fixed rate encoder.
- `analyse.py --source COMPUTE_ID` selects both pre-switch cue conditions,
  verifies unchanged sensory/background trains, and calculates plotted traces.
- `present.py --source ANALYSE_ID` renders the two figures and native diagram.
- `test.py` contains non-simulation checks. Every stage accepts `--run-id`.

```sh
uv run python experiments/exp164/compute.py --source exp163-r007-compute-phase-scratch-digit-routing
uv run python experiments/exp164/analyse.py --source COMPUTE_ID
uv run python experiments/exp164/present.py --source ANALYSE_ID
uv run pytest experiments/exp164/test.py -q
```

The source trajectories last 2,400 ms and switch at 1,200 ms. Figures display
only 0–1,100 ms, with phase/spatial measurements at 650–1,100 ms and conductance
close-ups at 900–1,050 ms. The cue stays fixed throughout each displayed excerpt.
Rates/phase use 1 ms SD temporal smoothing; maps are raw E-cell rates at fixed
pixel locations. Both figures share corresponding scales.

Native conductance recordings were not retained. Conductances are explicitly
labelled analytical reconstructions from saved spikes and native weights,
planner delays and decay, with zero initial conductance. Float64 population
means can differ from native float32 traces by rounding. Low inhibition is a
proxy rather than measured transmission gain. These views use one tuned image
pair and input realization, without a timing intervention or general decoding
claim. Original simulation provenance remains pinned through the import input.

Migration evidence: `exp164-r001-compute` imports the original trial and retains
`history.zip`, a byte-verified archive of the unique scratch source, notes,
configuration and launch history. Recovered run copies and transfer archives
were authenticated against their canonical Pingstore runs before the migrated
scratch directory was removed. Existing scientific runs were not rewritten.
`exp164-r002-analyse` and `exp164-r003-present` reproduce both original PNGs
byte-for-byte. Import, analysis and presentation took 2.094, 6.222 and 2.770
seconds respectively. The report rendered successfully with native-bound
parameter tables. No new simulations, backup or publication were performed.
