# exp172 — Spatial Gaussian Selective Relay

A uniform rate cue selects between two spatial Gaussian patterns in an A/B → C
network with 400 E and 100 I neurons per circuit. The report shows seed 185 at
3 Hz/channel, with C recurrence scale 0.05 and inhibition scale 1.31. These
settings were selected on the displayed seed; this is a tuned demonstration,
not held-out validation or an isolated causal CTC test.

1. `uv run python experiments/exp172/compute.py` executes the fixed seven-cue
   protocol and two single-sender controls. It validates the pinned native
   parameter source and retains spikes, settings, bundles and start snapshots.
   Historical selection evidence is pinned separately; this stage does not
   repeat the search. Simulation requires explicit authorisation.
2. `uv run python experiments/exp172/analyse.py --source COMPUTE_ID` validates
   the saved trial grid and recomputes measurements from recorded spikes.
3. `uv run python experiments/exp172/present.py --source ANALYSE_ID` renders
   the report's four figures and native network diagram without simulation.
   The default cue is the recorded representative cue; `--cue` and `--seed`
   select another recorded trial.
4. `uv run pytest experiments/exp172/test.py -q` runs non-simulation checks.

Every stage accepts `--run-id`. Downstream sources must be explicit completed
Pingstore IDs; stages never launch one another. `recipe.py` owns configuration,
input generation and network authoring. No scratch or other experiment code is
required. Existing runs remain immutable.

The figures retain E/I spikes, spatial profiles, unwrapped
relative phase, 100 ms moving-window spatial amplitudes (5 ms step), and both
executed sender→C weight matrices. Overlapping fits blur brief excursions and
are not independent samples. The diagram uses the saved compiled architecture;
weight labels refer to native start-snapshot arrays, with zero pathways marked.
