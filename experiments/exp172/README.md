# exp172 — Spatial Gaussian Selective Relay

Test whether a uniform rate cue selects which of two static spatial Gaussian
patterns C represents. The circuit has 400 E / 100 I
per circuit, identical narrow Gaussian sender→C weights, weak local C E→E recurrence,
cue A then B, one input seed 185 and seven cue strengths. The current feedback setting uses C E→E scale 0.05, Gaussian SD ten E neurons,
and C I→E scale 1.31. The cue and feedback setting were selected on the displayed
seed after inspecting their sweeps; this is a tuned demonstration, not held-out
validation or evidence isolating a causal CTC mechanism.

1. `uv run python experiments/exp172/compute.py` executes the configured nine CPU
   trials only. It validates the explicitly pinned native parameter input and
   saves raw spikes, original settings, native bundles and start snapshots.
   Estimated retention is below 50 MB; no dense conductance recordings.
2. `uv run python experiments/exp172/compute.py --import-from DIRECTORY` imports
   an existing completed scratch data directory without simulation. It validates
   the full trial grid and recomputes its metrics, checks parameter-source hashes,
   and preserves original payloads, history, timing and source code provenance.
3. `uv run python experiments/exp172/analyse.py --source COMPUTE_ID` independently
   recomputes and checks all trials from their saved spikes without simulation.
4. `uv run python experiments/exp172/present.py --source ANALYSE_ID` renders the
   four figures (rasters, spatial profiles, phase/selection and weights) without simulation. Optional `--cue` and `--seed`
   choose an existing trial. The default selection maximises recorded joint
   fidelity/selection with phase criteria; no new search is performed.
5. `uv run pytest experiments/exp172/test.py -q` runs non-simulation checks.

Every stage accepts `--run-id`; downstream inputs must be explicit completed
Pingstore IDs. Settings, input generation and network authoring live in recipe.py;
compute, analysis and presentation have separate entry points and no automatic
upstream or downstream execution. Imports have no execution side effects.

The four figures retain all E/I spikes, spatial profiles, Gaussian references,
unwrapped phase, 100 ms moving-window A-like/B-like amplitudes (5 ms step), and
both executed sender→C matrices. Overlapping fits blur short events and are not
independent samples. All executed weights come from saved native snapshots;
bundle diagrams describe graph initializers before parameter replacement.

The initial migration imported existing evidence rather than claiming a fresh simulation. Subsequent feedback-search and follow-up runs explicitly execute new simulations.
Original settings and execution timings remain unchanged; migration timings are
recorded separately. The original source archive is retained with imported
compute evidence. No scratch code, local scratch path or other experiment module
is needed by the formal implementation. Backup and publication are separate
operations; migration does not publish the report.

## Local recurrence and inhibitory-feedback search

1. `compute.py --stability-sweep` executes a paired 12-condition local CPU grid:
   C E→E scales 0, 0.1, 0.25 and 0.5 of inherited strength, circular Gaussian
   recurrence SD 0.025 (ten E neurons), crossed with C I→E scales 1.25, 1.375
   and 1.5. The cue is fixed at 3 Hz/channel and the input seed at 185.
   The original model is the zero-recurrence / 1.25-inhibition control.
2. Override only the requested axes with `--recurrence-scales`,
   `--inhibition-scales` or `--seeds`. Complete phase runs retain each trial's
   original settings, raw spikes, native bundle and start snapshot. Expected
   retained size is below 60 MB for the initial grid.
3. `analyse.py --stability-source SEARCH_COMPUTE_ID` independently recomputes
   spatial fits and phase metrics. Spatial competition uses 40 ms windows with
   5 ms steps in 250–1100 and 1450–2300 ms, broader than the original fidelity
   windows. Unresolved responses count as failures. The longest excursion is
   the span of consecutive failed window centres, not an exact event duration.
4. Switch latency is the end of the first wholly post-switch fit beginning
   100 ms of continuous B dominance. Overlapping windows remain correlated.
   Selection must also preserve both-window ER ≥ 0.90 and direct-copy R² ≥ 0.85;
   a persistent wrong pattern or suppression of both patterns is not improvement.
5. A full follow-up run can use `compute.py --c-ee-scale VALUE --c-ie-scale VALUE
   --c-ee-kernel-sigma VALUE [--seeds ...]`. This preserves the standard cue
   sweep and single-sender controls; analysis and presentation remain separate.

The 21-condition paired search selected recurrence scale 0.05 and inhibition
scale 1.31, preserving both-window fidelity and selection targets while reducing
the worst-window competing-pattern fraction from 20.0% to 14.1% on seed185.
The displayed operating cue remains 3 Hz/channel. The original zero-recurrence
baseline and all unsuccessful conditions remain immutable. This is selected
single-seed tuning; neither elimination of excursions nor generalisation is claimed.
Full follow-up compute pins both search analyses with `--tuning-sources ID ID`.
