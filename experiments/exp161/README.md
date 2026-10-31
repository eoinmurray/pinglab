# exp161 — Competitive entrainment

One mean-field A/B → C trial with equal sensory drives and equal sender→C
projections. A separate constant-rate spike-input cue is represented by its mean
rate and synaptic filter, targeting A before 1,200 ms and B afterward. Default
settings reproduce the tuned scratch example: sensory 24 Hz/channel, cue
6 Hz/channel, C background 28 Hz/channel, coupling 0.04 µS per pathway.

1. `recipe.py` owns resolved settings, the native graph and mean-field reduction.
2. `compute.py --parameter-source COMPUTE_ID` initializes and integrates one trial.
   The explicit parameter reference is `exp159-r001-compute`. Optional overrides:
   `--coupling-us`, `--cue-hz`, `--background-hz`. Optional migration arguments:
   `--exploration-source PHASE_ID`, `--preserve-exploration DIRECTORY`.
3. `analyse.py --source COMPUTE_ID` validates the saved native trajectory and cue
   routing, then measures phase locking in the two predeclared epochs.
4. `present.py --source ANALYSE_ID` produces the three-panel figure, network diagram
   and report numbers. Each stage accepts `--run-id`.
5. `uv run pytest experiments/exp161/test.py -q` checks phase discrimination,
   symmetric architecture, cue routing, settings isolation and standalone imports.

```sh
uv run python experiments/exp161/compute.py --parameter-source exp159-r001-compute
uv run python experiments/exp161/analyse.py --source COMPUTE_ID
uv run python experiments/exp161/present.py --source ANALYSE_ID
```

The first successful formal compute is `exp161-r002-compute`, analysis
`exp161-r003-analyse`, presentation `exp161-r004-present`. An initial allocation
failed before simulation because collection metadata had not yet been written.
The selected formal trajectory exactly reproduces every time and state value of
`exp159-r015-compute-phase-scratch-competition`. Cued PLVs are 0.984766 and
0.979770; uncued PLVs are 0.250681 and 0.184397. This is finite-time preferred
locking, not an information-transmission test or a robust attention result.

Local incoming sums and cell parameters are imported through the validated native
exp159 reference. They are implemented by identical constant matrices in all
three circuits; heterogeneous reference matrices are not reused. Sensory gain is
included once in the native weights. The report's compact summary and two cue-phase
connectivity matrices bind to the same selected native start snapshot.

The formal compute pins the selected exploration run and its full completed
attempt chain. Its byte-verified `export/exploration.zip` preserves the original
scratch source, README, iteration table and incomplete first diagnostic. The
scratch directory was removed after successful migration checks; no completed
Pingstore runs were removed or rewritten. The original incomplete diagnostic
remains in Pingstore as well as in the archive. Reproduction does not require
scratch files. The migrated simulation took 16.408 seconds; analysis took about
0.1 seconds and presentation about 1 second. The exploratory nine simulations
(including the initial writer failure and selected repeat) took 147.046 seconds.

### Compact circuit diagram

The diagram follows exp160's mathematical labels, compact E/I pairs and recurrent
loops, with mirrored A/B pathways converging on C in the centre. Redraw only the
network from a validated presentation, preserving its result figures and numbers:

```bash
uv run python experiments/exp161/present.py --redraw-from PRESENT_ID
```
