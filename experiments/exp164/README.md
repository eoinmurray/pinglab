# exp164 — Untrained digit routing

Standalone CPU simulation of a fixed A/B → C spiking circuit driven by two
8×8 scikit-learn digit images. No other experiment or saved run is required.
No classifier, decoder or synaptic weight is trained.

```sh
uv run python experiments/exp164/compute.py
uv run python experiments/exp164/analyse.py --source COMPUTE_ID
uv run python experiments/exp164/present.py --source ANALYSE_ID
uv run pytest experiments/exp164/test.py -q
```

`recipe.py` owns the circuit, fixed parameters, image selection, rate encoder,
input seeds and observation settings. `compute.py` accepts `--seed` (input seed, default 101); the native network seed is 85.
Every stage accepts `--run-id`; analysis and presentation require explicit input
IDs. Compute retains generated inputs, spikes, source images, a compiled bundle,
circuit diagram and native simulation-start snapshot. Two conditions share their
sensory/background draws and cue draw, with opposite cue orders. Nominal graph
initializers are transformed into pixel-block sensory weights and normalized 2D
Gaussian A/B → C E weights before the native snapshot and simulation.

Trials last 2,400 ms and switch cue at 1,200 ms. Figures display 0–1,100 ms;
phase/spatial measurements use 650–1,100 ms and conductance close-ups use
900–1,050 ms. Rates/phase use 1 ms SD temporal smoothing; maps are raw E-cell
rates at fixed pixel locations. Corresponding axes share scales.

Conductances are analytical reconstructions from saved spikes, native weights,
planner delays and decay, with zero initial conductance, rather than native
recordings. Float64 population means can differ from native float32 traces.
Low inhibition is a proxy, not measured transmission gain. This is one fixed
image pair and input realization, not a timing intervention or decoding benchmark.

The standalone recipe replaces the import-only workflow using the recovered
original specification: first digit-0/digit-1 samples, 28 Hz C background,
0.45-pixel kernel, input seed 101, native seed 85 and recorded incoming-weight
constants. Within-population connections explicitly exclude self-connections.
These are versioned recipe constants; no historical run is loaded at execution.

The corrected fresh two-condition run reproduced the historical trial's images,
all input arrays, every native parameter array and all six populations' spikes
exactly over the full 2,400 ms. Historical evidence and ancestry were recovered
from Hetzner into a separate directory and validated for this comparison. C's
pre-switch image correlations were 0.878/0.475 (A/B, A cued) and 0.358/0.952
(A/B, B cued); cued sender–C PLVs were 0.959 and 0.980. These are one-pair,
one-seed results. Analysis records image correlations and cued-minus-uncued
correlation margins alongside phase metrics.

The first local reconstruction changed the image pair, background, kernel, seeds
and weight constants and omitted recurrent topology; it failed to reproduce the
routing effect. Those runs remain unchanged as failed-attempt provenance.
Corrected outputs are exp164-r006-compute → exp164-r007-analyse (or a later
analysis/presentation of that same compute). Arrays and native artifacts occupy
tens of files, with expected retained size below 20 MB per pipeline.

### Compact circuit diagram

The diagram follows exp160's mathematical labels, compact E/I pairs and recurrent
loops, with mirrored A/B pathways converging on C in the centre. Redraw only the
network from a validated presentation, preserving its result figures and numbers:

```bash
uv run python experiments/exp164/present.py --redraw-from PRESENT_ID
```
