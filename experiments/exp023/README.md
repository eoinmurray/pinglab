# Exp023 — PING fundamentals

## Current independent stages

Recipe v3 authors the untrained circuit with `snnlab.lang` and executes typed
`ExecutionSpec` requests through snnlab 0.3.0's `GraphExecutor`. Compute does not
invoke the simulator CLI or mutate simulator globals. Each stage is explicit:

```sh
uv run python experiments/exp023/compute.py
uv run python experiments/exp023/analyse.py --source <native-compute-run>
uv run python experiments/exp023/present.py --source <native-analyse-run>
```

Analysis and presentation accept only completed v4 evidence with the exact
native recipe v3. Older recipe v1/v2 computations cannot be reprocessed by these
entry points. Their stored runs, provenance and completed presentations remain
unchanged and viewable. There are no historical recording fallbacks, reporting
corrections or unused-readout drawing branches in the operational pipeline.
The retired combined launcher only explains the three independent commands.

## Scientific protocol

The circuit contains 1,024 E and 256 I neurons, with reciprocal loop strengths
0/1.5 and an I→E parent mean twice E→I. Input drives E only; same-population
recurrence and a classification readout are absent. Parent input weights have
mean/SD 1.5/0.3, lower-clamped Gaussian draws, 95% Bernoulli initialization
zeroing and survivor rescaling followed by fan-in normalization. Recurrent
parent SD is 10% of its mean, without initial zeroing.

Two scope trials use 1,024 input channels at 5/45 Hz. Fourteen matched-drive
f–I trials use 784 channels at 2, 5, 10, 20, 40, 70 and 100 Hz for both loop
conditions. Each trial uses seed 42, a 0.1-ms timestep, 400-ms production duration
(200 ms with `PINGLAB_SMOKE=1`), 2/6-ms AMPA/GABA decay and exact 1.2/0.6-ms E/I
refractory periods. Reciprocal synapses have explicit one-step delays.
Graph initialization has a different random-draw order from the historical CLI;
seed equality does not establish equality of realized recurrent matrices.

Analysis preserves full-trial rates, demeaned Welch density with one full-trial
window, the 5–150-Hz peak search and half-bin-clamped parabolic interpolation.
Traces select the first maximum spike-count neuron, E index zero when silent,
and omit silent I traces. A peak is reported only with I spiking; it is not a
rhythmicity significance test or evidence of measured gamma absence in COBA.

## Evidence and storage

Four flat scientific network units retain compiled graphs and realized weights.
Scope exports retain compact multimodal `recording.npz`; f–I exports retain
`spikes.npz` with online population counts. Graph digests, resolved Poisson
protocols, initialization metadata and per-trial timing live in `run.json`.
Repeated rates must realize identical weights within each condition.

Every stage uses the v4 atomic run contract, explicit input pins and independent
completion. Presentation uses the measured spectra, traces and compute rasters;
it does not select neurons again, simulate, recompute spectra or publish.
Preview selects a completed present run separately. Publication needs its own
authorization. Current targets are Runner Guide **4.9.0**, Storage Guide
**4.9.0** and Writing Guide **36.0.0**.

## Validation and completed native execution

The 400-ms production chain is `exp023-r017-compute` → `exp023-r018-analyse` →
`exp023-r020-present`. Its PING raster peak is 55.8505 Hz, E/I rates are
8.4326/75.0977 Hz, and COBA E rate is 23.8599 Hz. The native schematic omits
unused-readout arrows. These are new initialized results, not byte-identical
replacements of historical measurements. The `.legacy` marker is removed.

**19 focused checks passed** after native-only cleanup. They cover the independent
pipeline, checksum/pin validation,
rejection of old recipes and dense recording layouts, compact trace/count
measurements, graph/legacy conformance with identical weights and drive,
6-ms GABA decay and online reduction equivalence. Legacy executor use is
confined to numerical comparison tests, not production processing.

The cleanup was also checked against the native production data: reanalysis
`exp023-r021-analyse` and presentation `exp023-r022-present` completed without
simulation. Scientific JSON values and measured arrays match the original
native chain exactly. PNG bytes match; SVGs differ only in render timestamps
and generated identifiers. Historical stored runs were not modified.

```sh
uv run pytest experiments/exp023/test.py
```

Earlier execution, migration and correction accounts are preserved in
[HISTORY.md](HISTORY.md); their version and collection instructions are historical.
