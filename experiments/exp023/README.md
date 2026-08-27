# Exp023 — PING fundamentals

The implementation is maintained against Experiment Runner Guide 2.0.0,
Storage Guide 2.0.0 and Writing Guide 8.0.0. Current operational requirements
are defined by the [Runner Guide](../README.md), [Storage Guide](../../tools/pingstore/README.md)
and [Writing Guide](../../writings/README.md). These commands are independent and
never materialize or publish. This implementation change does not establish a scientific rerun.

```sh
uv run python experiments/exp023/compute.py
uv run python experiments/exp023/analyse.py --source <exp023-compute-run>
uv run python experiments/exp023/present.py --source <exp023-analyse-run>
```

Compute retains two raster trials and fourteen matched-drive sweep trials.
Analysis reads explicitly selected v3 evidence, retaining spectra, selected-cell
traces/currents, firing rates and peak estimates. Presentation reads those
measurements and the pinned computation's rasters, without recomputing spectra,
selecting cells or simulating. Its flat export retains the existing figure names
and a numbers.json projection. Select that present run separately for preview;
materialization/publication requires separate authorization.

## Scientific preservation

- Loop strengths remain 0 and 1.5, with input parent weights 1.5/0.3 and 95%
  initialization zeroing. Raster drives remain 5/45 Hz, and the f–I grid remains
  2, 5, 10, 20, 40, 70, 100 Hz. Each condition has one seed-42 trial.
- The raster command used 1,024 input channels. The old f–I command omitted
  --n-in, so the current simulator used 784. This difference is now explicit,
  not silently corrected. Matching their geometries requires a scientific decision.
- Production duration remains 400 ms; PINGLAB_SMOKE=1 retains the existing
  200 ms profile. Only compute reads this environment setting. Downstream
  stages use retained settings; replay records the compute profile.
- The full-trial population rates, Welch estimator, 5–150 Hz peak search,
  half-bin-clamped interpolation and highest-spike-count neuron selection are
  preserved. The historical f_gamma_hz field is not a rhythmicity significance
  test: it reports the band peak only when I activity is present. It must not
  be interpreted as proof of gamma or as an independent measured absence in COBA.
- Current simulator defaults are not evidence of historical execution. The local
  production chain is `exp023-r001-compute-local`, `exp023-r002-analyse-local`
  and `exp023-r003-present-local`. This conformance pass reused it. The
  article renders quantities only from a selected presentation; absent evidence
  produces the shared unavailable-data notice.

## Storage and dispatch boundary

Every new stage uses v3, stage-labelled IDs, atomic completion and both payload
and authoritative-manifest input pins. Simulation commands, configurations and
logs are retained under provenance/. Failed executions remain hidden. --run-id
accepts only an unused reservation; reruns use fresh identities.

The former combined commands, including --plot-only and --skip-training, fail
with stage directions before creating outputs. New collection plans reserve
all three stage IDs before scheduler dispatch and retain stage references;
old monolithic exp023 plans are rejected, never rewritten or recaptured as v2.

Historical runs and reservations are untouched. No v2 import or migration is
provided. Exp023 enforces Storage 2.0.0 locally. Repository-wide guides also
require v3; remaining shared v2 readers and unrelated legacy runners are
nonconforming implementations, not permitted compatibility exceptions.

## Verification

Tests use fabricated arrays in temporary stores, not scientific experiments:

```sh
uv run pytest experiments/tests/test_exp023_drive_provenance.py experiments/tests/test_exp023_stages.py
```
