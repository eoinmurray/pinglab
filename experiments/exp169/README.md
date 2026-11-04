# exp169 — COBA with asynchronous GABA inhibition

Compare one unpenalised TR02 COBA raster, its PING reference, and frozen COBA
receiving one private inhibitory event stream per E neuron. The streams match
each E neuron's PING conductance mean and approximately its temporal SD over the
same 200-ms digit-0 input. GABA decay is 6 ms. No classifier is retrained.

1. `compute.py --import-source RETAINED_ID [--run-id ID]` imports authenticated
   scratch evidence without simulation. The migration source is
   `exp022-r021-compute-phase-scratch-artificial-inhibition`; its bank is pinned
   in its run record. Historical settings, scripts and execution times are retained.
2. `compute.py [--run-id ID]` explicitly runs calibration and two native CPU
   probes (baseline and artificial inhibition). This requires simulation authorization.
3. `analyse.py --source COMPUTE_ID [--run-id ID]` verifies source inputs, checkpoint
   parameters, native snapshots, the delayed GABA filter, moments and baseline
   per-neuron counts; records measurements without simulation.
4. `present.py --source ANALYSE_ID [--run-id ID]` renders three rasters and native
   architecture diagrams, with evidence-bound report numbers.
5. `uv run pytest experiments/exp169/test.py` runs non-simulation checks.

Run commands from the repository root with `uv run python experiments/exp169/`.
Imports never execute stages; no file in `.scratch` is needed.

This is a single-image diagnostic with one artificial input seed. The streams
are moment-conditioned Bernoulli-discretized Poisson draws, not unconditioned
Poisson samples. The fitted event rate is not constrained to PING's I rate.
Removing shared inhibition also removes feedback and changes higher-order input
statistics. COBA and PING have different learned weights. Matching conductance
moments does not match voltage-dependent inhibitory current or isolate synchrony.
The native baseline preserves every original E spike count; 20 spike timings
shift relative to the legacy recording. Original failed calibration/replay
attempts remain historical scratch provenance and are not passed results.


## Small classification test

1. `uv run python experiments/exp169/compute.py --classification` explicitly
   evaluates 100 official MNIST test images (10 per digit) under COBA, PING and
   image-conditioned artificial inhibition. The same sensory encoding is used
   across all three, with frozen original readouts, 200-ms trials and no burn-in.
   This is a separate `classification` compute phase; expect about five minutes
   on one local CPU thread and less than 1 GB of retained evidence.
2. `uv run python experiments/exp169/analyse.py --source RASTER_COMPUTE_ID --classification-source CLASSIFICATION_COMPUTE_ID`
   validates the raster and all 300 classification decisions/start snapshots,
   checks conductance calibration, and combines both evidence branches.
3. `uv run python experiments/exp169/present.py --source COMBINED_ANALYSIS_ID`
   renders the existing figures and exposes classification accuracy through the
   same report runset. Reports never hard-code run IDs or accuracy values.

Classification selects the largest mean pre-reset readout voltage. Artificial
inputs are calibrated separately against PING on the same test image, without
labels or classifier scores. This diagnoses the intervention rather than a
standalone classifier with a fixed inhibition generator. Full conductance traces
are transient; predictions, per-cell calibration moments and native start
snapshots are retained. No scratch imports are used.
