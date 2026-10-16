# SHD: paired COBA and PING training

This experiment does one comparison: train the calibrated two-layer COBA and
PING networks, measure their training curves, and record one matched held-out
response with the final weights. The report presents a parameter table, two
training-curve figures and one two-layer raster per model. The rasters illustrate
one preselected utterance; they are not estimates of typical population activity.

## Code

1. `recipe.py` declares the networks, native initialization and training
   specifications. The small registered extensions implement the calibrated
   equal-class objective, Adam optimizer, centered readout, fixed identity gain
   and rate-normalized score. These preserve the existing scientific recipe.
2. `compute.py` verifies/downloads the official SHD training archive, creates
   the seed-42 per-class 10% validation holdout, and calls native SNNLab `train`
   and `infer` for both models. The official test set is untouched.
3. `analyse.py` reads native epoch measurements and parameter snapshots,
   validates the split, paired initialization and learned/fixed weights, and
   extracts the parameter table from the executed graph/training definitions.
   A small compatibility reader supports the immutable current report evidence.
4. `present.py` renders six SVG figures and `numbers.json`.

There are no historical variants, calibration/gamma phases, separate raster
commands, consolidation commands or backend-comparison presentations.

## Commands

Each command produces one new immutable run. Analyse and present require the
explicit completed run ID returned by the previous stage.

```sh
# Full training: 50 epochs, batch 256, both models. Substantial computation.
.venv/bin/python experiments/exp124/compute.py --device cuda
.venv/bin/python experiments/exp124/analyse.py --source <compute-run-id>
.venv/bin/python experiments/exp124/present.py --source <analyse-run-id>

# Small local pipeline check: 20 training / 20 validation utterances, one epoch.
.venv/bin/python experiments/exp124/compute.py --smoke --epochs 1 --device cpu
```

Only epochs, batch size, seed and execution settings are CLI overrides. Scientific
architecture and optimizer choices live directly in the recipe. Both models run
sequentially in one compute invocation. The sample is always the first validation
utterance in source order, with one seed+4 corruption draw shared by both models;
sample inference runs on CPU without optimizer updates.

The shared Modal launcher supports the same paired workflow:

```sh
uv run --no-project --python 3.12 --with 'modal>=1.1,<2' --with PyYAML \
  modal run --detach experiments/helpers/modal_compute.py \
  --experiment exp124 --gpu H100 --cpu 4
```

A submission requires separate authorization for paid full training. No such
submission was made during cleanup. Legacy `--variant`, `--models`,
`--full-dataset`, example-count and readout-gain options were removed.

## Retained outputs

1. Compute retains complete native network bundles and start/end parameter
   snapshots for training and inference, compact native epoch metrics,
   the dataset selection, and the single matched input/response recording.
   Final training snapshots are the authoritative trained weights; no separate
   handwritten initial/final weight copies are written.
2. The full event dataset and native optimizer/RNG checkpoints are temporary
   compute scratch. A checkpoint transfers final weights into native inference;
   scratch is discarded at successful completion. Completed runs do not promise
   resumable training. The verified source archive and split identity remain
   recorded, so new executions can reacquire the source.
3. Analyse retains report measurements and the parameter table. Present retains
   six SVGs and report numbers, without PNG duplicates. Checksum-protected
   native bundle/snapshot files remain intact under Pingstore's reversible path
   mapping; their required retention takes precedence over minimizing copies.

## Current report and verification

The report remains pinned to `exp124-r151-present`, with ancestry
`exp124-r149-compute` → `exp124-r150-analyse` → `exp124-r151-present`.
These three runs and their trained weights were left byte-for-byte intact.
The native snapshot requirement applies to new writers; r149 retains its older
NPZ weights and native checkpoints unchanged.

Reanalysis and presentation from r149 produced `exp124-r152-analyse` and
`exp124-r153-present`. The one-epoch local check produced
`exp124-r154-compute` → `exp124-r155-analyse` → `exp124-r156-present`.
A final check of the reduced-data writer produced `exp124-r157-compute`; its
weights and measurements match r154 exactly. None replaces the article pin.
Reduced-data checks are not performance evidence.

Verification covered all 51 epoch rows per model, weight-change measurements,
timing and raster summaries. Replaying the retained final weights through the
simplified graphs reproduced every recorded score and raster spike exactly.
The full split and noisy sample input also matched exactly; all 37 original
chain files retained their SHA-256 hashes. The site built successfully, figures
and the extracted table were inspected, and 23 artifact/dispatch tests passed.

Before any later pruning, build a separately authorized self-contained
replacement for the report evidence, prove equality of all epoch measurements,
raw sample spikes, parameter values and report numbers, and inspect its rendered
figures/table. Only then update the pin and review an experiment-scoped,
hash-bound prune plan. No full retraining, pruning, publication or remote
submission was performed during this cleanup.

## Raster autocorrelation panels — 16 October 2026

Reanalysis of the retained full-training compute run `exp124-r149-compute`
produced `exp124-r159-analyse`; presentation produced `exp124-r160-present`. A display-only revision produced
`exp124-r161-present`, previously selected by the article, with all ACF y axes
limited to −0.05–1 and the estimator equation added to Methods. Each layer raster (A, C) has a short corresponding
autocorrelation panel (B, D), with separate E/I traces for PING. Analysis sums
spikes across each population in 1 ms bins over the full utterance, subtracts
the temporal mean and divides lagged product sums by the zero-lag sum of
squares, without overlap correction. Lags span 0–200 ms; constant series yield
undefined values. No training or inference was rerun. Prior training and raster
measurements match r152 exactly. An initial analysis check also completed as
`exp124-r158-analyse`; r159 uses an explicit sum of squares to avoid local
BLAS warnings. Older runs and their exports remain intact.


### SNNLab estimator revision

The article now selects `exp124-r163-present`, generated from
`exp124-r162-analyse` and the same retained `exp124-r149-compute` spikes.
Analysis calls `snnlab.analysis.spike_autocorrelogram` directly with 1 ms bins
and a 200 ms maximum lag; the local adapter only serializes arrays and maps
undefined values to JSON nulls. The estimator uses uncentered counts, per-lag
overlap correction and squared-mean normalization, with a unity independence
baseline and undefined zero lag. Methods Equation 2 and both captions describe
this definition. Each short panel includes all E/I peaks with 10% upper
headroom and a dotted baseline at one. All six curves were checked against
direct SNNLab calls; other analysis measurements remain unchanged. No training
or inference was rerun.


### Footer network diagrams

`exp124-r164-present` adds COBA and PING circuit diagrams generated through
SNNLab's diagram/render APIs from the executed graph and training definitions
in the validated retained compute run. The historical evidence predates Lang
bundle manifests, so presentation wraps these definitions in an in-memory
Bundle for diagram projection only; it does not load or re-execute a network.
The circuit projection omits measurement reductions, and disconnected output
boxes for those reductions are also omitted. Both circuits appear in the
report footer immediately before the parameter table. Other presentation
measurements remain unchanged.
