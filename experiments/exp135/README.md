# exp135 — DVS Gesture: paired COBA and PING

Prepared comparison on the same 108 training / 11 validation examples, twenty
planned epochs. Training now uses fresh seeded random 500 ms crops at 1 ms
resolution, compiled neuron kernels, and batch size 54 selected by
GPU benchmarking. Validation uses fixed centred 500 ms crops. Previous
full-gesture training and benchmarks were cancelled; no production relaunch
has been made for this protocol.

## Scientific comparison

1. Both models have 16 E maps at 32 × 32, then 32 at 16 × 16, with 3 × 3 AMPA
   convolution, two 2 × 2 average pools and 11 nonspiking readout units. COBA
   has no I populations or feedback. PING additionally has 4/8 I maps and four
   trainable 1 × 1 reciprocal E/I kernels. These are a PING circuit proposal;
   gamma activity remains unverified. The cases have 27,424/28,064 trainable
   parameters, respectively, so parameter count is not matched.
2. Parameter-local CPU RNG draws give **exactly identical initial feedforward
   kernels and readout weights**, independent of graph declaration order.
   Dataset tensors, selection, example shuffling, seed 42, optimizer and
   duration are shared. All weights train; conductance weights stay nonnegative.
3. Source: Tonic's segmented [training archive](https://ndownloader.figshare.com/files/38022171),
   checked against MD5 `3a8f0d4120a166bac7591f77409cb105` from its
   [loader](https://github.com/neuromorphs/tonic/blob/develop/tonic/datasets/dvsgesture.py).
   Require 1,077 source examples. Selection seed 135 chooses 108 balanced training
   examples (10.03%); the two highest-numbered subjects supply one validation
   example per class. The official test archive is never downloaded.
4. Retain polarity, divide x/y by four and encode binary 1 ms occupancy.
   Store complete source events, their durations and sample offsets. Training
   chooses a seeded random 500 ms crop per presentation, refreshed each epoch;
   matched batches have identical crops across COBA and PING. Validation and
   matched sample rasters use fixed centred 500 ms crops. Zero-pad only source
   examples shorter than the crop. Readout voltage averages the crop's 500 steps.
   Fractional millisecond timestamps are preserved until binning.

5. Native SNNLab training uses AdamW, learning rate 0.001, no weight decay and
   gradient clipping at 1.0. Store online pre-update training and fixed-weight
   validation metrics, final/validation-selected checkpoints, authenticated
   network bundles, all native diagram views and start/end parameter snapshots.
6. Each worker also performs native inference on the **lexicographically first
   selected validation example**, using final-epoch weights and no optimization.
   The centred validation crop is identical for both cases. Retain all hidden
   population spikes as sparse `(step, batch, neuron)` coordinates and the
   scores. Selection does not depend on correctness or activity.

## One compute run, two GPUs

The shared Modal worker requests **two GPUs in one container**, then starts two
independent processes before waiting: shard 0 / COBA gets the first GPU and
shard 1 / PING the second. `CUDA_VISIBLE_DEVICES` isolates each process. This
is concurrent independent model training, not distributed data parallelism.
Using one container provides a real shared filesystem for Pingstore's locks.

`experiments.helpers.hpc.concurrent_compute` owns the worker/collector locks,
immutable completion records, payload checks and worker provenance. The recipe
owns the two work items and fixed allocation. No worker creates a separate
completed run or writes its sibling's condition directory.

1. Dispatch freezes the committed recipe, source identity, archive pin,
   work-item/GPU mapping and resources in the durable submission receipt.
2. The coordinator downloads/verifies the training archive once, selects and
   encodes the subset once, then freezes both dataset hashes, exact selection
   and resolved duration inside the unused run's `.scratch/paired-plan.json`.
3. Both processes write **the same reserved run**, under `export/coba/` and
   `export/ping/`, reading the shared NPZ inputs. A successful shard records its
   actual command, device visibility, timestamps, configuration and file hashes.
4. The coordinator invokes the exclusive collector only after both workers
   succeed. It validates exact condition coverage, input/source identity and
   payloads before creating `run.json`, preserving worker provenance and
   atomically exposing **one completed v4 compute run**.
5. Failure stops the sibling and leaves the common reservation hidden. A
   completed matching shard may be explicitly verified/reused; no automatic
   retry or resubmission occurs. Never remove stale locks automatically.
6. The existing Modal worker then validates and commits that completed run to
   the results volume. Local collection downloads and validates it under the
   same run ID. Analysis and presentation remain separate explicit stages.

The parallel path requires a **clean committed checkout**. Commit the requested
changes in an isolated checkout. The image uses that Git baseline and the
submitted dependency lock. An unpublished snapshot can be transferred directly
to Modal using an incremental Git bundle: set `PINGLAB_MODAL_GIT_BUNDLE` to
its absolute local path and `PINGLAB_MODAL_BASE_COMMIT` to its already-published
parent baseline. The launcher checks that the bundle names the exact local
HEAD; the image fetches the published baseline, applies the bundle, and resets
to the submitted commit. This requires no GitHub push. The local plan command works with uncommitted edits.

## Verified dataset preparation

The direct `ndownloader.figshare.com` endpoint serves the pinned archive. The
website endpoint can return an empty HTTP 202 response. Downloads accept only
complete HTTP 200 gzip responses, check declared length and the original MD5,
and retry at most three times before failing. Invalid responses never become
cached archives. An existing archive must pass the same checksum.

Before requesting GPUs, populate and verify the persistent cache on Modal CPU:

```sh
uv run --no-project --python 3.12 --with 'modal>=1.1,<2' --with PyYAML python -m modal run --detach experiments/helpers/modal_compute.py::prepare_gesture_archive
```

This verifies both the archive bytes and the 1,077-sample inventory and commits
the cache to the results volume. It performs no training and creates no
Pingstore compute run. The subsequent detached training job verifies and reuses
that cache independently of the local client.

## Plan and future Modal command

```sh
.venv/bin/python experiments/exp135/compute.py --plan
```

This prints configuration without downloading, reserving a run or training.
Scientific options remain committed in `recipe.py`; the stage CLI accepts
execution controls only. The old single-GPU invocation is rejected.

Detached submission command (requires committed source):

```sh
uv run --no-project --python 3.12 --with 'modal>=1.1,<2' --with PyYAML python -m modal run --detach experiments/helpers/modal_compute.py --experiment exp135 --gpu A100-80GB:2 --cpu 4 --timeout 86400
```

When dispatching from an isolated clean checkout, `--store /absolute/path/to/.pingstore`
keeps reservations and receipts in the main store. The collector accepts the
same destination with `--root /absolute/path/to/.pingstore`. Neither option
changes remote scientific inputs or the run identity.

This requests two 80 GB GPUs, four CPU cores split between the model processes,
32 GiB host memory and a 24-hour ceiling. The compiled 500 ms benchmark measured 37.3/23.4 examples/s and
18.4/21.2 GiB peak CUDA allocation for COBA/PING at batch 54. For the 108-example development subset, batch 54 gives two
updates per epoch and forty updates across twenty epochs, and benchmark throughput is not convergence evidence. Archive preparation initially occupies the allocated
container; the source archive is cached in the results volume. Graphviz is
required and installed by the default image. A reused image must also have
Graphviz, the same committed Git baseline and the submitted SNNLab version. No `--smoke`, seed, epoch or
batch-size overrides are accepted for this committed protocol.

After an explicitly authorised launch, download the single completed run:

```sh
uv run --no-project --python 3.12 --with 'modal>=1.1,<2' --with PyYAML python experiments/helpers/modal_jobs.py --run-id <printed-compute-run-id> --collect
.venv/bin/python experiments/exp135/analyse.py --source <compute-run-id>
.venv/bin/python experiments/exp135/present.py --source <analyse-run-id>
```

No command publishes the article or automatically executes a downstream stage.
A future local CPU compute command runs both cases sequentially in one run;
use it only deliberately, since complete sequences may take hours:

```sh
.venv/bin/python experiments/exp135/compute.py --dataset-archive /path/to/ibmGestureTrain.tar.gz --device cpu
```

`--download` permits acquisition when absent. `PINGLAB_DATASET_CACHE` changes
its cache location (default `/tmp/pinglab-datasets`). The complete training
archive is needed despite the 10% selection. Explicit incomplete-run operations
are `compute.py --run-id <reserved-id> --shard-index 0|1 --device cuda` and
`compute.py --run-id <reserved-id> --collect`; they do not select latest work.
A fresh invocation is required after an unreviewed failure or source change.

## Analysis and report

Following exp124's presentation pattern, analysis validates initial-weight and
sample-input pairing, finite learned weights, complete histories and the chosen
sample. It extracts parameter counts from executed definitions, measures rates,
and calls SNNLab's population spike autocorrelogram on all cells' counts.

Present matches exp124's figure dimensions, panel labels, typography and black/red
encodings. It exports separate COBA/PING loss-and-percentage-accuracy curves
with matched axes and an 11-class uniform-guessing line,
and one four-panel raster/ACF figure per case, plus native circuit/training/
expanded diagrams and report numbers. Raster displays use up to 256 E and 64 I
cells per layer, evenly spaced across flattened neuron indices. ACFs use all
cells, 1 ms bins, 0–200 ms lags, unity baseline and undefined zero lag, without
smoothing, transient removal or trial averaging. Stimulus envelopes and crop
boundaries contribute; one response cannot establish a general oscillation claim.

The report follows exp124's Abstract, four Results subsections and numbered
Methods structure. Network circuits and a full comparison table appear in the
Parameters footer; additional native views are expandable. Recorded values bind
to presentation measurements, and the table derives from executed graphs and
training definitions rather than current recipe defaults. It remains `txt` until real
validated presentation data exists; synthetic fixture output is never selected.

## Development checks

Tests use small synthetic archives and short CPU presentations in temporary
repositories. They check balanced selection, polarity/collision encoding,
paired initialization, changes to both feedforward kernels and the readout,
checkpoint restoration, both stage paths, one-run shard collection, missing or
corrupted outputs, changed data/source and isolated GPU process assignments.
These checks do not download the dataset, launch Modal or provide gesture
learning evidence. `recipe.py` can export both proposal bundles/diagrams via its
CLI without performing simulation or training.


## Local one-epoch diagnostic

```sh
.venv/bin/python experiments/exp135/compute.py --phase local-check --dataset-archive /Users/eoin/pinglab/temp/datasets/dvs-gesture/ibmGestureTrain.tar.gz --device cpu
```

The committed local-check recipe runs both complete networks for one epoch on
the first two training and two validation examples from the fixed selection,
using their first 250 ms. It retains normal native bundles, start/end snapshots,
final/selected checkpoints and matched sample inference. The result is a v4
`compute-phase-local-check` run with its actual prefix duration and selection,
not the full-gesture five-percent comparison. The comparison analysis rejects
this diagnostic phase so it cannot supply production report figures.


Use `--phase local-check-full` to repeat the two-example, one-epoch diagnostic
with complete gesture extents and padding to the longest of those four
selected examples. This is still a diagnostic phase, not the 108/11-example
comparison. Both variants retain their actual duration and source extents.

## Compiled crop benchmark

`compute.py --phase gpu-benchmark-coba` and `--phase gpu-benchmark-ping`
measure batch sizes 1, 2, 4, 8, 16, 32 and 54 on random 500 ms crops. Each
case uses four updates: one warmup, then three measured updates. Cases run
in isolated subprocesses and retain bundles, snapshots and benchmark results.
`modal_benchmark.py` dispatches one A100 80GB worker per model under a remote
coordinator. These phase runs cannot supply accuracy plots to the main report.
