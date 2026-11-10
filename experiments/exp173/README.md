# exp173 — PING inhibitory-pool size

Run four matched single-image trials from the trained exp022 TR02 (no rate penalty) PING network,
with 32, 64, 128 and the original 256 inhibitory neurons. Retain one raster per
size and plot E, I and total population-wide spike rates from those exact same trials.
These are single-image diagnostics, not dataset-wide averages. Accuracy uses the manuscript's ordered subset of 1,000 official MNIST test images,
sampled uniformly without replacement using seed 42, without class stratification.
The same images and one matched encoding draw per image are shared across sizes;
CPU encoder seed 20260415 and batch size 64 match exp025's evaluation protocol.
There is no retraining.

## Protocol

1. Pin the explicit source run and digest. The local source is
   `exp022-r015-compute`, digest
   `sha256:a542a714021fd7bb69c5d08ccf006763f19ce83b01d8ecdca86dba150118491d`,
   cell `ping__off__seed42`, epoch-50 final checkpoint. Validate its hash,
   dimensions and recorded exact refractory timing before loading.
2. Use the first digit-zero image in the source validation split. Encode one
   200-ms Poisson spike train with the first source validation encoder seed;
   reuse the exact same array at all sizes and verify its hash.
3. Keep all 1,024 E neurons and trained input/readout weights fixed. Smaller I
   pools are nested prefixes of one seed-42 permutation. Slice original E→I
   columns and I→E rows; multiply I→E weights by 256 divided by retained I count.
   At 256 neurons preserve all original arrays and ordering. This compensates
   nominal summed inhibition, without guaranteeing equal conductance or dynamics.
4. Start every trial from fresh state. Record all E/I spikes and population counts
   over the full realised presentation. Require raster counts to equal measured
   counts. Rates divide counts by population size and duration in seconds.
5. Use native SNNLab bundles, start snapshots and one circuit diagram per size,
   retained through Pingstore with source pins, configuration and timings.
   Save the 1,000 official-test indices, labels and per-condition predictions;
   accuracy uses mean pre-reset output voltage, without retaining its rasters. Expected
   retention is below 150 MB and 120 files. Preserve no full-dataset recordings.
6. Render four rasters with red I dots above black E dots and common axes, plus
   E, I and total population spike rates (spikes/s) versus pool size, with
   separate test accuracy (%) and E→I/I→E gain panels. Bind report parameters to the same selected
   ancestry's native snapshots and recorded configuration.

## Run

Compute requires explicit simulation authorization. Run stages in order:

```sh
uv run python experiments/exp173/compute.py --source SOURCE_RUN --source-digest SOURCE_DIGEST
uv run python experiments/exp173/analyse.py --source COMPUTE_ID
uv run python experiments/exp173/present.py --source ANALYSE_ID
```

Non-simulation checks:

```sh
uv run pytest experiments/exp173/test.py -q
```

The initial full-validation attempts were stopped or failed and are not the
source of the four-trial result. No backup, publication or cleanup is automatic.
