# exp171 — Fine-tuning COBA with artificial inhibition

A bounded adaptation pilot: 200 original-training images, 100 original-validation
images, 20 epochs and 200 updates. The only reused scientific run is
`exp022-r015-compute`, authenticated by an explicit digest. No code from exp022
or any other experiment is imported. Installed SNNLab and maintained shared
libraries provide native execution and storage.

1. `uv run python experiments/exp171/compute.py` fits per-image private inhibitory
   generators using frozen PING, then fine-tunes the original COBA input and
   readout matrices. CPU locally, or CUDA through `SNNLAB_DEVICE=cuda`; the command uses the
   maintained Modal dispatcher. This command simulates and trains.
2. `uv run python experiments/exp171/analyse.py --source COMPUTE_ID` validates
   provenance, partition membership, calibration, updates, snapshots and paired
   validation metrics without simulation.
3. `uv run python experiments/exp171/present.py --source ANALYSE_ID` renders
   learning curves, an optional best-checkpoint raster and native diagrams for the normal report.
4. `uv run pytest experiments/exp171/test.py -q` runs non-simulation checks.

Compute accepts `--epochs N` for an explicit total epoch count; `--epochs 3`
reproduces the original pilot duration. Every stage accepts optional `--run-id`. Imports have no execution side effects.
Scientific evidence lives in Pingstore; existing runs remain immutable.

Each image has one fixed sensory encoding. Artificial probabilities and amplitudes
are fitted once to its frozen PING conductance moments over the scored 200 ms,
after a 60-ms inhibition-only prehistory, then held fixed. Training
uses fresh unconditional draws each epoch; validation uses three independent,
fixed draws per image. Moment matching therefore applies to calibration draws,
not exactly to subsequent draws. Private inputs carry amplitude-scaled events
through a frozen identity projection of 1 µS; this is equivalent to binary events
through per-image diagonal weights. The native delayed filter is checked in compute.

Every new training and validation draw includes its own 60-ms private inhibitory
history. Only its accumulated conductance and pending delayed event enter the
native runtime at image onset. Neuronal voltages, refractory state, sensory
conductance and readout start fresh; all 200 ms are scored. History events use
the same independent per-image/domain/epoch-or-draw RNG stream as their trial,
and hashes cover both history and scored events. No subsequent draw is
recalibrated, rejected or selected for its E response. The teacher is unchanged.
Compute retains one reproducible native initial-inhibition state and verifies
the full scored filter and deterministic epoch-zero replay. This removes the
zero-conductance startup ramp; other stimulus-onset dynamics may remain.

The completed `exp171-r009-compute` results predate this correction and remain
unchanged. The corrected 20-epoch run is `exp171-r012-compute`; it recalibrates
from the pinned source checkpoint.

The primary comparison is final epoch against epoch zero, averaged over the same
validation draws. Lower held-out cross-entropy and higher accuracy together mark
a promising pilot. Best-validation snapshots are retained separately; no early
stopping or hyperparameter search occurs. This is fine-tuning, one seed, and an
image-conditioned PING-dependent intervention, not an independent classifier or
a synchrony test. The original validation partition was already used for source
checkpoint selection. No official test images are used here. Negative short-pilot
results are inconclusive. Timings are retained per compute/analysis/presentation
stage; runtime is measured during execution. Expected retained size is <250 MB.

## Native benchmark and full mode

Both modes start from the pinned original epoch-50 COBA weights and use
`snnlab.sim.train()` with streamed sensory and inhibitory inputs. The effective
batch is 256, including the 156-image final batch. Each effective batch has one
AdamW update. The benchmark uses 412 source-training images and 32 source
validation images for two epochs, so it measures both effective batch shapes
and captures cold and warmed timings. The native trainer measures E rates with
fixed-weight split evaluation. Benchmark estimates include that evaluation. It
retains numerical parity checks,
native execution paths, peak device memory and a full-run cost projection.

Dispatch the benchmark with one A100 80 GB and a 20-minute timeout:

```sh
uv run --no-project --python 3.12 --with 'modal>=1.1,<2' --with PyYAML \
  python -m libraries.modal_dispatcher dispatch --experiment exp171 \
  --gpu A100-80GB --cpu 8 --memory 32768 --timeout 1200 \
  --input bank=exp022-r015-compute -- --benchmark
```

Use the dispatcher’s `check --collect` with the exact returned compute ID.
If the source was restored into a separate store, pass the same `--root` to
both dispatch and collection. Review `benchmark.json` and `native-metrics.json`
before authorizing a full run.

Execution comparisons accept `--compile-mode eager|default` and
`--microbatch-size 16|32|64`. For `default`, dispatch with
`PINGLAB_MODAL_CUDA_DEVEL=1`. To time the same already calibrated inputs, pass
`--input calibration=RUN_ID` to the dispatcher and
`--calibration-id RUN_ID --calibration-digest sha256:DIGEST` to compute.
The full-run estimate still includes calibration. Full mode must use the
execution settings recorded by the reviewed benchmark.

Full mode uses all 6,300/700 source-partition images for 20 epochs, with
epoch-zero and per-epoch validation on three fixed draws. Its compute
arguments are `--full --benchmark-id RUN_ID --benchmark-digest sha256:DIGEST`.
A Modal dispatch must explicitly supply both the `bank` and `benchmark` input
roles. The worker checks the reviewed benchmark against the checkpoint pin,
scientific settings, source hashes, lock, device and execution policy. Benchmark
completion and collection are independent of full-mode submission.

## Best-checkpoint digit raster

1. `uv run python experiments/exp171/compute.py --raster-source TRAINING_COMPUTE_ID`
   performs a short local CPU replay of the first validation batch at its retained
   best-validation checkpoint (lowest cross-entropy). It saves one trial's E spikes,
   private events and conductance, the native start snapshot, and batch score/rate
   agreement. The first image and draw are fixed without activity selection.
2. `uv run python experiments/exp171/analyse.py --source TRAINING_COMPUTE_ID --raster-source RASTER_COMPUTE_ID`
   validates both branches, including frozen checkpoint identity, digit membership,
   regenerated events, delayed filter, scores and rates.
3. Run presentation with that new analysis ID to replace the report's raster.
   The upper red dots are external inhibitory events, not biological I spikes.
   No training or calibration is launched by the raster command. Expected retained
   raster evidence is below 25 MB; previous runs remain immutable.
