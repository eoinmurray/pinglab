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

## Modal execution

Run a fresh 20-epoch job from the original checkpoint (not a continuation of the
three-epoch run), with one A10G GPU, four CPUs, 32 GiB RAM and a two-hour timeout:

```sh
uv run --no-project --python 3.12 --with 'modal>=1.1,<2' --with PyYAML \
  python -m libraries.modal_dispatcher dispatch --experiment exp171 \
  --gpu A10G --cpu 4 --memory 32768 --timeout 7200 \
  --input bank=exp022-r015-compute -- --epochs 20
```

Use the exact returned compute ID with the dispatcher's `check --collect`, then
run analysis and presentation separately. MNIST is downloaded into
`PINGLAB_DATASET_CACHE` on a fresh worker and checked by torchvision; actual
array hashes are retained. Source roles and source digest remain unchanged.
The original three-epoch evidence stays immutable; report values and figures
follow the currently selected completed runset.

The CUDA executor uses highest float32 matmul precision and batches ten independent
PING presentations during calibration. CPU calibration retains single-image
batches. Generator fitting remains on CPU, and per-image RNG streams are unchanged.
Snapshots record the actual resolved device. Accelerator arithmetic may differ
slightly from the original CPU pilot; primary comparisons remain paired within
each run. The initial Modal CPU attempt was cancelled after timings showed it
could not complete the requested run within its resource timeout.

The completed 20-epoch run uses local CPU. Both Modal attempts were cancelled;
measured throughput for this eager executor was better locally. The Modal command
above remains an execution option, while the first compute command is the local
route used for the completed evidence.

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
