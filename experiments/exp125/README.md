# Scikit Digits

Collection: `datasets`. A supervised fully connected 64 → 128 → 10 LIF
baseline using scikit-learn's bundled 8 × 8 handwritten digits.

## Scientific protocol

1. Divide intensities by 16. Stratify examples into 60% training, 20%
   validation and 20% test, with split seeds 42 and 43. Splits are by image,
   not by writer. Record original pixels, labels and exact indices.
2. Generate independent Bernoulli events with probability pixel × 0.25 per
   1 ms step (250 Hz maximum), for 50 steps. This approximates a Poisson rate
   code on the discrete grid; at this probability it is not a fine-grid
   approximation. Reset all hidden/output states at each presentation.
3. Use snnTorch LIF hidden/output layers, decay 0.9, threshold 1, subtractive
   delayed reset and fast-sigmoid surrogate slope 25. Dense signed weights use
   PyTorch Linear default initialization; both biases start at 0.1. Sum output
   spikes over the whole presentation as class logits.
4. Train both dense layers and biases with Adam, learning rate 0.001,
   zero weight decay, batch size 64, 50 epochs, gradient norm clipped at 1.
   Weight initialization, minibatch ordering and training encoding use separate
   recorded seeds. Evaluate entire training/validation splits at epoch zero
   and after each epoch with fixed split-specific encoding draws and batches.
5. Select minimum validation cross-entropy from epochs 0–50, choosing the
   earliest epoch on ties. Evaluate test images once using this checkpoint
   and an independent fixed encoding seed. Save initial, final and selected
   parameters, learning curves and per-example test predictions. One seed
   establishes a baseline, not statistical robustness or a PING comparison.

## Independent stages

```sh
uv run python experiments/exp125/compute.py --cpu-threads 1
uv run python experiments/exp125/compute.py --phase sample-raster --source <compute-run-id> --cpu-threads 1
uv run python experiments/exp125/analyse.py --source <compute-run-id> --raster-source <sample-raster-run-id>
uv run python experiments/exp125/present.py --source <analyse-run-id>
```

Each command creates one validated v4 run. No command launches another stage,
selects latest evidence, or publishes. Compute uses CPU only; the dataset is
bundled with scikit-learn and needs no download. Recipes live in `recipe.py`.
Training wall time includes epoch-zero and per-epoch full-split evaluation and
selected-weight serialization; it excludes dataset loading, initialization,
final test evaluation and stage completion hashing. Total training-and-test
wall time adds final weight serialization, checkpoint loading and test inference.
Test inference has its own timer around evaluation of all 360 held-out images:
encoding, forward simulation, decoding and loss/accuracy calculation, in batches
of at most 256. It excludes checkpoint loading and prediction serialization.
This is a warmed CPU evaluation after training, not a cold-start latency.

The optional `sample-raster` compute phase reuses the explicitly named training
run's validation-selected parameters. It records the first validation image
with a fresh seed-46 encoding draw, zero initial states and no optimization.
Its before/after external-backend snapshots and raw input/hidden/output spikes
are new inference evidence; it does not retrain or reevaluate the test split.
Analysis checks training ancestry, selection and count-based prediction, then
measures pooled population-count autocorrelograms in 1 ms bins at 1–20 ms lags.
Present adds the hidden/output raster and autocorrelogram figure to the unchanged
training curves. Omitting `--raster-source` retains the training-only pipeline.

Follow the [Experiment Runner Guide](../../md/notes/experiment-rules.md),
[Pingstore Rules](../../md/notes/pingstore-rules.md) and
[Writing Guide](../../md/notes/writing-rules.md).
