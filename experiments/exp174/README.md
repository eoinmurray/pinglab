# exp174 — Decision latency and cumulative spike budget

Compare only exp171's unpenalised COBA, PING and artificial-I final checkpoints.
Use all 1,000 images of its retained ordered MNIST official-test subset, extending
the initial 100-image prototype. No retraining, checkpoint selection or new fitting.

1. Reproduce the original full 64-image CPU sensory batches from retained RNG
   states before slicing. Verify hashes against exp171. Every model receives the
   same sensory spikes; native simulation state starts fresh for each image.
2. Preserve the source models' refractory conventions: exact/following-steps for
   baseline COBA/PING; legacy COBA for the fine-tuned artificial-I model. Artificial
   inhibition uses the original frozen rates/amplitudes and seed derivation, with a
   new CPU Torch Poisson draw (not bit-identical to the GPU draw). Restore 60 ms of
   inhibition-only synaptic prehistory, leaving neuron state fresh at onset.
3. At every 0.1-ms step, predict from running mean pre-reset readout voltage.
   Ties use the lowest digit index. Define latency as the earliest step after
   which cohort accuracy stays at least 80% through 200 ms; report unreached if the
   final accuracy is below 80%. This retrospective measure is not an implementable
   label-free online stopping rule.
4. At that common model-specific time, average cumulative E spikes over all
   images, including errors. Show accuracy versus time and E count; retain PING I
   spikes and artificial events separately. Raw artificial bin counts include
   coincident events. Budgets exclude sensory/readout spikes, teacher calibration
   and artificial-I prehistory; they are not energy estimates.
5. Retain per-image prefix predictions and population counts, native bundles/start
   snapshots, input pins and timings. No per-neuron rasters. Expected retention:
   below 200 MB and 100 files for the 1,000-image run.

## Run

Explicit simulation authorization is required for compute. Default samples: 1,000.

```sh
uv run python experiments/exp174/compute.py --samples 1000
uv run python experiments/exp174/analyse.py --source COMPUTE_ID
uv run python experiments/exp174/present.py --source ANALYSE_ID
uv run pytest experiments/exp174/test.py -q
```

The current CLI supports prefixes up to the retained 1,000-image subset. Evaluation
of the full 10,000-image official test set is deferred and will require new matched
sensory streams and frozen-teacher moment fits for the remaining images, without
training or test-driven model selection. The 1,000-image extension does not
authorize that run.
