# SHD: paired COBA and PING training

Compare two-layer COBA and PING classifiers on SHD, using a stratified validation
holdout and one matched held-out response. Generate the report's training curves,
rasters, autocorrelograms and circuit diagrams. The official test set is unused.

1. `compute.py`: optional `--run-id`; optional `--parallel` (one spawned process
   per model, requiring two visible CUDA GPUs and at least two CPU threads);
   optional `--protocol {full,local-check}`
   (default `full`: 50 epochs on all official training utterances; `local-check`:
   one epoch with one training and validation utterance per class).
   Runs COBA then PING by default, or both concurrently with `--parallel`,
   saving native bundles, circuits, training snapshots,
   best-validation snapshots, replay-start snapshots, metrics and recordings.
   `SNNLAB_DEVICE` overrides device selection; `SHD_CACHE` selects the verified
   archive cache; `PINGLAB_CPU_THREADS` sets the positive CPU thread count
   (default 4). Local checks are diagnostics, not performance evidence.
2. `analyse.py`: required `--source <compute-run-id>`. Validates the explicit
   run and saves training measurements and population autocorrelograms.
3. `present.py`: required `--source <analyse-run-id>`. Writes the report's six
   SVGs and `numbers.json`, retaining the explicit compute ancestry. The report
   reads parameters directly from that compute run's native snapshots.

Regenerate only network diagrams while preserving an explicit presentation's
scientific figures and data: `uv run python experiments/exp124/present.py
--redraw-from PRESENT_ID [--run-id ID]`.
