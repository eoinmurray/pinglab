# exp082

Evaluate frozen variable-rate PING classifiers on matched digit streams while retaining hidden state and supplying explicit decision boundaries.

## CLI

Run each stage independently from the repository root with `uv run python`.

1. `compute.py`: `--source` (required), `--run-id` (optional), `--shard-index` (optional).
2. `analyse.py`: `--source` (required), `--showcase-source` (optional), `--run-id` (optional).
3. `present.py`: `--source` (required), `--run-id` (optional).
4. `custom/hpc.py`: `--source` (required), `--showcase` (required), `--plan` (required), `--account` (required), `--cpu-account` (required), `--mnist-cache` (required), `--profile` (optional; default production), `--partition` (optional; default ampere), `--cpu-partition` (optional; default icelake), `--walltime` (optional; default 04:00:00), `--collector-walltime` (optional; default 01:00:00), `--cpus` (optional; default 4), `--memory-gb` (optional; default 32), `plan` (required), `--live` (optional), `--test-only` (optional), `stage` (required).
5. `custom/consolidate.py`: `--evaluation` (required), `--showcase` (required), `--run-id` (optional).
6. `test.py`: `uv run pytest experiments/exp082/test.py`; optional pytest arguments.

Optional environment: `PINGLAB_SMOKE=1` selects the bounded smoke profile; `SNNLAB_DEVICE` (or the legacy `PINGLAB_DEVICE`) selects the execution device; `PINGLAB_EXP082_STREAMS_PER_CELL`, `PINGLAB_EXP082_DIGITS_PER_STREAM` and `PINGLAB_EXP082_STREAM_BATCH_SIZE` override the stream profile through `recipe.configuration(...)`.

Regenerate the documented-model illustrations without running a scientific stage:
`uv run python experiments/exp082/present.py --draw-schematics`.
These authored SVGs do not reconstruct missing historical bundles.
