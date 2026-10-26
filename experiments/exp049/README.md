# exp049

Compare fixed and trainable recurrent PING weights, their initialization dependence, activity, accuracy and reference-image rhythmicity.

## CLI

Run each stage independently from the repository root with `uv run python`.

1. `compute.py`: `--source` (required exp022 training-bank run), `--run-id` (optional). With `--rasters-only`, `--source` instead identifies an existing exp049 endpoint compute run; replay the first test digit before and after training on identical input spikes.
2. `analyse.py`: `--source` (required endpoint compute run), `--run-id` (optional), `--raster-source` (optional paired-raster compute run).
3. `present.py`: `--source` (required), `--run-id` (optional).
4. `test.py`: `uv run pytest experiments/exp049/test.py`; optional pytest arguments.

Optional environment: `PINGLAB_SMOKE=1` selects the bounded smoke profile; `SNNLAB_DEVICE` (or the legacy `PINGLAB_DEVICE`) selects the execution device.
