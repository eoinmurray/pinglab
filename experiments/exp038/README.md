# exp038

Enable reciprocal inhibition in frozen feedforward classifiers and measure the resulting changes in firing rate and held-out accuracy.

## CLI

Run each stage independently from the repository root with `uv run python`.

1. `compute.py`: `--source` (required), `--run-id` (optional).
2. `analyse.py`: `--source` (required), `--run-id` (optional).
3. `present.py`: `--source` (required), `--run-id` (optional).
4. `custom/hpc.py`: `--source` (required), `--plan` (required), `--account` (required), `--cpu-account` (required), `--mnist-cache` (required), `--profile` (optional; default production), `--walltime` (optional; default 02:00:00), `plan` (required), `--live` (optional), `--test-only` (optional), `stage` (required).
5. `test.py`: `uv run pytest experiments/exp038/test.py`; optional pytest arguments.

Optional environment: `PINGLAB_SMOKE=1` selects the bounded smoke profile; `SNNLAB_DEVICE` (or the legacy `PINGLAB_DEVICE`) selects the execution device.
