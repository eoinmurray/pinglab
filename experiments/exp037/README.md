# exp037

Measure how spike deletion and insertion affect frozen COBA and PING classifiers, with insertion doses normalized to each network’s baseline activity.

## CLI

Run each stage independently from the repository root with `uv run python`.

1. `compute.py`: `--source` (required), `--run-id` (optional), `--shard-index` (optional).
2. `analyse.py`: `--source` (required), `--run-id` (optional).
3. `present.py`: `--source` (required), `--run-id` (optional).
4. `custom/hpc.py`: `--source` (required), `--plan` (required), `--account` (required), `--cpu-account` (required), `--mnist-cache` (required), `--profile` (optional; default production), `--walltime` (optional; default 02:00:00), `plan` (required), `--live` (optional), `--test-only` (optional), `stage` (required).
5. `test.py`: `uv run pytest experiments/exp037/test.py`; optional pytest arguments.

Optional environment: `PINGLAB_SMOKE=1` selects the bounded smoke profile; `SNNLAB_DEVICE` (or the legacy `PINGLAB_DEVICE`) selects the execution device.

Regenerate the documented-model illustrations without running a scientific stage:
`uv run python experiments/exp037/present.py --draw-schematics`.
These authored SVGs do not reconstruct missing historical bundles.
