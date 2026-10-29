# exp025

Compare held-out accuracy and firing rate across COBA/PING activity ceilings, then test cycle participation and sensitivity to input coupling.

## CLI

Run each stage independently from the repository root with `uv run python`.

1. `compute.py`: `--source` (required), `--run-id` (optional).
2. `analyse.py`: `--source` (optional), `--reuse-analysis` (optional), `--baseline-bank` (optional), `--run-id` (optional).
3. `present.py`: `--source` (required), `--run-id` (optional).
4. `custom/hpc.py`: `--source` (required), `--plan` (required), `--account` (required), `--cpu-account` (required), `--mnist-cache` (required), `--profile` (optional; default production), `--walltime` (optional; default 02:00:00), `plan` (required), `--live` (optional), `--test-only` (optional), `stage` (required).
5. `test.py`: `uv run pytest experiments/exp025/test.py`; optional pytest arguments.

Optional environment: `PINGLAB_SMOKE=1` selects the bounded smoke profile; `SNNLAB_DEVICE` (or the legacy `PINGLAB_DEVICE`) selects the execution device.

Regenerate the documented-model illustrations without running a scientific stage:
`uv run python experiments/exp025/present.py --draw-schematics`.
These authored SVGs do not reconstruct missing historical bundles.
