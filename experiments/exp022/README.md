# exp022

Train the seven MNIST classifier families, retain their validation histories and checkpoints, and measure fixed-image activity for downstream comparisons.

## CLI

Run each stage independently from the repository root with `uv run python`.

1. `compute.py`: `--bank-create` (optional), `--bank-status` (optional), `--bank-list` (optional), `--bank-train-cell` (optional), `--bank-validate` (optional), `--bank-finalize` (optional), `--bank` (optional), `--execution-origin` (optional; default slurm-wilkes), `--tier` (optional; default all), `--retry-only` (optional), `--recover-stale` (optional), `--run-id` (optional), `--import-source` (optional), `--source` (optional), `--diagnostics` (optional).
2. `analyse.py`: `--source` (required), `--run-id` (optional).
3. `present.py`: `--source` (required), `--retained-presentation` (optional), `--run-id` (optional), `--training-overview` (optional).
4. `custom/hpc.py`: `--root` (required), `--plan` (required), `--replacement` (optional), `--account` (required), `--mnist-cache` (required), `--partition` (optional; default ampere), `--walltime` (required), `--collector-walltime` (optional; default 02:00:00), `--concurrency` (required), `--cpus` (optional; default 4), `--memory-gb` (optional; default 32), `plan` (required), `--live` (optional), `--test-only` (optional), `stage` (required).
5. `custom/reuse.py`: `--reuse-plan` (optional), `--reuse-reserve` (optional), `--reuse-status` (optional), `--reuse-train-cell` (optional), `--reuse-finalize` (optional), `--run-id` (optional), `--execution-origin` (optional; default slurm-wilkes), `--recover-stale` (optional), `--recover-finalization` (optional).
6. `test.py`: `uv run pytest experiments/exp022/test.py`; optional pytest arguments.

Optional environment: `SNNLAB_DEVICE` (or the legacy `PINGLAB_DEVICE`) selects the execution device.

Regenerate the documented-model illustrations without running a scientific stage:
`uv run python experiments/exp022/present.py --draw-schematics`.
These authored SVGs do not reconstruct missing historical bundles.
