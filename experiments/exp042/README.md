# exp042

Replay count-preserving perturbations of recorded inhibitory spikes and measure their effects on frozen classifiers’ excitatory activity and accuracy.

## CLI

Run each stage independently from the repository root with `uv run python`.

1. `compute.py`: `--source` (required), `--run-id` (optional), `--shard-index` (optional).
2. `analyse.py`: `--source` (required), `--run-id` (optional).
3. `present.py`: `--source` (required), `--run-id` (optional).
4. `custom/hpc.py`: `--source` (required), `--plan` (required), `--account` (required), `--cpu-account` (required), `--mnist-cache` (required), `--profile` (optional; default production), `--partition` (optional; default ampere), `--cpu-partition` (optional; default icelake), `--walltime` (optional; default 02:00:00), `--collector-walltime` (optional; default 00:30:00), `--cpus` (optional; default 4), `--memory-gb` (optional; default 32), `plan` (required), `--live` (optional), `--test-only` (optional), `stage` (required).
5. `test.py`: `uv run pytest experiments/exp042/test.py`; optional pytest arguments.

Optional environment: `PINGLAB_SMOKE=1` selects the bounded smoke profile; `SNNLAB_DEVICE` (or the legacy `PINGLAB_DEVICE`) selects the execution device.
