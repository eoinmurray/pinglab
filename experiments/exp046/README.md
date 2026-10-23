# exp046

Measure excitatory spikes per inhibitory-burst cycle in eighteen PING networks. Compare pooled neuron–cycle distributions with equally weighted network distributions.

## CLI

Run each stage independently from the repository root with `uv run python`.

1. `compute.py`: `--source` (required), `--run-id` (optional).
2. `analyse.py`: `--source` (required), `--frequency-source` (required), `--run-id` (optional).
3. `present.py`: `--source` (required), `--run-id` (optional).
4. `test.py`: `uv run pytest experiments/exp046/test.py`; optional pytest arguments.

Optional environment: `PINGLAB_SMOKE=1` selects the bounded smoke profile; `SNNLAB_DEVICE` (or the legacy `PINGLAB_DEVICE`) selects the execution device.
