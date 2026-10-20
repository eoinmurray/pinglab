# exp141 — Scikit-learn digits

Test whether fixed inhibitory feedback improves handwritten-digit classification
by comparing feed-forward COBA-LIF and PING networks on scikit-learn digits.
Compare their learning curves, held-out accuracy and spike activity.

## CLI

1. `compute.py`: `--run-id <id>` (optional); `SNNLAB_DEVICE` (optional environment
  override, for example `cpu` or `cuda:0`; default: CUDA when available, otherwise CPU).
2. `analyse.py`: `--source <compute-run-id>` (required), `--run-id <id>` (optional).
3. `present.py`: `--source <analyse-run-id>` (required), `--run-id <id>` (optional).
4. `test.py`: `uv run pytest experiments/exp141/test.py`; optional pytest arguments.
