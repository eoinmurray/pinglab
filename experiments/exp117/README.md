# exp117

Locate and numerically characterize oscillatory onset in an independent four-variable mean-field closure, including an inhibitory-decay sweep.

## CLI

Run each stage independently from the repository root with `uv run python`.

1. `compute.py`: `--run-id` (optional).
2. `analyse.py`: `--source` (required), `--run-id` (optional).
3. `present.py`: `--source` (required), `--run-id` (optional).
4. `test.py`: `uv run pytest experiments/exp117/test.py`; optional pytest arguments.
