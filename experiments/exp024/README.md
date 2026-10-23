# exp024

Audit whether validation accuracy and population firing rates settle together in the six unregularised COBA and PING learning histories.

## CLI

Run each stage independently from the repository root with `uv run python`.

1. `analyse.py`: `--source` (required), `--run-id` (optional).
2. `present.py`: `--source` (required), `--run-id` (optional).
3. `test.py`: `uv run pytest experiments/exp024/test.py`; optional pytest arguments.
