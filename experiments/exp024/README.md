# exp024

Audit whether validation accuracy and population firing rates settle together in the six unregularised COBA and PING learning histories.

## CLI

Run each stage independently from the repository root with `uv run python`.

1. `analyse.py`: `--source` (required), `--run-id` (optional).
2. `present.py`: `--source` (required), `--run-id` (optional).
3. `test.py`: `uv run pytest experiments/exp024/test.py`; optional pytest arguments.

Regenerate the documented-model illustrations without running a scientific stage:
`uv run python experiments/exp024/present.py --draw-schematics`.
These authored SVGs do not reconstruct missing historical bundles.
