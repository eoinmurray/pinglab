# exp081

Compare simulated finite-window pixel-voltage features with a stationary, linearized filtering approximation across rates and event strengths.

## CLI

Run each stage independently from the repository root with `uv run python`.

1. `compute.py`: `--run-id` (optional).
2. `analyse.py`: `--source` (required), `--run-id` (optional).
3. `present.py`: `--source` (required), `--run-id` (optional).
4. `test.py`: `uv run pytest experiments/exp081/test.py`; optional pytest arguments.

Optional environment: `PINGLAB_SMOKE=1` selects the bounded smoke profile.

Regenerate the documented-model illustrations without running a scientific stage:
`uv run python experiments/exp081/present.py --draw-schematics`.
These authored SVGs do not reconstruct missing historical bundles.
