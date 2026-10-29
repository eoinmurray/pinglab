# exp110

Synthesize explicitly selected coupling, mean-field, cycle-participation and robustness evidence into manuscript figures. This experiment performs presentation only.

## CLI

Run each stage independently from the repository root with `uv run python`.

1. `present.py`: `--source` (required), `--theory-source` (required), `--run-id` (optional).
2. `test.py`: `uv run pytest experiments/exp110/test.py`; optional pytest arguments.

Regenerate the documented-model illustrations without running a scientific stage:
`uv run python experiments/exp110/present.py --draw-schematics`.
These authored SVGs do not reconstruct missing historical bundles.
