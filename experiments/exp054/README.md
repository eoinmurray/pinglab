# exp054

Measure autocorrelation lobe–trough contrast across reciprocal-coupling strengths, with private- and shared-input null controls.

## CLI

Run each stage independently from the repository root with `uv run python`.

1. `compute.py`: `--run-id` (optional).
2. `analyse.py`: `--source` (required), `--run-id` (optional).
3. `present.py`: `--source` (required), `--run-id` (optional).
4. `test.py`: `uv run pytest experiments/exp054/test.py`; optional pytest arguments.

Optional environment: `PINGLAB_SMOKE=1` selects the bounded smoke profile; `SNNLAB_DEVICE` (or the legacy `PINGLAB_DEVICE`) selects the execution device.

Regenerate the documented-model illustrations without running a scientific stage:
`uv run python experiments/exp054/present.py --draw-schematics`.
These authored SVGs do not reconstruct missing historical bundles.
