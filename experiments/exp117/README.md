# exp117

Locate and numerically characterize oscillatory onset in a four-variable SNNLab mean-field closure, including an inhibitory-decay sweep. SNNLab owns the gain, symbolic equations,
equilibrium search, stability crossings and integration. Native model documents
and the authored circuit are exported; article payloads retain spikes/ms rates.

See the [pedagogical mean-field note](../../notes/mean-field-model.mdx) for the
derivation, exp148 extension, and general shared/private-input formulation.

## CLI

Run each stage independently from the repository root with `uv run python`.

1. `compute.py`: `--run-id` (optional).
2. `analyse.py`: `--source` (required), `--run-id` (optional).
3. `present.py`: `--source` (required), `--run-id` (optional).
4. `test.py`: `uv run pytest experiments/exp117/test.py`; optional pytest arguments.

Regenerate the documented-model illustrations without running a scientific stage:
`uv run python experiments/exp117/present.py --draw-schematics`.
These authored SVGs do not reconstruct missing historical bundles.
