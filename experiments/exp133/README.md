# exp133 — Lowet (2015): Coupled Rhythms

Map finite-time phase locking between two reciprocally coupled gamma circuits,
then show coupling onset at one selected map point. The report compares the map
qualitatively with Lowet et al. (2015).

## CLI

1. `compute.py`: `--protocol grid|sender-receiver` (optional; default `grid`),
   `--parameter-source <compute-run-id>` (optional for the grid; required for
   sender–receiver), `--run-id <id>` (optional). Reproduce the report's expanded
   grid using `--parameter-source exp133-r025-compute`; its onset diagnostic
   uses `--protocol sender-receiver --parameter-source exp133-r028-compute`.
2. `analyse.py`: `--source <compute-run-id>` (required), `--run-id <id>` (optional).
3. `present.py`: `--source <grid-analyse-run-id>` and
   `--sender-receiver <onset-analyse-run-id>` (both required),
   `--run-id <id>` (optional).
4. `test.py`: `uv run pytest experiments/exp133/test.py`; optional pytest arguments.
