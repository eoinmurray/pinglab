# exp099 — Gamma versus input drive

Test how a private excitatory afferent pulse changes bursting in a modified
Susin–Destexhe conductance-LIF circuit. The report describes one selected seed.

Run each stage independently from the repository root with `uv run python`.

1. `compute.py`: optional `--run-id ID`; no required arguments.
2. `analyse.py`: required `--source COMPUTE_RUN`; optional `--run-id ID`.
3. `present.py`: required `--source ANALYSIS_RUN`; optional `--run-id ID`.
4. `test.py`: `uv run pytest experiments/exp099/test.py`; optional pytest arguments.

Regenerate saved-bundle diagrams while preserving an explicit presentation:
`uv run python experiments/exp099/present.py --redraw-from PRESENT_ID [--run-id ID]`.
