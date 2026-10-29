# exp127 — Introduction to spiking neural networks

Explain current-based integration and train a six-weight synthetic classifier.
Recorded before/after responses and a separate fixed-batch teaching trace
illustrate inference and surrogate-gradient learning.

Run stages independently from the repository root with `uv run python`.

1. `compute.py`: optional `--run-id ID`; no required arguments.
2. `analyse.py`: required `--source COMPUTE_RUN`; no optional arguments.
3. `present.py`: required `--source ANALYSIS_RUN`; no optional arguments.
4. `test.py`: `uv run pytest experiments/exp127/test.py`; optional pytest arguments.

Regenerate only network diagrams while preserving an explicit presentation's
scientific figures and data: `uv run python experiments/exp127/present.py
--redraw-from PRESENT_ID [--run-id ID]`.
