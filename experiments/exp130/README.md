# exp130 — Theta-nested gamma

Compare theta-modulated and constant private afferent input in the same
conductance-based E/I circuit, with paired weights and random streams.

Run stages independently from the repository root with `uv run python`.

1. `compute.py`: optional `--run-id ID`; no required arguments.
2. `analyse.py`: required `--source COMPUTE_RUN`; no optional arguments.
3. `present.py`: required `--source ANALYSIS_RUN`; no optional arguments.
4. `test.py`: `uv run pytest experiments/exp130/test.py`; optional pytest arguments.
