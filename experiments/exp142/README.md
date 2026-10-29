# exp142 — Continuous-time scikit digits

Compare frozen COBA-LIF and PING digit classifiers at four discrete timesteps
and in continuous time. Replay one exp141 training image with identical saved
input events, and measure trajectory errors and spike-count classifications.

## CLI

1. `compute.py`: `--source <exp141-compute-id>` (required; must match the recipe's
   run and digest pin), `--input-rate-hz <Hz>` (optional; default 100),
   `--run-id <id>` (optional). Runs locally on CPU without training.
2. `analyse.py`: `--source <exp142-compute-id>` (required), `--run-id <id>` (optional).
3. `present.py`: `--source <exp142-analyse-id>` (required), `--run-id <id>` (optional).
4. `test.py`: run with `uv run pytest -q experiments/exp142/test.py`;
   no experiment arguments.

Regenerate only network diagrams while preserving an explicit presentation's
scientific figures and data: `uv run python experiments/exp142/present.py
--redraw-from PRESENT_ID [--run-id ID]`.
