# exp080 — Decoder Accuracy Improves with Input Rate

Calibrate input rates by training nonlinear decoders on finite-window, nonspiking
membrane features and evaluating held-out MNIST accuracy.

## Run

From the repository root, run these stages in order.
Compute trains models and requires explicit authorisation.

```sh
uv run python experiments/exp080/compute.py
uv run python experiments/exp080/analyse.py --source COMPUTE_ID
uv run python experiments/exp080/present.py --source ANALYSE_ID
```

Replace `COMPUTE_ID` and `ANALYSE_ID` with the completed run IDs
from the preceding stages.

To run automated checks:

```sh
uv run pytest experiments/exp080/test.py -q
```

## Folder guide

1. `recipe.py` — configuration and model definition.
2. `compute.py` — train models and save evidence.
3. `analyse.py` — calculate measurements from saved evidence.
4. `present.py` — generate figures and report assets.
5. `custom/` — experiment-specific models, helpers and execution tools.
6. `report/` — report, appendix and supporting components.
7. `test.py` — automated checks.
