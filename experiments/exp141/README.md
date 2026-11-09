# exp141 — Scikit-learn digits

Test whether fixed inhibitory feedback improves handwritten-digit classification by
comparing feed-forward COBA-LIF and PING networks on scikit-learn digits. Compare their
learning curves, held-out accuracy and spike activity.

## Run

From the repository root, run these stages in order.
Compute trains models and requires explicit authorisation.

```sh
uv run python experiments/exp141/compute.py
uv run python experiments/exp141/analyse.py --source COMPUTE_ID
uv run python experiments/exp141/present.py --source ANALYSE_ID
```

Replace `COMPUTE_ID` and `ANALYSE_ID` with the completed run IDs
from the preceding stages.

To run automated checks:

```sh
uv run pytest experiments/exp141/test.py -q
```

## Folder guide

1. `recipe.py` — configuration and model definition.
2. `dataset.py` — input preparation and encoding.
3. `compute.py` — train models and save evidence.
4. `analyse.py` — calculate measurements from saved evidence.
5. `present.py` — generate figures and report assets.
6. `custom/` — experiment-specific models, helpers and execution tools.
7. `report/` — report, appendix and supporting components.
8. `test.py` — automated checks.
