# exp099 — Gamma versus input drive

Test how a private excitatory afferent pulse changes bursting in a modified
Susin–Destexhe conductance-LIF circuit. The report describes one selected seed.

## Run

From the repository root, run these stages in order.
Compute runs simulations and requires explicit authorisation.

```sh
uv run python experiments/exp099/compute.py
uv run python experiments/exp099/analyse.py --source COMPUTE_ID
uv run python experiments/exp099/present.py --source ANALYSE_ID
```

Replace `COMPUTE_ID` and `ANALYSE_ID` with the completed run IDs
from the preceding stages.

To run automated checks:

```sh
uv run pytest experiments/exp099/test.py -q
```

## Folder guide

1. `recipe.py` — configuration and model definition.
2. `dataset.py` — input preparation and encoding.
3. `compute.py` — run simulations and save evidence.
4. `analyse.py` — calculate measurements from saved evidence.
5. `present.py` — generate figures and report assets.
6. `custom/` — experiment-specific models, helpers and execution tools.
7. `report/` — report, appendix and supporting components.
8. `test.py` — automated checks.
