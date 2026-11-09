# exp145 — Shared-input conductance AI probe

Measure irregular firing and population synchrony in a Renart-inspired conductance
network with shared external input. The default run measures one second after a 50 ms
burn-in.

## Run

From the repository root, run these stages in order.
Compute runs simulations and requires explicit authorisation.

```sh
uv run python experiments/exp145/compute.py
uv run python experiments/exp145/analyse.py --source COMPUTE_ID
uv run python experiments/exp145/present.py --source ANALYSE_ID
```

Replace `COMPUTE_ID` and `ANALYSE_ID` with the completed run IDs
from the preceding stages.

To run automated checks:

```sh
uv run pytest experiments/exp145/test.py -q
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
