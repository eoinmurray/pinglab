# exp146 — NMT Single-Neuron Compatibility

Implement the Neuromorphic Toolbox proposal’s single sodium/potassium neuron in SNNLab
and compare it with an independent Euler implementation. Toolbox reference traces are
not available.

## Run

From the repository root, run these stages in order.
Compute runs simulations and requires explicit authorisation.

```sh
uv run python experiments/exp146/compute.py
uv run python experiments/exp146/analyse.py --source COMPUTE_ID
uv run python experiments/exp146/present.py --source ANALYSE_ID
```

Replace `COMPUTE_ID` and `ANALYSE_ID` with the completed run IDs
from the preceding stages.

To run automated checks:

```sh
uv run pytest experiments/exp146/test.py -q
```

## Folder guide

1. `recipe.py` — configuration and model definition.
2. `compute.py` — run simulations and save evidence.
3. `analyse.py` — calculate measurements from saved evidence.
4. `present.py` — generate figures and report assets.
5. `custom/` — experiment-specific models, helpers and execution tools.
6. `report/` — report, appendix and supporting components.
7. `test.py` — automated checks.
