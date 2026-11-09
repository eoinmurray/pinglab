# exp153 — Mean-field coupling

Map phase locking between two coupled excitatory–inhibitory circuits using a delayed
rate–conductance model, then inspect coupling onset and receiver susceptibility.

## Run

From the repository root, run these stages in order.
Compute runs numerical integration and requires explicit authorisation.

```sh
uv run python experiments/exp153/compute.py
uv run python experiments/exp153/analyse.py --source COMPUTE_ID
uv run python experiments/exp153/present.py --source ANALYSE_ID
```

Replace `COMPUTE_ID` and `ANALYSE_ID` with the completed run IDs
from the preceding stages.

To add the susceptibility figure, use the same compute run and the presentation
ID returned above:

```sh
uv run python experiments/exp153/analyse.py --susceptibility-only --source COMPUTE_ID
uv run python experiments/exp153/present.py --fig3-only --source SUSCEPTIBILITY_ANALYSE_ID --preserve-source PRESENT_ID
```

Replace `SUSCEPTIBILITY_ANALYSE_ID` with the new susceptibility analysis run ID
and `PRESENT_ID` with the preceding presentation run ID.

To run automated checks:

```sh
uv run pytest experiments/exp153/test.py -q
```

## Folder guide

1. `recipe.py` — configuration and model definition.
2. `compute.py` — run computations and save evidence.
3. `analyse.py` — calculate measurements from saved evidence.
4. `present.py` — generate figures and report assets.
5. `custom/` — experiment-specific models, helpers and execution tools.
6. `report/` — report, appendix and supporting components.
7. `test.py` — automated checks.
