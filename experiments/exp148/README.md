# exp148 — External-I-drive transition

Test whether reducing external excitation of inhibitory neurons produces population
oscillations in a shared-input conductance network. Compare the spiking trajectory with
an optional matched mean-field model.

## Run

From the repository root, run these stages in order.
Compute runs simulations and requires explicit authorisation.

```sh
uv run python experiments/exp148/compute.py
uv run python experiments/exp148/analyse.py --source COMPUTE_ID
uv run python experiments/exp148/present.py --source ANALYSE_ID
```

Replace `COMPUTE_ID` and `ANALYSE_ID` with the completed run IDs
from the preceding stages.

To include the matched mean-field model, run compute again with
`--mean-field-source COMPUTE_ID`, then add `--mean-field MODEL_COMPUTE_ID`
to analysis using the new model run ID.

To run automated checks:

```sh
uv run pytest experiments/exp148/test.py -q
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
