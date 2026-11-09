# exp124 — SHD: paired COBA and PING training

Compare two-layer COBA and PING classifiers on the Spiking Heidelberg Digits dataset.
Measure validation learning curves and matched spike responses; the official test set is
unused.

## Run

From the repository root, run these stages in order.
Compute trains models and requires explicit authorisation.

```sh
uv run python experiments/exp124/compute.py
uv run python experiments/exp124/analyse.py --source COMPUTE_ID
uv run python experiments/exp124/present.py --source ANALYSE_ID
```

Replace `COMPUTE_ID` and `ANALYSE_ID` with the completed run IDs
from the preceding stages.

The default protocol trains each model for 50 epochs. Add `--protocol local-check` for a
small diagnostic run. `SHD_CACHE` selects the dataset archive cache; `SNNLAB_DEVICE`
selects the device.

To run automated checks:

```sh
uv run pytest experiments/exp124/test.py -q
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
