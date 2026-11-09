# exp123 — Output evidence against time and gamma cycles

Measure decision timing and spike cost across a paired inhibitory-decay sweep in one
frozen trained PING network. All report figures use the same balanced sample.

## Run

From the repository root, run these stages in order.
Compute runs simulations and requires explicit authorisation.

```sh
uv run python experiments/exp123/compute.py --source TRAINING_RUN
uv run python experiments/exp123/analyse.py --source COMPUTE_ID
uv run python experiments/exp123/present.py --source ANALYSE_ID
```

Replace `COMPUTE_ID` and `ANALYSE_ID` with the completed run IDs
from the preceding stages.

`TRAINING_RUN` is a completed exp022 training-bank compute run.

The official MNIST test set must already be available under
`.pingstore/datasets/MNIST/`, or add `--image-source IMAGE_RUN` using a validated exp123
image bank.

To run automated checks:

```sh
uv run pytest experiments/exp123/test.py -q
```

## Folder guide

1. `recipe.py` — configuration and model definition.
2. `compute.py` — run simulations and save evidence.
3. `analyse.py` — calculate measurements from saved evidence.
4. `present.py` — generate figures and report assets.
5. `custom/` — experiment-specific models, helpers and execution tools.
6. `report/` — report, appendix and supporting components.
7. `test.py` — automated checks.
