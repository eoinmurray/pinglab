# exp134 — Leak and capacitance modulation of PING dynamics

Compare paired leak-conductance and membrane-capacitance interventions in one frozen
trained PING circuit, measuring frequency, firing, participation and rhythm contrast
while reusing the same images and encoded spike trains.

## Run

From the repository root, run these stages in order.
Compute runs simulations and requires explicit authorisation.

```sh
uv run python experiments/exp134/compute.py --source TRAINING_RUN --image-source IMAGE_RUN
uv run python experiments/exp134/analyse.py --source COMPUTE_ID
uv run python experiments/exp134/present.py --source ANALYSE_ID
```

Replace `COMPUTE_ID` and `ANALYSE_ID` with the completed run IDs
from the preceding stages.

`TRAINING_RUN` is a completed exp022 training-bank compute run.

`IMAGE_RUN` is a completed exp123 image-bank compute run.

To run automated checks:

```sh
uv run pytest experiments/exp134/test.py -q
```

## Folder guide

1. `recipe.py` — configuration and model definition.
2. `compute.py` — run simulations and save evidence.
3. `analyse.py` — calculate measurements from saved evidence.
4. `present.py` — generate figures and report assets.
5. `custom/` — experiment-specific models, helpers and execution tools.
6. `report/` — report, appendix and supporting components.
7. `test.py` — automated checks.
