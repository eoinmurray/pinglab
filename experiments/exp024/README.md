# exp024 — Accuracy Plateaus While Firing Rate Rises

Audit whether validation accuracy and population firing rates settle together in the six
unregularised COBA and PING learning histories.

## Run

From the repository root, run these stages in order.
This experiment analyses saved evidence without running a compute stage.

```sh
uv run python experiments/exp024/analyse.py --source TRAINING_RUN
uv run python experiments/exp024/present.py --source ANALYSE_ID
```

Replace `ANALYSE_ID` with the completed analysis run ID.

`TRAINING_RUN` is a completed exp022 training-bank compute run.

To run automated checks:

```sh
uv run pytest experiments/exp024/test.py -q
```

## Folder guide

1. `recipe.py` — configuration and model definition.
2. `analyse.py` — calculate measurements from saved evidence.
3. `present.py` — generate figures and report assets.
4. `custom/` — experiment-specific models, helpers and execution tools.
5. `report/` — report, appendix and supporting components.
6. `test.py` — automated checks.
