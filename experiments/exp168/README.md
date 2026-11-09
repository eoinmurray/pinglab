# exp168 — AI and PING with and without EE/II recurrence

Compare asynchronous and rhythmic activity with and without excitatory and inhibitory
self-recurrence. The conditions use selected single-seed network and input settings.

## Run

From the repository root, run these stages in order.
Compute runs simulations and requires explicit authorisation.

```sh
uv run python experiments/exp168/compute.py simulate
uv run python experiments/exp168/analyse.py
uv run python experiments/exp168/present.py --source ANALYSE_ID
```

Replace `ANALYSE_ID` with the completed analysis run ID.

Analysis currently reads the fixed `exp168-r029-compute` recurrence-on run and the
pinned recurrence-off run in `recipe.py`. It accepts no `--source` argument, so a new
simulation is not automatically analysed.

To run automated checks:

```sh
uv run pytest experiments/exp168/test.py -q
```

## Folder guide

1. `recipe.py` — configuration and model definition.
2. `dataset.py` — input preparation and encoding.
3. `compute.py` — run simulations and save evidence.
4. `analyse.py` — calculate measurements from saved evidence.
5. `present.py` — generate figures and report assets.
6. `report/` — report, appendix and supporting components.
7. `test.py` — automated checks.
