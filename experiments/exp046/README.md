# exp046 — One Spike per Gamma Cycle

Measure excitatory spikes per inhibitory-burst cycle in eighteen PING networks. Compare
pooled neuron–cycle distributions with equally weighted network distributions.

## Run

From the repository root, run these stages in order.
Compute runs simulations and requires explicit authorisation.

```sh
uv run python experiments/exp046/compute.py --source TRAINING_RUN
uv run python experiments/exp046/analyse.py --source COMPUTE_ID --frequency-source FREQUENCY_RUN
uv run python experiments/exp046/present.py --source ANALYSE_ID
```

Replace `COMPUTE_ID` and `ANALYSE_ID` with the completed run IDs
from the preceding stages.

`TRAINING_RUN` is a completed exp022 training-bank compute run.

`FREQUENCY_RUN` is a completed exp041 analysis run.

To run automated checks:

```sh
uv run pytest experiments/exp046/test.py -q
```

## Folder guide

1. `recipe.py` — configuration and model definition.
2. `compute.py` — run simulations and save evidence.
3. `analyse.py` — calculate measurements from saved evidence.
4. `present.py` — generate figures and report assets.
5. `custom/` — experiment-specific models, helpers and execution tools.
6. `report/` — report, appendix and supporting components.
7. `test.py` — automated checks.
