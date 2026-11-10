# exp022 — MNIST

Train the seven MNIST classifier families, retain their validation histories and
checkpoints, and measure fixed-image activity for downstream comparisons.

## Run

From the repository root, run these stages in order.
Compute trains models and requires explicit authorisation.

```sh
uv run python experiments/exp022/compute.py
uv run python experiments/exp022/analyse.py --source COMPUTE_ID
uv run python experiments/exp022/present.py --source ANALYSE_ID
```

Replace `COMPUTE_ID` and `ANALYSE_ID` with the completed run IDs
from the preceding stages.

The default compute command trains the full bank sequentially. `SNNLAB_DEVICE`
selects the execution device.

To run automated checks:

```sh
uv run pytest experiments/exp022/test.py -q
```

To prepare compact fine-tuning inputs without simulation, run analysis with
`--source COMPUTE_ID --export-cell CELL`, repeating `--export-cell` for each
required cell. This creates a new analysis run containing only each cell’s
configuration, baseline metrics and verified epoch-50 final checkpoint, linked to
the source bank.

## Folder guide

1. `recipe.py` — configuration and model definition.
2. `compute.py` — train models and save evidence.
3. `analyse.py` — calculate measurements from saved evidence.
4. `present.py` — generate figures and report assets.
5. `custom/` — experiment-specific models, helpers and execution tools.
6. `report/` — report, appendix and supporting components.
7. `test.py` — automated checks.
