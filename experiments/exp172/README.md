# exp172 — Spatial Gaussian Selective Relay

Test how a uniform rate cue selects between two spatial Gaussian input patterns in an
A/B → C spiking network.

## Run

From the repository root, run these stages in order.
Compute runs simulations and requires explicit authorisation.

```sh
uv run python experiments/exp172/compute.py
uv run python experiments/exp172/analyse.py --source COMPUTE_ID
uv run python experiments/exp172/present.py --source ANALYSE_ID
```

Replace `COMPUTE_ID` and `ANALYSE_ID` with the completed run IDs
from the preceding stages.

To run automated checks:

```sh
uv run pytest experiments/exp172/test.py -q
```

## Folder guide

1. `recipe.py` — configuration and model definition.
2. `compute.py` — run simulations and save evidence.
3. `analyse.py` — calculate measurements from saved evidence.
4. `present.py` — generate figures and report assets.
5. `report/` — report, appendix and supporting components.
6. `test.py` — automated checks.

## Cycle decoder

C selection uses one spatial fit per trough-to-trough cycle, containing a whole excitatory volley. Population-rate minima between detected peaks define the boundaries; flat minima use the point nearest the peak midpoint. This avoids artificial changes from combining the unequal spatial halves of adjacent volleys. Peak detection, spatial templates, strength threshold and three-cycle latency rule are unchanged.

Analysis validates the nine trials declared by the current recipe. Additional historical trials stay in their immutable source run and are explicitly listed as excluded in the new analysis. Regenerate analysis and presentation from the pinned compute run to update report metrics without simulation.
