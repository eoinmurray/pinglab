# exp171 — Fine-tuning COBA with artificial inhibition across activity ceilings

Fine-tune COBA digit classifiers with private, image-conditioned artificial inhibition
calibrated from PING. Compare learning and firing across activity ceilings, including an
unpenalised control.

The default `poisson_moment_matched_v1` mechanism matches exp170's Poisson
construction, implemented within exp171. Each image's frozen PING pass supplies
per-E conductance mean and variance. For decay `a = exp(-dt/tau)`, set the fixed
event amplitude to `(1+a)*variance/mean` and rate in Hz to
`mean**2*(1-a)/(dt_seconds*(1+a)*variance)`. This exactly targets the stationary
moments of the discrete GABA filter. Silent teacher channels receive zero input;
constant positive targets fail explicitly.

Generate independent Poisson counts per timestep, retaining coincident events.
Rates and amplitudes stay fixed; training redraws events each epoch and validation
reuses three draws. Keep 60-ms inhibition-only prehistory and fresh neurons at
image onset. Only sensory→E and E→readout weights train. Artificial event rates
are fitted, not constrained to measured PING I rates; those remain diagnostics.
Finite-trial conductance moments fluctuate and are not forced to match each draw.

All six activity ceilings, including penalty off, use this method. Old fixed-count
and teacher-count calibration caches are rejected; historical evidence remains
readable. All six need fresh runs; this code change does not launch training.
Replay retains integer counts rather than binary flags. No per-epoch/full-dataset
event cache is retained; expected storage remains below 350 MB per cell.

## Run

From the repository root, run these stages in order.
Compute trains models and requires explicit authorisation.

```sh
uv run python experiments/exp171/compute.py
uv run python experiments/exp171/analyse.py --source COMPUTE_ID
uv run python experiments/exp171/present.py --source ANALYSE_ID
```

Replace `COMPUTE_ID` and `ANALYSE_ID` with the completed run IDs
from the preceding stages.

For all six activity ceilings, add `--sweep` to compute. Analyse the six run IDs
together by repeating `--source COMPUTE_ID` once per condition, then present that
combined analysis. `SNNLAB_DEVICE=cuda` selects CUDA.

For compact exp022 input exports, pass both `--source ANALYSE_ID` and
`--source-digest sha256:DIGEST` to a single-condition compute command. The export
must contain the matching COBA and PING configurations, checkpoints and baseline
metrics. Sweep exports must pin the same original compute bank. Original full-bank defaults remain
supported. On CSD3, deploy the compact run with `--input ANALYSE_ID
--direct-inputs-only`; ancestor references are preserved without copying their
payloads.

`--profile` records synchronized sensory and artificial-input encoding timings
in `encoding-profile.json`, plus peak host memory. This diagnostic changes CUDA
overlap; native phase timings and GPU memory remain in `native-metrics.json`.

To run automated checks:

```sh
uv run pytest experiments/exp171/test.py -q
```

## Folder guide

1. `recipe.py` — configuration and model definition.
2. `compute.py` — train models and save evidence.
3. `analyse.py` — calculate measurements from saved evidence.
4. `present.py` — generate figures and report assets.
5. `report/` — report, appendix and supporting components.
6. `test.py` — automated checks.
