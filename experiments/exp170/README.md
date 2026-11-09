# exp170 — COBA with conductance-matched Poisson inhibition

Run one frozen COBA trial with independent private Poisson input to each E neuron.
Rates and constant event amplitudes are fitted once to PING's per-E conductance
mean and variance. Original sensory/readout weights and the fitted private
weights stay frozen. There is no training.

The versioned `poisson_moment_matched_v1` method draws independent Poisson counts
per 0.1-ms bin, including coincident events. It uses the exact stationary moments
of the discrete 6-ms GABA filter with one-step delay. For decay factor
`a = exp(-dt/tau)`, amplitude is `(1+a)*variance/mean`; rate in Hz is
`mean**2*(1-a)/(dt_seconds*(1+a)*variance)`.

Keep 60 ms of inhibition-only prehistory; carry only private conductance and
pending events into fresh neurons at t=0. Match stationary expectations, not
individual random trials: save and report achieved trial mean/SD errors without
conditioning counts, rejecting draws, or rescaling sampled traces. Artificial
rates need not equal PING I rates. Private streams also remove shared
fluctuations, so this single-image COBA/PING comparison does not isolate gamma.
Expected retention remains below 40 MB. Historical fixed-count runs retain their
original generator and validation semantics.

## Run

From the repository root, run these stages in order.
Compute runs simulations and requires explicit authorisation.

```sh
uv run python experiments/exp170/compute.py
uv run python experiments/exp170/compute.py --record-traces COMPUTE_ID
uv run python experiments/exp170/analyse.py --source COMPUTE_ID
uv run python experiments/exp170/present.py --source ANALYSE_ID --traces TRACE_COMPUTE_ID
```

Replace `COMPUTE_ID` with the first compute run ID, `TRACE_COMPUTE_ID` with
the trace replay run ID, and `ANALYSE_ID` with the analysis run ID.

To run automated checks:

```sh
uv run pytest experiments/exp170/test.py -q
```

## Folder guide

1. `recipe.py` — configuration and model definition.
2. `compute.py` — run simulations and save evidence.
3. `analyse.py` — calculate measurements from saved evidence.
4. `present.py` — generate figures and report assets.
5. `custom/` — experiment-specific models, helpers and execution tools.
6. `report/` — report, appendix and supporting components.
7. `test.py` — automated checks.
