# exp025 — accuracy–rate frontier

`exp025` reuses an explicitly selected, completed `exp022` model-bank compute
run. It evaluates the retained COBA and PING networks, measures the resulting
accuracy/rate behaviour, and renders the article inputs.

## Independent stages

```sh
uv run python experiments/exp025/compute.py --source <exp022-compute-run>
uv run python experiments/exp025/analyse.py --source <exp025-compute-run>
uv run python experiments/exp025/present.py --source <exp025-analyse-run>
```

Each command creates exactly one immutable `pingstore.run/v4` run. Every source
is explicit and checksum-pinned. No stage selects a latest run, launches another
stage, publishes, or writes `.artifacts/`.

The bank supplies 36 `TR-02` frontier cells and 12 `TR-07` low-input-weight
controls. Their final-epoch checkpoints and training histories are validated
before inference or analysis and their complete ancestry is checked again before
stage completion.

Production compute executes 98 inference jobs:

- 36 official-test frontier evaluations;
- 12 representative-seed PFG evaluations;
- 48 input-weight-scale evaluations; and
- two digit-0/sample-0, 400 ms raster snapshots.

Quantitative evaluations use all 1,000 selected official MNIST test images.
`PINGLAB_SMOKE=1` uses 100 images, three scale factors, and seed 42 for the
low-input-weight history summary, while still requiring the complete trained
bank.

## Stage ownership

- Compute retains raw metrics, PING population traces and sparse E/I events,
  scale-sweep sample-wise E rates, and E/I illustrative recordings.
- Analyse owns frontier means and SEMs, Welch frequency, inhibitory-cycle
  participation, sample-wise rate penalties, history summaries, scale-crossing
  midpoint, and raster coordinates.
- Present renders five figures in their required vector or raster formats plus
  `numbers.json`. It does not remeasure the analysis.

Compute execution configuration, commands, environment, checkpoint pins and
timing belong in `run.json` or discarded writer scratch space. New compute runs
therefore do not export a parallel metadata envelope. The historical
`exp025-r001-compute` payload retains its original `evidence.json`; readers
validate that immutable legacy v4 file when it is present but do not require it
from new runs.

## Scientific scope

COBA PFG rows contain accuracy and firing rates only; frequency and cycle
participation remain undefined. PING frequency uses Welch spectra of retained E
population traces, and participation counts active E cell-cycle pairs using
inhibitory burst peaks. The participation-frequency product is an approximation.
The input-scale crossing is the midpoint of the first sampled pair crossing
0.05 Hz I rate, not a fitted bifurcation.

The retained bank uses gradient damping 1 for COBA and 1000 for PING, which
limits causal attribution. Low-input-weight curves are validation histories;
frontier endpoints are final-epoch official-test evaluations.

## Retained historical lineage

The original Gold-2 import was an explicit offline historical operation. It did
not train, simulate, publish, or modify the source archive. The current retained
lineage is `exp025-r001-compute` → `exp025-r002-analyse` →
`exp025-r007-present`, pinned to `exp022-r001-compute`. Its source identities,
operation history and checksums remain in the run records and README histories.

Existing immutable runs and external archives are not rewritten by source-code
cleanup. The retained `r007` presentation still contains its historical
standalone raster figures; future presentations omit those unused duplicates
because the article and synthesis consume the rasters through
`results_compound`.
