# Exp022: shared gamma-gated sparsity model bank

Exp022 owns the 102-cell COBA/PING training registry reused by the
gamma-gated-sparsity collection. Operational execution follows the repository
[Experiment Runner Guide](../README.md) and requires `pingstore.run/v4`.

The current retained bank descends from the repaired Gold-2 campaign: 60 cells
were inherited and 42 were retrained with the corrected firing-rate penalty.
The complete repaired bank is a standalone source, not an overlay to merge with
the earlier base bank. See [ANCESTRY.md](ANCESTRY.md) for the verified historical
lineage. Historical v2/v3 runs remain non-operational evidence.

## Code layout

- `recipe.py` owns the scientific registry and read-only model-bank interface.
- `compute.py` owns direct training, RunPod dispatch, campaign capture, v4 bank
  import, and explicitly requested retained diagnostic simulations.
- `campaign.py` owns manifest validation, worker ownership, retry bookkeeping,
  and aggregation checks.
- `analyse.py` measures a completed compute run without executing new science.
- `present.py` renders a completed analysis and can explicitly carry verified
  historical raster images when the original raw probes were not retained.
- `slurm/` contains the Wilkes environment checks, submission scripts, and the
  [operator runbook](slurm/README.md). The gamma-gated-sparsity collection reuses
  its environment and array helpers.

`experiments.exp022` exports the recipe for downstream consumers. Execution uses
the explicit stage modules rather than the package root.

## Operational workflow

Each command creates or completes exactly one v4 stage. Downstream stages never
launch upstream work or publish output automatically.

```sh
# Train a new bank locally.
uv run python -m experiments.exp022.compute

# Analyse an explicit completed compute run.
uv run python experiments/exp022/analyse.py --source <compute-run-id>

# Present an explicit completed analysis run.
uv run python experiments/exp022/present.py --source <analyse-run-id>
```

RunPod, Slurm campaign, import, recovery, and diagnostic modes are explicit
compute operations. Use `compute.py --help` and the Slurm runbook for their
arguments. Campaign aggregation completes only its preallocated compute run.
It does not analyse, present, materialize, or publish.

Completed runs contain exactly `run.json`, `README.md`, and `export/`. The 102
model cells are direct scientific-unit directories in compute exports. Analysis
exports contain measurements and plot-ready arrays. Presentation exports are
flat publication inputs. All readers validate the complete v4 source and its
payload digest before use.

## Firing-rate penalty calibration

The activity penalty uses the sample-wise, population-normalized hertz
objective. The earlier strength of `0.001` came from a neuron-summed spike-count
objective and became 40.96 times weaker after normalization over 1,024
excitatory neurons and 0.2 seconds.

A bounded seed-42 calibration tested strengths `0.004`, `0.016`, `0.041`, and
`0.1` for both COBA and PING. The measured elbow, `0.041`, became the production
value and is frozen in `recipe.py`. The retired pilot runner is preserved in Git
history; it is not part of current experiment execution. The present repository
does not claim that its raw pilot outputs are retained as an operational run.

## Historical evidence and publication

The ancestry record and Pingstore run histories preserve the distinction
between local import and historical Slurm execution. Source-neutral run IDs do
not encode execution origin. Earlier naming and format migrations are historical
records, not compatibility paths; see the repository
[source-neutral ID record](../../tools/pingstore/SOURCE_NEUTRAL_IDS.md).

Preview may use only a validated present run. Materialization and publication
require separate authorization and must select the complete flat export of that
present run. Compute and analyse runs cannot be published directly.
