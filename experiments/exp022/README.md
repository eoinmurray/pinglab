# Exp022: shared gamma-gated sparsity model bank

Exp022 owns the 102-cell COBA/PING training registry reused by the
gamma-gated-sparsity collection. Operational execution follows the repository
[Experiment Runner Guide](../README.md) and requires `pingstore.run/v4`.

The earlier repaired Gold-2 bank contains 60 inherited cells and 42 cells
retrained with the corrected firing-rate penalty.
The complete repaired bank is a standalone source, not an overlay to merge with
the earlier base bank. See [ANCESTRY.md](ANCESTRY.md) for the verified historical
lineage. Historical v2/v3 runs remain non-operational evidence.

## Adopted refractory bank — 2026-09-09

The current replacement bank is `exp022-r007-compute`: **102 models**, consisting
of **90 unchanged reused models and twelve new timestep models**. Its accepted
analysis and presentation are `exp022-r010-analyse` and `exp022-r014-present`.
The latter replaces `exp022-r013-present` only to remove internal run-ID stamps
from the seven learning curves; scientific values and all 35 raster PNGs are
unchanged.
All 34 seed-42 final-epoch diagnostic recordings were newly generated, including
those of reused models. The old `exp022-r001-compute` remains the explicit source
for unchanged downstream measurements; those consumers were not repinned.

The five timestep conditions are **0.05, 0.1, 0.2, 0.3 and 0.6 ms**, with exact
**1.2/0.6-ms E/I refractory holds**. Their counters are 24/12, 12/6, 6/3, 4/2 and
2/1. Nominal 200-ms presentations have 4,000/2,000/1,000/666/333 steps and last
200/200/200/199.8/199.8 ms. Rates and diagnostic spectra use realised durations.
The source bank and all historical definitions remain unchanged. PLAN.md records
the execution-equivalence checks, HPC training and complete source digests.

## Code layout

- `recipe.py` owns the scientific registry and read-only model-bank interface.
- `compute.py` owns direct training, RunPod dispatch, campaign capture, v4 bank
  import, and explicitly requested retained diagnostic simulations.
- `campaign.py` owns manifest validation, worker ownership, retry bookkeeping,
  and aggregation checks.
- `reuse_contract.py` validates the single approved retained bank and its exact
  90-reused/12-replacement partition; `reuse.py` owns this bank's reservation,
  worker recovery and atomic completion.
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

## Refractory replacement workflow

This workflow is restricted to the source and partition in
[PLAN.md, step 5.1](../../PLAN.md#51-implement-exp022s-bank-reuse-workflow).
It accepts only `exp022-r001-compute` with payload digest
`sha256:9e3c93df9541809d1d019fe5290afbf7dff7d07ec14b07160fabe7ad79c9a0a8`.
The 90 compatible cells are copied without changing any of their four files.
The twelve production workers cover seeds 42–44 at 0.05, 0.2, 0.3 and 0.6 ms,
with explicit 1.2/0.6-ms refractories and exact timestep conversion.

```sh
# Read-only inspection: source, all cell names, original origins and file hashes.
uv run python -m experiments.exp022.compute --reuse-plan

# On the intended execution checkout, allocate before submitting workers.
# Requires a clean checkout and the validated source bank locally available.
uv run python -m experiments.exp022.compute --reuse-reserve --execution-origin slurm-wilkes

# Use the returned ID. Each worker trains only its named replacement cell.
uv run python -m experiments.exp022.compute --reuse-train-cell ping__dt0p2__seed42 --run-id <reserved-id>
uv run python -m experiments.exp022.compute --reuse-status --run-id <reserved-id>

# Separately authorized after all twelve workers validate: assemble the bank
# and regenerate all 34 seed-42 diagnostics using final-epoch checkpoints.
uv run python -m experiments.exp022.compute --reuse-finalize --run-id <reserved-id>
```

When dispatching to a separate checkout, an identity already allocated in the
canonical store can be initialized with `--reuse-reserve --run-id <reserved-id>`
after transferring its exact fresh v4 reservation. The declared origin must
match. Initialization rejects any populated or previously initialized writer.
The [reuse array worker](slurm/reuse-array.sbatch) runs the frozen replacement
selection and never finalizes the bank; account, wall time and array limits are
supplied explicitly at submission.

Reservation immediately records the input pin in the incomplete writer's
`run.json`, protecting its ancestry during pruning. Training outputs remain in
that writer's scratch area. Finished cells are skipped; failed attempts are
preserved before retry. Add `--recover-stale` to a worker only for a provably
inactive prior owner. Unconfirmed or active owners are rejected. Use these reuse
commands, rather than the older campaign commands, for this reservation.

Finalization excludes active workers, verifies all twelve training results and
both checkpoint roles, copies all 102 cells, and regenerates the diagnostics.
Original per-cell training origins, new worker attempts and diagnostic commands
are retained in `run.json`; recordings are canonical flat singleton exports.
No analysis, presentation, materialization or publication is triggered.

An interrupted finalization stays hidden. Retry it with
`--reuse-finalize --run-id <reserved-id> --recover-finalization`. Before the
assembled export is checkpointed, recovery retains the trained cells and repeats
diagnostic generation. After that checkpoint, recovery validates the recorded
export checksum and finishes cleanup/atomic visibility without retraining or
regenerating diagnostics. Completed runs cannot be reopened for mutation.

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
