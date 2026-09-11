# Exp022: COBA/PING model bank

Exp022 owns a 102-cell COBA/PING training registry used by several downstream
experiments. Operational execution follows the repository [Experiment Runner
Guide](../README.md) and requires `pingstore.run/v4`.

## Code layout

- `recipe.py` owns the scientific registry and read-only model-bank interface.
- `compute.py` owns direct and parallel training, frozen bank manifests,
  per-cell validation and recovery, v4 bank import, and explicitly requested
  retained diagnostic simulations.
- `analyse.py` measures a completed compute run without executing new science.
- `present.py` renders a completed analysis and can explicitly carry verified
  historical raster images when the original raw probes were not retained.
- `hpc.py` owns the reviewed Slurm plan and standard submission adapter.

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

Bank-array, import, recovery, and diagnostic modes are explicit operations. Use
`compute.py --help` and `hpc.py --help` for their arguments. No exp022 command
schedules other experiments, analyses, presentations, or publication.

Completed runs contain exactly `run.json`, `README.md`, and `export/`. The 102
model cells are direct scientific-unit directories in compute exports. Analysis
exports contain measurements and plot-ready arrays. Presentation exports are
flat publication inputs. All readers validate the complete v4 source and its
payload digest before use.

## Parallel HPC bank training

The standard HPC adapter freezes all selected cell configurations, the
one-cell-per-task allocation, clean source identity and scheduler resources in
a reviewable plan. Preparing creates the empty bank and reserves its compute
run; reviewing does not submit unless `--live` is explicit:

```sh
uv run python -m experiments.exp022.hpc prepare \
  --root <working-root> --plan .scratch/exp022-hpc/production.json \
  --account <gpu-account> --mnist-cache <persistent-torch-data> \
  --walltime <measured-HH:MM:SS> --concurrency <reviewed-limit>
uv run python -m experiments.exp022.hpc review \
  .scratch/exp022-hpc/production.json
uv run python -m experiments.exp022.hpc review \
  .scratch/exp022-hpc/production.json --test-only
uv run python -m experiments.exp022.hpc review \
  .scratch/exp022-hpc/production.json --live
```

The working root and plan must not already exist. Receipt-first submission
launches one array task per cell and an `afterok` collector. The collector never
trains a missing cell: it validates all 102 cells, generates the retained
diagnostic probes and atomically completes the reserved v4 compute run.

Exp022 retains its richer per-cell attempt and recovery records underneath the
standard reviewed-plan interface. Every fresh plan covers the complete 102-cell
bank so its dependent collector can finish the reserved run. Valid cells are
skipped; failed partial output is preserved, and stale ownership requires
explicit reviewed recovery after the scheduler confirms the prior owner is
inactive.

After every task is inactive, inspect `--bank-status <working-root>/bank.json`.
Run `--bank-finalize <working-root>/bank.json` on an allocated GPU node: it
requires all 102 cells, validates them again, generates the diagnostic probes,
and atomically completes the preallocated v4 compute run. It does not analyse,
present, materialize, or publish. The retired 90/12 reconstruction workflow and
its source-pinning contract remain recoverable in Git history.

## Exp110 COBA gradient-damping replacement

Exp110's comparison requires every COBA cell to use the same backward
voltage-gradient damping divisor as PING. These are the three `TR-01` canonical
cells trained on the full 60,000-image pool plus the eighteen `TR-02` cells for
the six activity conditions (off, 25, 10, 5, 2.5 and 1 Hz) at seeds 42–44. Their
divisor is 1000 (`--v-grad-dampen 1000`) and their recurrent loop remains
disabled (`--ei-strength 0`).

The replacement workflow is pinned to `exp022-r007-compute`. It creates one new
standalone 102-cell bank by copying 81 compatible cells byte-for-byte and
training only the 21 replacements. Both checkpoint roles are retained. The
source run, per-cell file hashes, training attempts, resolved parameters and
regenerated seed-42 diagnostic provenance are recorded in the new run's
`run.json`; no provenance sidecars enter `export/`.

```sh
# Read-only inspection of the exact source and 81/21 partition.
uv run python -m experiments.exp022.compute --reuse-plan

# Reserve and execute through the reviewed HPC adapter.
uv run python -m experiments.exp022.hpc prepare \
  --replacement exp110-coba-damping \
  --root <working-root> \
  --plan <working-root>/production.json \
  --account <account> \
  --mnist-cache <persistent-data-root> \
  --walltime 12:00:00 \
  --concurrency 21

uv run python -m experiments.exp022.hpc review <working-root>/production.json
uv run python -m experiments.exp022.hpc review \
  <working-root>/production.json --test-only
uv run python -m experiments.exp022.hpc review \
  <working-root>/production.json --live
```

Preparation requires a clean committed checkout, validates the pinned source,
allocates the compute identity before dispatch and freezes exactly 21 one-cell
workers. Review is read-only. Test-only and live submission are separate
explicit operations. The collector validates all replacements and both
checkpoint roles, assembles all 102 cells, regenerates the 34 seed-42 diagnostic
recordings and atomically exposes the run. It never launches analysis,
presentation or publication.
