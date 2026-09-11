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
