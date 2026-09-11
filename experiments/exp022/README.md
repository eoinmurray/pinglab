# Exp022: COBA/PING model bank

Exp022 owns a 102-cell COBA/PING training registry used by several downstream
experiments. Operational execution follows the repository [Experiment Runner
Guide](../README.md) and requires `pingstore.run/v4`.

## Code layout

- `recipe.py` owns the scientific registry and read-only model-bank interface.
- `compute.py` owns direct and parallel training, frozen bank manifests,
  per-cell validation and recovery, RunPod dispatch, v4 bank import, and
  explicitly requested retained diagnostic simulations.
- `analyse.py` measures a completed compute run without executing new science.
- `present.py` renders a completed analysis and can explicitly carry verified
  historical raster images when the original raw probes were not retained.
- `slurm/` contains exp022 Wilkes environment checks, the generic bank-array
  worker, and the [operator runbook](slurm/README.md).

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

RunPod, bank-array, import, recovery, and diagnostic modes are explicit compute operations.
Use `compute.py --help` and the Slurm runbook for their arguments. No exp022
command schedules other experiments, analyses, presentations, or publication.

Completed runs contain exactly `run.json`, `README.md`, and `export/`. The 102
model cells are direct scientific-unit directories in compute exports. Analysis
exports contain measurements and plot-ready arrays. Presentation exports are
flat publication inputs. All readers validate the complete v4 source and its
payload digest before use.

## Parallel HPC bank training

The single-command HPC entry point creates a new empty bank for all 102 cells
and submits one Slurm array task per cell:

```sh
uv run python -m experiments.exp022.compute --hpc-root <working-root> --hpc
```

Set the same `EXP022_SLURM_ACCOUNT`, `EXP022_WALLTIME`,
`EXP022_CONCURRENCY`, `EXP022_MNIST_CACHE`, and optional `EXP022_UV`
variables used by `submit-bank.sh`. The working root must not already exist.
This never edits an existing Pingstore run. After the array finishes, status and
finalization remain explicit operations.

Exp022 retains an experiment-local parallel runner. It does not schedule other
experiments or read a collection registry. A clean checkout creates one frozen
manifest for all 102 committed scenarios and preallocates its compute run:

```sh
uv run python -m experiments.exp022.compute \
  --bank-create <working-root> --execution-origin slurm-wilkes
experiments/exp022/slurm/submit-bank.sh <working-root>/bank.json all
```

The submission wrapper freezes the retry selection and launches one cell per
Slurm array task. `--tier` selections partition the same manifest without
changing scientific parameters. Valid cells are skipped; failed partial output
is preserved, and stale ownership requires an explicit `--recover-stale` worker
invocation after the scheduler confirms the prior owner is inactive.

After every task is inactive, inspect `--bank-status <working-root>/bank.json`.
Run `--bank-finalize <working-root>/bank.json` on an allocated GPU node: it
requires all 102 cells, validates them again, generates the diagnostic probes,
and atomically completes the preallocated v4 compute run. It does not analyse,
present, materialize, or publish. The retired 90/12 reconstruction workflow and
its source-pinning contract remain recoverable in Git history.
