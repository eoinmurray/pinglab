# Exp022 on Cambridge Wilkes3 SL2

These helpers support exp022 only. They do not plan or execute the
gamma-gated-sparsity collection.

New work follows the repository [Experiment Runner Guide](../../README.md):
compute, analyse and present are independent commands, and every downstream
input is an explicit completed `pingstore.run/v4` run.

## Environment checks

Use a reviewed clean checkout, a frozen `uv.lock`, a persistent UV cache, and a
prepopulated MNIST cache. Run the local diagnostic first, then the short Slurm
diagnostic on an allocated GPU node:

```sh
uv sync --frozen
uv run python experiments/exp022/slurm/wilkes_diagnostic.py \
  --data-root <mnist-cache> --output <diagnostic-root>/local.json

sbatch --account=<account> --output=<diagnostic-root>/diagnostic-%j.out \
  --export=PINGLAB_ROOT=<checkout>,EXP022_DIAGNOSTIC_ROOT=<diagnostic-root>,PINGLAB_DATA_ROOT=<mnist-cache>,EXP022_UV=<uv-path> \
  experiments/exp022/slurm/diagnostic.sbatch
```

The result must identify the reviewed commit and lockfile, a clean checkout,
CUDA/PyTorch, the allocated GPU, readable MNIST data, and a successful atomic
write. Never run production computation on a login node.

## Parallel exp022 scenarios

Create a complete frozen exp022 bank manifest from a clean reviewed checkout:

```sh
uv run python -m experiments.exp022.compute --bank-create <working-root> \
  --execution-origin slurm-wilkes
```

Set `EXP022_SLURM_ACCOUNT`, `EXP022_WALLTIME`, `EXP022_CONCURRENCY`, and
`EXP022_MNIST_CACHE`, then submit all scenarios or one resource tier:

```sh
experiments/exp022/slurm/submit-bank.sh <working-root>/bank.json all --dry-run
experiments/exp022/slurm/submit-bank.sh <working-root>/bank.json all
```

The wrapper validates the manifest, freezes the retry list read-only, and
submits `bank-array.sbatch`. Each array task owns and trains exactly one scenario.
Use job arrays rather than loops of `sbatch`, and keep separate submissions at
least 120 seconds apart.

After the array is inactive, inspect status. Use `--recover-stale` only after
confirming the recorded owner is no longer active. Finalization generates
diagnostic probes, so run it on an allocated GPU node rather than a login node:

```sh
uv run python -m experiments.exp022.compute --bank-status <working-root>/bank.json
uv run python -m experiments.exp022.compute --bank-finalize <working-root>/bank.json
uv run python experiments/exp022/analyse.py --source <run-id>
uv run python experiments/exp022/present.py --source <analysis-run-id>
```

Historical campaign directories and archives are evidence. Do not alter their
commands, manifests, or payloads; restore an existing remote snapshot only into
a separate absent or empty directory using `experiments/helpers/archive.py`.
