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

Prepare a complete frozen exp022 bank and review its exact allocation from a
clean checkout:

```sh
uv run python -m experiments.exp022.hpc prepare \
  --root <working-root> --plan <plan.json> \
  --account <account> --mnist-cache <torch-data> \
  --walltime <HH:MM:SS> --concurrency <N>
uv run python -m experiments.exp022.hpc review <plan.json>
uv run python -m experiments.exp022.hpc review <plan.json> --test-only
uv run python -m experiments.exp022.hpc review <plan.json> --live
```

The plan freezes complete cell configurations and one cell per array task. The
shared wrapper writes its receipt before submitting the array and dependent
collector. Keep separate submissions at least 120 seconds apart.

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
