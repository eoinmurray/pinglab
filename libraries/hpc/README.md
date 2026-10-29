# HPC experiment deployment

From the repository root, preview a CSD3 / Wilkes3 deployment:

```sh
uv run python -m libraries.hpc deploy \
  --experiment exp141 \
  --remote-dir /rds/user/em586/hpc-work/deployments/exp141-first \
  --service SL3 --walltime 04:00:00 \
  --cpus 4 --memory-gb 32 --gpus 1 \
  --receipt .pingstore/hpc/exp141-first.json
```

1. Inspect the printed commit, resources, destination and experiment arguments.
   Add `--live` to upload and submit. Live deployment requires a clean committed
   checkout, including new files. Nothing contacts SSH or Slurm during preview.
2. Sources travel as a Git bundle, preserving the commit and lockfile. The new
   remote directory must not exist. Its `repo/` is a detached checkout; deployment
   metadata, the batch script and logs stay outside it. SSH uses `csd3` by default,
   including the configured authentication. Git, tar and uv must be available on
   CSD3; uv is also found in `$HOME/.local/bin`.
3. Dependency installation (`uv sync --frozen`) and execution happen inside the
   Slurm allocation. Workers load the maintained Wilkes3 module stack. SL2 selects
   `OLEARY-SL2-GPU` / `gpu1`; SL3 selects `OLEARY-SL3-GPU` / `gpu2`, both on `ampere`.
   GPU count, CPU threads, memory and walltime are explicit resource options.
   `--gpus 0` selects CPU execution within this same partition.
4. Choose `--stage compute`, `--stage analyse` or `--stage present`. Arguments after
   `--` are passed unchanged to that entrypoint. Scientific configuration remains
   in each experiment; there is no experiment adapter or mandatory run-ID flag.
   One deployment submits one stage. Stages are never inferred or chained.
5. For source data, repeat `--input RUN_ID` for the explicit local completed runs.
   The tool validates and uploads their full ancestry, preserves manifests and
   checksums, then validates the pins again on the worker. Pass the corresponding
   experiment arguments too, for example `--stage analyse --input RUN_ID ... --
   --source RUN_ID`. Datasets and other files outside Git/Pingstore are not
   uploaded; provision them explicitly and pass the experiment's supported cache
   arguments. MNIST is not assumed for generic deployments.
6. The receipt is created before any remote mutation and records the acknowledged
   job ID. Never retry a receipt after a failed or ambiguous response: inspect
   its status, destination and Slurm first. Use a fresh destination and receipt
   for a deliberately new job. No automatic resubmission or cleanup occurs.
7. Use `ssh csd3 'squeue -j JOB_ID'` for a status check and retain at least
   120 seconds between recurring scheduler queries. Logs are in the deployment's
   `logs/`; scientific outputs are under `repo/.pingstore/`. Collection is manual:
   follow the Pingstore rules to transfer and validate completed runs and ancestry
   before using them locally or deleting remote data.

The other modules preserve the existing experiment-specific array/collector
adapters: `slurm_submit.py`, `slurm-stage.sbatch`, the Wilkes/MNIST shell helpers,
and `concurrent_compute.py`. Existing adapters now import them from `libraries.hpc`.
Use the generic CLI for new deployments; it does not assume these specialized
bank protocols.
