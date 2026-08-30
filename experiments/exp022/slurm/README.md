# exp022 on Cambridge Wilkes3 SL2

This is the executable operator runbook for the 102-cell exp022 campaign. The
scientific registry remains solely in `experiments/exp022/recipe.py`; these scripts
only map a reviewed manifest onto Wilkes3 resources.

The [experiment README](../README.md) describes the stage and dependency layout.
Campaign validation lives in `experiments/exp022/campaign.py`. Site-specific
entrypoints live here in `experiments/exp022/slurm/`.

## Staged execution

Compute owns campaign submission, retries and retained diagnostic simulations.
A new manifest reserves its compute run identity before submission. Campaign
aggregation now completes that compute run only; it does not analyse, plot or
publish. Follow it explicitly with:

```sh
uv run python experiments/exp022/analyse.py --source <completed-compute-run>
uv run python experiments/exp022/present.py --source <completed-analysis-run>
```

Do not replace commands in an already-running historical campaign. Continue it
on its recorded checkout, then import the completed retained bank. New campaigns
use the staged entrypoint and unchanged scientific registry/checkpoint policy.
The collection driver composes the stages explicitly and copies presentation
into its own isolated working view, never the published repository view.

## 1. Persistent layout and reviewed checkout

Choose persistent locations permitted by the site policy. Do not publish the
real account, login host, or private directory names.

```bash
export PINGLAB_REPO=<persistent-repository>
export PINGLAB_UV_CACHE=<persistent-uv-cache>
export PINGLAB_MNIST_CACHE=<persistent-mnist-cache>
export EXP022_CAMPAIGNS=<persistent-campaign-root>
export EXP022_UV="$(command -v uv)"
cd "$PINGLAB_REPO"
git fetch origin
git checkout --detach <reviewed-commit>
git status --short
git diff --exit-code
sha256sum uv.lock
```

The checkout must be clean and its commit must match the campaign manifest.
Keep the repository, uv cache/environment, MNIST cache, campaign outputs, and
logs on persistent storage rather than node-local scratch. Before setup, check
both quota and free space using the site-supported quota command and `df -h`.

On an allocated Ampere node, use the Wilkes3 software stack:

```bash
module purge
module load rhel8/default-amp
export UV_CACHE_DIR="$PINGLAB_UV_CACHE"
uv sync --frozen
uv run python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

Do not reuse an environment produced from a different lockfile.

## 2. Prepopulate MNIST once

The current SNN loader reads `/tmp/mnist`. Prepopulate `MNIST/` once under the
persistent cache, then let the diagnostic and array scripts create a guarded
node-local `/tmp/mnist` symlink to that cache. They refuse an existing path that
resolves elsewhere. Confirm the files are readable from a second
non-interactive job before arrays start. Never let array workers race to
download MNIST.

```bash
export PINGLAB_DATA_ROOT="$PINGLAB_MNIST_CACHE"
uv run python -c 'from torchvision.datasets import MNIST; import sys; MNIST(sys.argv[1], train=True, download=True); MNIST(sys.argv[1], train=False, download=True)' "$PINGLAB_MNIST_CACHE"
experiments/exp022/slurm/ensure-mnist-link.sh "$PINGLAB_MNIST_CACHE"
uv run python experiments/exp022/slurm/wilkes_diagnostic.py --data-root "$PINGLAB_MNIST_CACHE" --output <diagnostic-root>/local.json
```

## 3. Account and diagnostic job

Use `mybalance` to identify the SL2 GPU account and inspect its remaining
allocation. Never commit the account name. Create a disposable diagnostic root
and submit the diagnostic script with explicit paths:

```bash
mybalance
sbatch --account=<SL2-GPU-account> --output=<diagnostic-root>/diagnostic-%j.out \
  --export=PINGLAB_ROOT="$PINGLAB_REPO",EXP022_DIAGNOSTIC_ROOT=<diagnostic-root>,PINGLAB_DATA_ROOT="$PINGLAB_MNIST_CACHE",EXP022_UV="$EXP022_UV" \
  experiments/exp022/slurm/diagnostic.sbatch
```

The JSON result must show the reviewed commit, a clean checkout, PyTorch/CUDA,
an allocated GPU, readable MNIST data, and a successful atomic write inside
the diagnostic root.

## 4. Create and review isolated manifests

Create smoke, canary, and production campaigns under different roots. A
manifest refuses a dirty checkout and records every resolved command.

```bash
uv run python experiments/exp022/compute.py \
  --campaign-manifest "$EXP022_CAMPAIGNS/smoke-<UTC>" \
  --campaign-id smoke-<UTC> --execution-origin slurm-wilkes --tier standard --plumbing

uv run python experiments/exp022/compute.py \
  --campaign-manifest "$EXP022_CAMPAIGNS/production-<UTC>" \
  --campaign-id production-<UTC> --execution-origin slurm-wilkes --tier all
```

Review `campaign.json`, especially its commit, lock hash, destination, 102-cell
count, scientific parameters, and commands. The manifest hash detects edits.

## 5. Status, tiny flow, and failure rehearsal

```bash
uv run python experiments/exp022/compute.py --campaign-status <root>/campaign.json
uv run python experiments/exp022/compute.py --campaign-status <root>/campaign.json --json > <root>/status/status.json
uv run python experiments/exp022/compute.py --campaign-train-cell <cell> --campaign <root>/campaign.json
```

For the disposable rehearsal, interrupt or corrupt one plumbing cell, rerun the
same command, and verify the partial directory moved under `failed/<cell>/`.
Rerun a valid cell and verify it prints `skip-valid` without changing hashes.
An atomic status lock excludes active cells from `--retry-only`; a second owner
is refused before it can move output. A stale record is listed separately under
`recoverable_cells`. After `squeue` confirms its Slurm job is absent, recover it
explicitly with:

```bash
uv run python experiments/exp022/compute.py --campaign-train-cell <cell> \
  --campaign <root>/campaign.json --recover-stale
```

If Slurm activity cannot be checked, stale recovery fails closed.

## 6. Slurm plumbing and canaries

First run `--dry-run`, then `--test-only`. Final wall times and concurrency are
mandatory environment inputs; provisional values are acceptable only for the
five canaries and must be replaced with measured margins.

```bash
export EXP022_SLURM_ACCOUNT=<SL2-GPU-account>
export EXP022_WALLTIME=<HH:MM:SS>
export EXP022_CONCURRENCY=<N>
export EXP022_MNIST_CACHE="$PINGLAB_MNIST_CACHE"
bash experiments/exp022/slurm/submit-tier.sh <manifest> standard --dry-run
bash experiments/exp022/slurm/submit-tier.sh <manifest> standard --test-only
```

Run one representative cell from each shape: standard, fine timestep,
canonical COBA, canonical PING, and variable rate. Record job ID, elapsed time,
`MaxRSS`, GPU memory, output size, and anomalies using `sacct` and the job logs.
Repeat the failure rehearsal through Slurm. Only then choose production margins.

## 7. Production review gate

Before submission, report the campaign ID and manifest hash, exact tier cell
counts, measured wall time and peak memory, requested margins, concurrency,
remaining SL2 balance, and the exact five submission commands. Production must
not be submitted from provisional estimates.

After approval, submit tiers with the same wrapper. Each submission freezes its
retry cell list under `submissions/`; valid cells are skipped and never erased.
Scientific completion is determined by the validator, not Slurm state.

## 8. Aggregate and recover

After all 102 cells validate, aggregate from that one campaign only. The
aggregator must refuse incomplete or mixed banks and complete only the compute
run. Run analysis and presentation explicitly as described above; aggregation
does not render figures or launch downstream experiments. Archive only the
exact verified campaign bank named by its manifest:

```bash
uv run python experiments/helpers/archive.py archive-campaign <campaign-root>/campaign.json
uv run python experiments/helpers/archive.py list exp022
uv run python experiments/helpers/archive.py restore-campaign exp022 <snapshot-id> --destination <separate-empty-directory>
```

The campaign archive refuses an incomplete bank, records the campaign ID,
manifest hash, producing commit, and per-file SHA-256 inventory, then runs a
downloaded `rclone check`. Restore requires a separate absent or empty
destination and verifies it against R2. Never use `sync`, delete a good
snapshot, or publish credentials and sensitive configuration.
