# Compute options

This is the operational map for pinglab compute. It separates the persistent control plane from the machines that perform numerical work, and records the authentication boundary for each provider. Prices, balances, queues, and GPU stock change. Host aliases, hardware, accounts, and service names below are the recorded project configuration, not a live availability report. Check the relevant machine and provider before launch; an edit date is not evidence that credentials or capacity were revalidated.

Never place a password, TOTP seed, private SSH key, or Modal token in this repository. Authentication commands either use an encrypted local key or open the provider’s own interactive login flow.

## Execution workflow

Choose the stage before choosing the machine. From the repository root, these commands show the independent training-run entry points:

```sh
uv run python experiments/exp022/compute.py --help
uv run python experiments/exp022/analyse.py --help
uv run python experiments/exp022/present.py --help
```

The execution pattern is:

```sh
uv run python experiments/exp022/compute.py
uv run python experiments/exp022/analyse.py --source <compute-run-id>
uv run python experiments/exp022/present.py --source <analysis-run-id>
```

Replace each placeholder with the exact completed run ID printed by the preceding stage. The first command can perform substantial work: inspect the recipe and backend plan before running it. Analyse and present do not launch upstream work or automatically publish. Do not assume a command is a dry run merely because it lacks `--live`; that gate belongs to specific remote dispatch interfaces.

Each completed stage produces an immutable v3 run with `run.json` and `export/`. Hidden temporary directories are incomplete, not usable evidence. Preview selects a present run without changing retained evidence; materialization and deployment are separate operations. See the versioned [Experiment Runner Guide](https://github.com/eoinmurray/pinglab/blob/main/experiments/README.md) and [Storage Guide](https://github.com/eoinmurray/pinglab/blob/main/tools/pingstore/README.md).

## Local Mac

**When to use.** Use the Mac for editing, Demolab preview, plotting, analysis of collected results, dry-run dispatch plans, and plumbing-scale tests. It is the shortest feedback loop. It is not a CUDA training machine.

**Provider overview.** This is the local development workstation. It has no scheduler and no marginal compute charge. The working checkout is `/Users/eoin/pinglab`.

**How to use.** Local access requires the macOS user session rather than a separate infrastructure login.

```sh
cd /Users/eoin/pinglab
uv sync --dev

# Inspect a stage before deciding whether to execute it.
uv run python experiments/exp022/compute.py --help

# Preview the Demolab collection.
uv run demolab dev
```

Keep cloud commands in dry-run mode until spending has been explicitly authorised.

## Hetzner control plane

**When to use.** Use Hetzner for the persistent Codex session, orchestration, monitoring, result collection, and the public development preview. Do not use its small CPU and memory allocation for substantial numerical experiments.

**Provider overview.** Hetzner supplies the always-on Linux host named `pinglab-codex`. Caddy terminates HTTPS for `pl-hetzner.eoinmurray.info` and proxies to the Demolab development server on port `3000`. The server is a control plane, not a GPU worker.

**How to use.** The Mac alias `hetzner` authenticates as `eoin` with the dedicated Ed25519 identity configured in `~/.ssh/config`. The private key remains on the Mac. The Hetzner host has its own dedicated key for Olorin.

```sh
ssh hetzner

# Inspect the persistent session and preview service.
tmux list-sessions
# Attach only after choosing the session from the list.
tmux attach -t <session-name>
systemctl --user status demolab-dev.service

# Reach Olorin from the control plane.
ssh olorin 'hostname; whoami'
```

## Olorin

**When to use.** Use Olorin for large single-GPU or multi-GPU work when a GPU is visibly idle and the Division F fair-use policy permits the allocation. Its large VRAM makes it the first choice for workloads that do not fit on consumer GPUs.

**Provider overview.** Olorin is a shared Division F machine with four NVIDIA RTX PRO 6000 Blackwell Server Edition GPUs, each with approximately 96 GB of VRAM. It currently has no Slurm scheduler. Availability is therefore manual and may change between inspection and launch.

**How to use.** The Mac alias `olorin` connects through `gate.eng.cam.ac.uk` using `~/.ssh/olorin_codex`. Hetzner has a separate dedicated key. Passwords are not stored. Check utilization immediately before every launch, select only an idle GPU, and keep heavy files under `/scratch/em586`.

```sh
ssh olorin
nvidia-smi

# Inspect utilization without opening an interactive shell.
ssh olorin \
  'nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
  --format=csv,noheader'

# On Olorin, after confirming that GPU N is idle:
cd /scratch/em586/pinglab
CUDA_VISIBLE_DEVICES=N uv run python experiments/exp022/compute.py
```

Never launch on all four GPUs merely because the machine accepts the command. Shared infrastructure without a scheduler requires more restraint, not less.

## CSD3 Wilkes3, Service Level 2

**When to use.** Use SL2 for planned production runs where queued execution is acceptable. It is the scheduled backend for planned A100 jobs. Confirm the allocation balance, account permission, and queue before committing work.

**Provider overview.** Wilkes3 is Cambridge’s Slurm-managed A100 cluster. The SL2 GPU account is `OLEARY-SL2-GPU`; the Ampere partition uses QOS `gpu1`. Account balance and scheduler estimates are live administrative state and must be queried at submission time.

**How to use.** Connect as Cambridge user `em586`. SSH requires the Raven/UIS password plus the six-digit token labelled `CSD3: SSH Login`. The `csd3` alias uses `ControlMaster` so subsequent commands reuse the authenticated connection. An SSH key can replace the password factor, but mandatory TOTP remains unless Research Computing Services arranges an automation-specific solution.

```sh
# Establish or reuse the multiplexed connection.
ssh csd3

# Inspect purchased hours and account associations.
ssh csd3 mybalance
ssh csd3 'sacctmgr show assoc user="$USER" format=Account,Partition,QOS'

# On CSD3, submit a prepared job script with a reserved run identity.
sbatch \
  --account=OLEARY-SL2-GPU \
  --partition=ampere \
  --qos=gpu1 \
  --gres=gpu:1 \
  job.slurm

# On CSD3, ask Slurm for an estimate without submitting.
sbatch --test-only \
  --account=OLEARY-SL2-GPU --qos=gpu1 job.slurm
```

Command-line `sbatch` options override matching `#SBATCH` lines, so one job script can target either service level.

## CSD3 Wilkes3, Service Level 3

**When to use.** Use SL3 for non-urgent work that can wait for lower-priority capacity, for overflow, or for jobs submitted well before their results are needed. It is a poor interactive backend.

**Provider overview.** SL3 uses the same recorded Wilkes3 A100 hardware and Ampere partition with a lower-priority service level; no queue-time guarantee is implied. The GPU account is `OLEARY-SL3-GPU` and its QOS is `gpu2`.

**How to use.** Authentication is identical to SL2. Toggle the account and QOS at submission time:

```sh
sbatch \
  --account=OLEARY-SL3-GPU \
  --partition=ampere \
  --qos=gpu2 \
  --gres=gpu:1 \
  job.slurm

sbatch --test-only \
  --account=OLEARY-SL3-GPU --qos=gpu2 job.slurm

squeue -u "$USER"
```

The names are easy to invert: SL2 maps to `gpu1`, while SL3 maps to `gpu2`.

## Modal

**When to use.** Use Modal when managed function execution and reduced infrastructure management fit the workload; compare current cost and startup behaviour rather than assuming a fixed premium or guaranteed startup time. Only runners with an implemented Modal backend can use it.

**Provider overview.** Modal runs containerized functions and bills GPU, CPU, and memory by execution time. It provides managed scheduling rather than a persistent SSH host, so backend support must be confirmed in the selected runner.

**How to use.** `modal setup` opens Modal’s authentication flow and stores a local token. Never commit token values. For a runner implementing the project remote-dispatch contract, `--live` gates paid work. The low-level SNNSIM tool’s `--modal` is a different interface; do not assume it shares that gate.

```sh
# One-time interactive authentication.
uv run modal setup

# Replace <runner> with a runner confirmed to support Modal.
uv run python experiments/<runner>.py --help

# For a confirmed compatible runner: plan, then authorised dispatch.
uv run python experiments/<runner>.py --modal
uv run python experiments/<runner>.py --modal --live
```

Modal is not a drop-in flag for every experiment. If a runner does not expose `--modal`, adding a backend is implementation work rather than a command-line choice.

## Completion checklist

1.  **Before launch:** inspect the recipe, supported backend flags, resource request, output destination, and reserved run identity. Confirm shared-machine permission or paid-work approval.

2.  **During execution:** retain scheduler or job IDs, inspect logs and progress, and distinguish retries from a new scientific condition.

3.  **After execution:** require the stage’s completed run and validation, not just a successful scheduler exit or a directory containing files. Keep partial outputs hidden.

4.  **Before downstream work:** pass explicit validated v3 inputs. A new measurement needs analyse; a new rendering needs present. Neither should rerun compute implicitly.

5.  **Before leaving paid infrastructure:** verify collection and termination for this dispatch’s resources. Backup and recovery are described in [exp104](/exp104/) — [_Cloudflare R2 archive._](/exp104/)

## Decision order

Use the smallest adequate option. Develop locally; orchestrate from Hetzner; use an available Olorin GPU when shared-machine policy permits; submit planned production to Wilkes3 SL2; choose Modal when managed execution fits the task and current cost; and leave SL3 for work that can wait. [exp104](/exp104/) — [_Cloudflare R2 archive_](/exp104/)
