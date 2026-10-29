---
title: Compute Access
author: Eoin Murray
collection: notes
description: Host access and scheduler command examples, read only when needed.
created_at: "2026-10-29T17:42:34Z"
updated_at: "2026-10-29T17:42:34Z"
---

# Compute access

Read the relevant host section when connecting or preparing a job. Authorization
and host roles are defined in [Compute Rules](/notes/rules/compute).
Commands below are examples, not startup steps.

## Local Mac — development and analysis

Use for editing, small tests, plotting and analysis. Not for CUDA training.

```sh
cd /Users/eoin/pinglab
uv sync --dev
uv run python experiments/exp022/compute.py --help
```

## Hetzner — persistent control host

`pinglab-codex` is our always-on host for persistent sessions, orchestration and monitoring, not heavy numerical work.

```sh
ssh hetzner
tmux list-sessions
tmux attach -t <session-name>

# Olorin is also accessible from Hetzner.
ssh olorin
```

## Olorin — shared, large-memory GPUs

Shared machine with four RTX PRO 6000 Blackwell GPUs, approximately 96 GB VRAM each. No Slurm: check availability and select an idle GPU explicitly. Keep heavy files under `/scratch/em586`.

```sh
ssh olorin
nvidia-smi
cd /scratch/em586/pinglab

# Replace N with an idle GPU index.
CUDA_VISIBLE_DEVICES=N uv run python experiments/exp022/compute.py
```

## CSD3 / Wilkes3 — queued A100 jobs

We often refer to CSD3 / Wilkes3 as “hpc”. This is shorthand for this compute option, not a separate provider or SSH alias.

Connect with `ssh csd3`; authentication requires the CSD3 login token. Submit prepared Slurm scripts using the appropriate service level:

| Service | Account | Partition | QOS | Use |
|---|---|---|---|---|
| SL2 | `OLEARY-SL2-GPU` | `ampere` | `gpu1` | Planned production |
| SL3 | `OLEARY-SL3-GPU` | `ampere` | `gpu2` | Lower-priority work that can wait |

```sh
ssh csd3
mybalance

# Submit using one of these:
sbatch --account=OLEARY-SL2-GPU --partition=ampere --qos=gpu1 --gres=gpu:1 job.slurm
sbatch --account=OLEARY-SL3-GPU --partition=ampere --qos=gpu2 --gres=gpu:1 job.slurm

# Monitor, inspect and cancel.
squeue -u "$USER"
scontrol show job <job-id>
scancel <job-id>
```

Run substantial work inside Slurm allocations, never on login nodes. Batch status checks and leave at least 120 seconds between recurring Slurm requests; use job arrays for large batches.

## Modal

Use the [Modal dispatcher README](../../libraries/modal_dispatcher/README.md)
for authentication, dispatch, explicit source inputs, collection and recovery.
Use the [HPC deployment README](../../libraries/hpc/README.md) for maintained
CSD3 deployment commands rather than constructing a new submission workflow.
