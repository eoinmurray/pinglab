#!/bin/bash
#SBATCH --job-name=exp022
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --output=logs/exp022/%A_%a.out
#SBATCH --error=logs/exp022/%A_%a.err

set -euo pipefail

: "${PINGLAB_ROOT:?submit through hpc/exp022/submit-tier.sh}"
: "${EXP022_TIER:?submit through hpc/exp022/submit-tier.sh}"

module purge
module load rhel8/default-amp

cd "$PINGLAB_ROOT"
mapfile -t cells < <(
  uv run python experiments/exp022.py --list-cells "$EXP022_TIER"
)
cell="${cells[$SLURM_ARRAY_TASK_ID]}"

echo "job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID tier=$EXP022_TIER cell=$cell"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
uv run python experiments/exp022.py --train-cell "$cell"
