#!/bin/bash
#SBATCH --job-name=exp022
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

set -euo pipefail

: "${PINGLAB_ROOT:?submit through experiments/exp022/submit-tier.sh}"
: "${EXP022_MANIFEST:?submit through experiments/exp022/submit-tier.sh}"
: "${EXP022_TIER:?submit through experiments/exp022/submit-tier.sh}"
: "${EXP022_SELECTION:?submit through experiments/exp022/submit-tier.sh}"
: "${SLURM_ARRAY_TASK_ID:?this script must run as an array job}"

module purge
module load rhel8/default-amp

cd "$PINGLAB_ROOT"
uv run python experiments/exp022.py --campaign-validate "$EXP022_MANIFEST"
mapfile -t cells < "$EXP022_SELECTION"
if (( SLURM_ARRAY_TASK_ID >= ${#cells[@]} )); then
  echo "array task $SLURM_ARRAY_TASK_ID exceeds current retry list (${#cells[@]} cells)" >&2
  exit 2
fi
cell="${cells[$SLURM_ARRAY_TASK_ID]}"

echo "job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID tier=$EXP022_TIER cell=$cell"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
uv run python experiments/exp022.py --campaign-train-cell "$cell" --campaign "$EXP022_MANIFEST"
