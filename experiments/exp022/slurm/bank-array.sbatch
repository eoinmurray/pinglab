#!/bin/bash
#SBATCH --job-name=exp022-bank
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

set -euo pipefail

: "${PINGLAB_ROOT:?submit through experiments/exp022/slurm/submit-bank.sh}"
: "${EXP022_BANK:?submit through experiments/exp022/slurm/submit-bank.sh}"
: "${EXP022_SELECTION:?submit through experiments/exp022/slurm/submit-bank.sh}"
: "${EXP022_UV:?submit through experiments/exp022/slurm/submit-bank.sh}"
: "${PINGLAB_DATA_ROOT:?submit through experiments/exp022/slurm/submit-bank.sh}"
: "${SLURM_ARRAY_TASK_ID:?this script must run as an array job}"

cd "$PINGLAB_ROOT"
source experiments/exp022/slurm/load-wilkes-modules.sh
experiments/exp022/slurm/ensure-mnist-link.sh "$PINGLAB_DATA_ROOT"
"$EXP022_UV" run python experiments/exp022/compute.py --bank-validate "$EXP022_BANK"
mapfile -t cells < "$EXP022_SELECTION"
if (( SLURM_ARRAY_TASK_ID >= ${#cells[@]} )); then
  echo "array task $SLURM_ARRAY_TASK_ID exceeds frozen selection (${#cells[@]} cells)" >&2
  exit 2
fi
cell="${cells[$SLURM_ARRAY_TASK_ID]}"

echo "job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID cell=$cell"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
"$EXP022_UV" run python experiments/exp022/compute.py \
  --bank-train-cell "$cell" --bank "$EXP022_BANK"
