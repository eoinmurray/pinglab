#!/bin/bash
#SBATCH --job-name=exp112
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

set -euo pipefail

: "${PINGLAB_ROOT:?submit through experiments/exp112/slurm/submit.sh}"
: "${EXP112_RUN_0:?submit through experiments/exp112/slurm/submit.sh}"
: "${EXP112_RUN_1:?submit through experiments/exp112/slurm/submit.sh}"
: "${EXP112_RUN_2:?submit through experiments/exp112/slurm/submit.sh}"
: "${EXP112_RUN_3:?submit through experiments/exp112/slurm/submit.sh}"
: "${EXP112_UV:?submit through experiments/exp112/slurm/submit.sh}"
: "${EXP112_MNIST_CACHE:?submit through experiments/exp112/slurm/submit.sh}"
: "${SLURM_ARRAY_TASK_ID:?this script must run as a Slurm array}"

if [[ $SLURM_ARRAY_TASK_ID -lt 0 || $SLURM_ARRAY_TASK_ID -ge 4 ]]; then
  echo "exp112 requires array index 0-3" >&2
  exit 2
fi
run_variable="EXP112_RUN_${SLURM_ARRAY_TASK_ID}"
run_id="${!run_variable}"

cd "$PINGLAB_ROOT"
source experiments/exp112/slurm/load-wilkes-modules.sh
experiments/exp112/slurm/ensure-mnist-link.sh "$EXP112_MNIST_CACHE"
echo "job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID run=$run_id"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
"$EXP112_UV" run --no-sync python experiments/exp112/compute.py \
  --shard-index "$SLURM_ARRAY_TASK_ID" \
  --run-id "$run_id"
