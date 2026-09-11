#!/bin/bash
#SBATCH --job-name=exp112
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

set -euo pipefail

: "${SLURM_ARRAY_TASK_ID:?this script must run as a Slurm array}"

plan="${1:?reviewed plan path is required}"
cd "${2:?frozen checkout path is required}"
source experiments/exp112/slurm/load-wilkes-modules.sh
cache="$(.venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["mnist_cache"])' "$plan")"
experiments/exp112/slurm/ensure-mnist-link.sh "$cache"
echo "job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
exec .venv/bin/python -m experiments.exp112.hpc worker "$plan"
