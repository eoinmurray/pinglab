#!/bin/bash
#SBATCH --job-name=exp022-refractory
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=250G

# Resource limits, account, array selection and run identity are supplied at submission.
set -euo pipefail

: "${PINGLAB_ROOT:?set the frozen execution checkout}"
: "${EXP022_RUN_ID:?reserve the compute identity before submission}"
: "${EXP022_SELECTION:?set the frozen replacement-cell selection}"
: "${PINGLAB_DATA_ROOT:?set the prepopulated MNIST cache}"
: "${SLURM_ARRAY_TASK_ID:?this script must run as an array job}"

cd "$PINGLAB_ROOT"
source experiments/exp022/slurm/load-wilkes-modules.sh
experiments/exp022/slurm/ensure-mnist-link.sh "$PINGLAB_DATA_ROOT"
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
unset PINGLAB_NO_COMPILE PINGLAB_NB022_PLUMBING

mapfile -t cells < "$EXP022_SELECTION"
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= ${#cells[@]} )); then
    echo "array task is outside the frozen replacement selection" >&2
    exit 2
fi
cell="${cells[$SLURM_ARRAY_TASK_ID]}"
echo "run=$EXP022_RUN_ID job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID cell=$cell"
.venv/bin/python - <<'PY'
import json
import torch
assert torch.cuda.is_available(), "replacement training requires an allocated CUDA GPU"
print(json.dumps({"torch": torch.__version__, "cuda": torch.version.cuda,
                  "gpu": torch.cuda.get_device_name(0)}), flush=True)
PY

# This worker completes one training cell. Bank finalization remains a separate step.
.venv/bin/python -m experiments.exp022.compute \
    --reuse-train-cell "$cell" --run-id "$EXP022_RUN_ID"
