#!/bin/bash
#SBATCH --job-name=exp022-fr-pilot
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

set -euo pipefail

: "${PINGLAB_ROOT:?set the clean source checkout}"
: "${EXP022_PILOT_ROOT:?set an unused calibration output root}"
: "${EXP022_UV:?set the uv executable}"
: "${PINGLAB_DATA_ROOT:?set the persistent MNIST cache}"
: "${SLURM_ARRAY_TASK_ID:?submit this script as the registered 8-cell array}"

models=(coba coba coba coba ping ping ping ping)
strengths=(0.004 0.016 0.041 0.1 0.004 0.016 0.041 0.1)
if (( SLURM_ARRAY_TASK_ID >= ${#models[@]} )); then
  echo "array task $SLURM_ARRAY_TASK_ID exceeds the 8-cell pilot" >&2
  exit 2
fi

cd "$PINGLAB_ROOT"
source experiments/exp022_support/load-wilkes-modules.sh
experiments/exp022_support/ensure-mnist-link.sh "$PINGLAB_DATA_ROOT"
"$EXP022_UV" run python -m experiments.exp022_support.fr_strength_pilot \
  --model "${models[$SLURM_ARRAY_TASK_ID]}" \
  --strength "${strengths[$SLURM_ARRAY_TASK_ID]}" \
  --output-root "$EXP022_PILOT_ROOT"
