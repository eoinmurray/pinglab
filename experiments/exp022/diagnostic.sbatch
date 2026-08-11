#!/bin/bash
#SBATCH --job-name=exp022-diag
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00

set -euo pipefail
: "${PINGLAB_ROOT:?set the reviewed persistent checkout}"
: "${EXP022_DIAGNOSTIC_ROOT:?set a disposable persistent diagnostic root}"
: "${PINGLAB_DATA_ROOT:?set the prepopulated persistent MNIST cache}"
: "${EXP022_UV:?set the reviewed uv executable}"

cd "$PINGLAB_ROOT"
source experiments/exp022/load-wilkes-modules.sh
mkdir -p "$EXP022_DIAGNOSTIC_ROOT"
experiments/exp022/ensure-mnist-link.sh "$PINGLAB_DATA_ROOT"
"$EXP022_UV" run python experiments/exp022_wilkes_diagnostic.py \
  --data-root "$PINGLAB_DATA_ROOT" \
  --output "$EXP022_DIAGNOSTIC_ROOT/diagnostic-${SLURM_JOB_ID}.json"
