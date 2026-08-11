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

module purge
module load rhel8/default-amp
cd "$PINGLAB_ROOT"
mkdir -p "$EXP022_DIAGNOSTIC_ROOT"
if [[ -e /tmp/mnist || -L /tmp/mnist ]]; then
  [[ "$(readlink -f /tmp/mnist)" == "$(readlink -f "$PINGLAB_DATA_ROOT")" ]] || exit 2
else
  ln -s "$PINGLAB_DATA_ROOT" /tmp/mnist
fi
"$EXP022_UV" run python experiments/exp022_wilkes_diagnostic.py \
  --data-root "$PINGLAB_DATA_ROOT" \
  --output "$EXP022_DIAGNOSTIC_ROOT/diagnostic-${SLURM_JOB_ID}.json"
