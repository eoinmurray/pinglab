#!/bin/bash
#SBATCH --job-name=exp022-fr-eval
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

set -euo pipefail

: "${PINGLAB_ROOT:?set the clean source checkout}"
: "${EXP022_PILOT_ROOT:?set the completed calibration output root}"
: "${EXP022_UV:?set the uv executable}"
: "${PINGLAB_DATA_ROOT:?set the persistent MNIST cache}"
: "${SLURM_ARRAY_TASK_ID:?submit this script as the registered 8-cell array}"

cells=(
  coba__lambda0p004__seed42
  coba__lambda0p016__seed42
  coba__lambda0p041__seed42
  coba__lambda0p1__seed42
  ping__lambda0p004__seed42
  ping__lambda0p016__seed42
  ping__lambda0p041__seed42
  ping__lambda0p1__seed42
)
if (( SLURM_ARRAY_TASK_ID >= ${#cells[@]} )); then
  echo "array task $SLURM_ARRAY_TASK_ID exceeds the 8-cell pilot" >&2
  exit 2
fi

cd "$PINGLAB_ROOT"
source experiments/exp022_support/load-wilkes-modules.sh
experiments/exp022_support/ensure-mnist-link.sh "$PINGLAB_DATA_ROOT"
cell_root="$EXP022_PILOT_ROOT/${cells[$SLURM_ARRAY_TASK_ID]}"
test -f "$cell_root/config.json"
test -f "$cell_root/weights_final.pth"
test ! -e "$cell_root/official-test"
"$EXP022_UV" run python tools/snnsim/tool.py sim --infer \
  --load-config "$cell_root/config.json" \
  --load-weights "$cell_root/weights_final.pth" \
  --out-dir "$cell_root/official-test" \
  --max-samples 1000
