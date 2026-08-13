#!/bin/bash
#SBATCH --nodes=1

set -euo pipefail

: "${PINGLAB_ROOT:?submit through the collection campaign scheduler}"

if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "usage: $0 ACTION CAMPAIGN_ROOT UV MNIST_CACHE [EXPERIMENT]" >&2
  exit 2
fi

action="$1"
campaign_root="$2"
uv_bin="$3"
mnist_cache="$4"
experiment="${5:-}"
cd "$PINGLAB_ROOT"
source experiments/exp022_support/load-wilkes-modules.sh
experiments/exp022_support/ensure-mnist-link.sh "$mnist_cache"

echo "job=$SLURM_JOB_ID host=$(hostname) action=$action experiment=${experiment:-none}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
fi

command=("$uv_bin" run --frozen python -m experiments.collections.gamma_gated_sparsity)
case "$action" in
  aggregate-exp022)
    "${command[@]}" aggregate-exp022 --campaign-root "$campaign_root"
    ;;
  run-experiment)
    [[ -n "$experiment" ]] || { echo "run-experiment requires an experiment" >&2; exit 2; }
    "${command[@]}" run-experiment --campaign-root "$campaign_root" --slug "$experiment"
    ;;
  finalize)
    "${command[@]}" finalize --campaign-root "$campaign_root"
    ;;
  *)
    echo "unknown collection action: $action" >&2
    exit 2
    ;;
esac
