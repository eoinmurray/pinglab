#!/bin/bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: EXP022_SLURM_ACCOUNT=<PROJECT-GPU> $0 TIER" >&2
  echo "tiers: standard fine_dt canonical_coba canonical_ping variable_rate" >&2
  exit 2
fi

: "${EXP022_SLURM_ACCOUNT:?set this to the GPU project reported by mybalance}"

tier="$1"
case "$tier" in
  standard)       walltime="12:00:00"; concurrency=16 ;;
  fine_dt)        walltime="24:00:00"; concurrency=3 ;;
  canonical_coba) walltime="36:00:00"; concurrency=3 ;;
  canonical_ping) walltime="36:00:00"; concurrency=3 ;;
  variable_rate)  walltime="12:00:00"; concurrency=3 ;;
  *) echo "unknown tier: $tier" >&2; exit 2 ;;
esac

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$repo_root"
mapfile -t cells < <(uv run python experiments/exp022.py --list-cells "$tier")
if [[ ${#cells[@]} -eq 0 ]]; then
  echo "tier $tier contains no cells" >&2
  exit 1
fi

mkdir -p logs/exp022
last_index=$((${#cells[@]} - 1))
echo "submitting tier=$tier cells=${#cells[@]} array=0-${last_index}%${concurrency} time=$walltime"
sbatch \
  --account="$EXP022_SLURM_ACCOUNT" \
  --time="$walltime" \
  --array="0-${last_index}%${concurrency}" \
  --export="ALL,PINGLAB_ROOT=$repo_root,EXP022_TIER=$tier" \
  experiments/exp022/train-array.sbatch
