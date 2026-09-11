#!/bin/bash

set -euo pipefail

if [[ $# -gt 1 || "${1:-}" != "" && "${1:-}" != "--dry-run" ]]; then
  echo "usage: $0 [--dry-run]" >&2
  exit 2
fi
: "${EXP112_SLURM_ACCOUNT:?set the GPU project reported by mybalance}"
: "${EXP112_MNIST_CACHE:?set the prepopulated persistent MNIST cache}"
walltime="${EXP112_WALLTIME:-01:00:00}"
concurrency="${EXP112_CONCURRENCY:-4}"
[[ "$concurrency" == "4" ]] || { echo "exp112 requires concurrency 4" >&2; exit 2; }
[[ "$walltime" =~ ^[0-9]{2}:[0-9]{2}:[0-9]{2}$ ]] || { echo "walltime must be HH:MM:SS" >&2; exit 2; }

repo_root="$(cd "$(dirname "$0")/../../.." && pwd)"
uv_bin="${EXP112_UV:-$(command -v uv)}"
[[ -x "$uv_bin" ]] || { echo "uv executable is not usable: $uv_bin" >&2; exit 2; }
mnist_cache="$(realpath "$EXP112_MNIST_CACHE")"
[[ -d "$mnist_cache/MNIST" ]] || { echo "prepopulated MNIST directory missing under $mnist_cache" >&2; exit 2; }

cd "$repo_root"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "exp112 HPC submission requires a clean committed checkout" >&2
  exit 2
fi

echo "conditions: 4 (one A100 each)"
echo "array:      0-3%4"
echo "wall time:  $walltime per task"
echo "account:    $EXP112_SLURM_ACCOUNT"
echo "MNIST:      $mnist_cache"
if [[ "${1:-}" == "--dry-run" ]]; then
  echo "dry run: no identities reserved and no job submitted"
  exit 0
fi

run_ids=()
for _index in 0 1 2 3; do
  run_ids+=("$("$uv_bin" run --no-sync python -c \
    'import sys; from pathlib import Path; sys.path.insert(0, "tools"); from pingstore.stages import reserve_stage; print(reserve_stage(Path(".pingstore"), "exp112", "compute", origin="slurm-wilkes"))')")
done
joined="$(IFS=,; echo "${run_ids[*]}")"
log_dir="$repo_root/.pingstore/slurm/exp112"
mkdir -p "$log_dir"

printf 'reserved:   %s\n' "${run_ids[@]}"
submission="$(sbatch \
  --account="$EXP112_SLURM_ACCOUNT" \
  --time="$walltime" \
  --array="0-3%4" \
  --output="$log_dir/%A_%a.out" \
  --error="$log_dir/%A_%a.err" \
  --export="PINGLAB_ROOT=$repo_root,EXP112_RUN_0=${run_ids[0]},EXP112_RUN_1=${run_ids[1]},EXP112_RUN_2=${run_ids[2]},EXP112_RUN_3=${run_ids[3]},EXP112_UV=$uv_bin,EXP112_MNIST_CACHE=$mnist_cache" \
  experiments/exp112/slurm/array.sbatch)"
echo "$submission"
echo "run IDs: $joined"
