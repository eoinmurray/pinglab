#!/bin/bash

set -euo pipefail

usage() {
  echo "usage: EXP022_SLURM_ACCOUNT=<account> EXP022_WALLTIME=<HH:MM:SS> EXP022_CONCURRENCY=<N> $0 MANIFEST TIER [--dry-run|--test-only]" >&2
  echo "tiers: all standard fine_dt canonical_coba canonical_ping variable_rate" >&2
}

if [[ $# -lt 2 || $# -gt 3 ]]; then usage; exit 2; fi
: "${EXP022_SLURM_ACCOUNT:?set the GPU project reported by mybalance}"
: "${EXP022_WALLTIME:?set a canary-measured wall time with margin}"
: "${EXP022_CONCURRENCY:?set the reviewed bank concurrency}"
: "${EXP022_MNIST_CACHE:?set the prepopulated persistent MNIST cache}"
uv_bin="${EXP022_UV:-$(command -v uv)}"
[[ -x "$uv_bin" ]] || { echo "uv executable is not usable: $uv_bin" >&2; exit 2; }
mnist_cache="$(realpath "$EXP022_MNIST_CACHE")"
[[ -d "$mnist_cache/MNIST" ]] || { echo "prepopulated MNIST directory missing under: $mnist_cache" >&2; exit 2; }

manifest="$(realpath "$1")"
tier="$2"
mode="${3:-submit}"
case "$tier" in
  all|standard|fine_dt|canonical_coba|canonical_ping|variable_rate) ;;
  *) echo "unknown tier: $tier" >&2; exit 2 ;;
esac
case "$mode" in submit|--dry-run|--test-only) ;; *) usage; exit 2 ;; esac
[[ -f "$manifest" ]] || { echo "missing manifest: $manifest" >&2; exit 2; }
[[ "$EXP022_CONCURRENCY" =~ ^[1-9][0-9]*$ ]] || { echo "concurrency must be positive" >&2; exit 2; }

repo_root="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$repo_root"
"$uv_bin" run python experiments/exp022/compute.py --bank-validate "$manifest"
mapfile -t cells < <("$uv_bin" run python experiments/exp022/compute.py \
  --bank-list "$manifest" --tier "$tier" --retry-only)
if [[ ${#cells[@]} -eq 0 ]]; then
  echo "tier $tier has no missing or invalid cells"
  exit 0
fi

bank_root="$("$uv_bin" run python -c 'import json,sys; print(json.load(open(sys.argv[1]))["bank_root"])' "$manifest")"
mkdir -p "$bank_root/logs" "$bank_root/submissions"
submission_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
selection="$bank_root/submissions/${tier}__${submission_stamp}.cells"
printf '%s\n' "${cells[@]}" > "$selection"
chmod 0444 "$selection"
last_index=$((${#cells[@]} - 1))
array="0-${last_index}%${EXP022_CONCURRENCY}"

echo "bank:        $manifest"
echo "tier:        $tier"
echo "cells:       ${#cells[@]}"
printf '  %s\n' "${cells[@]}"
echo "wall time:   $EXP022_WALLTIME"
echo "concurrency: $EXP022_CONCURRENCY"
echo "account:     $EXP022_SLURM_ACCOUNT"
echo "array:       $array"

sbatch_args=(
  --account="$EXP022_SLURM_ACCOUNT"
  --time="$EXP022_WALLTIME"
  --array="$array"
  --output="$bank_root/logs/%A_%a.out"
  --error="$bank_root/logs/%A_%a.err"
  --export="PINGLAB_ROOT=$repo_root,EXP022_BANK=$manifest,EXP022_SELECTION=$selection,EXP022_UV=$uv_bin,PINGLAB_DATA_ROOT=$mnist_cache"
)
echo "command: sbatch ${sbatch_args[*]} experiments/exp022/slurm/bank-array.sbatch"
if [[ "$mode" == "--dry-run" ]]; then exit 0; fi
if [[ "$mode" == "--test-only" ]]; then sbatch_args+=(--test-only); fi
submission="$(sbatch "${sbatch_args[@]}" experiments/exp022/slurm/bank-array.sbatch)"
echo "$submission"
printf '%s\n' "$submission" > "$bank_root/submissions/${tier}__${submission_stamp}.txt"
