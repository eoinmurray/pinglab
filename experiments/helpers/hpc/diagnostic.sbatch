#!/bin/bash
set -euo pipefail

repo="${1:?reviewed checkout path is required}"
cache="${2:?persistent MNIST cache is required}"
output_root="${3:?diagnostic output root is required}"
module="${4:?diagnostic Python module is required}"

cd "$repo"
source experiments/helpers/hpc/load-wilkes-modules.sh
experiments/helpers/hpc/ensure-mnist-link.sh "$cache"
mkdir -p "$output_root"
exec .venv/bin/python -m "$module" \
  --data-root "$cache" \
  --output "$output_root/diagnostic-${SLURM_JOB_ID}.json"
