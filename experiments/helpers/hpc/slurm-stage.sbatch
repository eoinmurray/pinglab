#!/bin/bash
set -euo pipefail

plan="${1:?reviewed plan path is required}"
stage="${2:?stage is required}"
repo="${3:?frozen checkout path is required}"
environment="${4:?gpu or cpu environment is required}"
experiment="${5:?experiment module is required}"

cd "$repo"
if [[ "$environment" == "gpu" ]]; then
  source experiments/helpers/hpc/load-wilkes-modules.sh
elif [[ "$environment" == "cpu" ]]; then
  source /etc/profile.d/modules.sh
  module purge
  module load rhel8/default-icl
else
  echo "unknown execution environment: $environment" >&2
  exit 2
fi
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export PINGLAB_NO_SYNC=1
cache="$(.venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["mnist_cache"])' "$plan")"
experiments/helpers/hpc/ensure-mnist-link.sh "$cache"
exec .venv/bin/python -m "experiments.${experiment}.hpc" worker "$plan" "$stage"
