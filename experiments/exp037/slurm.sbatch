#!/bin/bash
set -euo pipefail
cd "${3:?frozen checkout path is required}"
if [[ "$2" == "compute" ]]; then
  source experiments/exp022/slurm/load-wilkes-modules.sh
else
  source /etc/profile.d/modules.sh
  module purge
  module load rhel8/default-icl
fi
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export PINGLAB_NO_SYNC=1
plan="$1"
stage="$2"
cache="$(.venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["mnist_cache"])' "$plan")"
bash experiments/exp022/slurm/ensure-mnist-link.sh "$cache"
export PATH="$HOME/.local/bin:$PATH"
exec .venv/bin/python -m experiments.exp037.hpc worker "$plan" "$stage"
