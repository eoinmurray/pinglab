#!/bin/bash
set -euo pipefail
repo_root="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$repo_root"
exec "${EXP112_UV:-uv}" run --no-sync python -m experiments.exp112.hpc "$@"
