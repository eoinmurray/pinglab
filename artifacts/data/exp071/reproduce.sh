#!/usr/bin/env bash
set -euo pipefail
uv run python experiments/exp071.py --plot-only
EXP071_ATTEMPT=cumulative_baseline uv run python experiments/exp071.py
EXP071_ATTEMPT=cumulative_baseline EXP071_STAGE=short uv run python experiments/exp071.py --runpod
# Add --live only with fresh, explicit RunPod spending authority.
EXP071_ATTEMPT=cumulative_baseline EXP071_STAGE=short uv run python experiments/exp071.py --runpod --live
EXP071_ATTEMPT=cumulative_baseline EXP071_STAGE=short uv run python experiments/exp071.py --runpod --collect
EXP071_ATTEMPT=cumulative_baseline EXP071_STAGE=short uv run python experiments/exp071.py --skip-training
