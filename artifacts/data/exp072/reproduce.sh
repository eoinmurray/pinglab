#!/usr/bin/env bash
set -euo pipefail
uv run python experiments/exp072.py --plot-only
EXP072_ATTEMPT=control uv run python experiments/exp072.py
EXP072_ATTEMPT=control EXP072_STAGE=short uv run python experiments/exp072.py --runpod
# Add --live only with fresh, explicit RunPod spending authority.
EXP072_ATTEMPT=control EXP072_STAGE=short uv run python experiments/exp072.py --runpod --live
EXP072_ATTEMPT=control EXP072_STAGE=short uv run python experiments/exp072.py --runpod --collect
EXP072_ATTEMPT=control EXP072_STAGE=short uv run python experiments/exp072.py --skip-training
