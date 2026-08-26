#!/usr/bin/env bash
set -euo pipefail
uv run python experiments/exp072.py --plot-only
for attempt in control train_leak adaptive_threshold combined; do
  EXP072_ATTEMPT="$attempt" uv run python experiments/exp072.py
  for cell in coba ping; do
    EXP072_ATTEMPT="$attempt" EXP072_STAGE=short uv run python experiments/exp072.py --runpod --only-cells "$cell"
    # Add --live only with fresh, explicit RunPod spending authority.
  done
done
EXP072_ATTEMPT=train_leak EXP072_STAGE=final40 uv run python experiments/exp072.py --runpod
EXP072_ATTEMPT=train_leak EXP072_STAGE=final40 uv run python experiments/exp072.py --runpod --collect
EXP072_ATTEMPT=train_leak EXP072_STAGE=final40 uv run python experiments/exp072.py --skip-training
EXP072_ATTEMPT=combined EXP072_STAGE=final80 uv run python experiments/exp072.py --runpod
EXP072_ATTEMPT=combined EXP072_STAGE=final80 uv run python experiments/exp072.py --runpod --collect
EXP072_ATTEMPT=combined EXP072_STAGE=final80 uv run python experiments/exp072.py --skip-training
