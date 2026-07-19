#!/usr/bin/env bash
set -euo pipefail
EXP070_ATTEMPT=temporal_2ms uv run python experiments/exp070.py
EXP070_ATTEMPT=temporal_2ms uv run python experiments/exp070.py --runpod
# Add --live only with fresh, explicit RunPod spending authority.
EXP070_ATTEMPT=temporal_2ms uv run python experiments/exp070.py --runpod --collect
EXP070_ATTEMPT=temporal_2ms uv run python experiments/exp070.py --skip-training
