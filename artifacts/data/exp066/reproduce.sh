#!/usr/bin/env bash
set -euo pipefail
uv run python experiments/exp066.py --runpod
# Add --live only with explicit RunPod spending authority.
uv run python experiments/exp066.py --runpod --collect
uv run python experiments/exp066.py --skip-training
