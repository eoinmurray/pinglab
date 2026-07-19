#!/usr/bin/env bash
set -euo pipefail
uv run python experiments/exp069.py
uv run python experiments/exp069.py --runpod
# Add --live only with explicit RunPod spending authority.
uv run python experiments/exp069.py --runpod --collect
uv run python experiments/exp069.py --skip-training
