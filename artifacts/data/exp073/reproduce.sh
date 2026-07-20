#!/usr/bin/env bash
set -euo pipefail
uv run python experiments/exp073.py --plot-only
EXP073_ATTEMPT=plastic_wee uv run python experiments/exp073.py
EXP073_ATTEMPT=plastic_wee EXP073_STAGE=short uv run python experiments/exp073.py --runpod
EXP073_ATTEMPT=plastic_wee EXP073_STAGE=short uv run python experiments/exp073.py --runpod --collect
EXP073_ATTEMPT=plastic_wee EXP073_STAGE=short uv run python experiments/exp073.py --skip-training
# Promote only if the short run materially improves over the exp071/exp072 ~70% regime.
EXP073_ATTEMPT=plastic_wee EXP073_STAGE=final40 uv run python experiments/exp073.py --runpod
EXP073_ATTEMPT=plastic_wee EXP073_STAGE=final40 uv run python experiments/exp073.py --runpod --collect
EXP073_ATTEMPT=plastic_wee EXP073_STAGE=final40 uv run python experiments/exp073.py --skip-training
