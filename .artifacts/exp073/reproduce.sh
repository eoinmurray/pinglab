#!/usr/bin/env bash
set -euo pipefail
uv run python experiments/exp073.py --plot-only
EXP073_ATTEMPT=plastic_wee uv run python experiments/exp073.py
# The matched local gate failed for COBA. With explicit follow-up authority,
# continue the surviving PING cell only by setting EXP073_PING_ONLY=1.
# Preferred cloud path after RunPod transport stalls: Modal synchronous dispatch.
# Optional: choose a SKU with PINGLAB_MODAL_GPU=L40S|A10G|A100|H100.
EXP073_PING_ONLY=1 EXP073_ATTEMPT=plastic_wee EXP073_STAGE=short uv run python experiments/exp073.py --modal --only-cells ping
EXP073_PING_ONLY=1 EXP073_ATTEMPT=plastic_wee EXP073_STAGE=short uv run python experiments/exp073.py --modal --only-cells ping --live
EXP073_PING_ONLY=1 EXP073_ATTEMPT=plastic_wee EXP073_STAGE=short uv run python experiments/exp073.py --skip-training
# Historical RunPod commands retained for the original pre-pivot design:
EXP073_PING_ONLY=1 EXP073_ATTEMPT=plastic_wee EXP073_STAGE=short uv run python experiments/exp073.py --runpod --only-cells ping
EXP073_PING_ONLY=1 EXP073_ATTEMPT=plastic_wee EXP073_STAGE=short uv run python experiments/exp073.py --runpod --only-cells ping --live
EXP073_PING_ONLY=1 EXP073_ATTEMPT=plastic_wee EXP073_STAGE=short uv run python experiments/exp073.py --runpod --collect
EXP073_ATTEMPT=plastic_wee EXP073_STAGE=short uv run python experiments/exp073.py --runpod
EXP073_ATTEMPT=plastic_wee EXP073_STAGE=short uv run python experiments/exp073.py --runpod --collect
EXP073_ATTEMPT=plastic_wee EXP073_STAGE=short uv run python experiments/exp073.py --skip-training
# Promote only if the short run materially improves over the exp071/exp072 ~70% regime.
EXP073_ATTEMPT=plastic_wee EXP073_STAGE=final40 uv run python experiments/exp073.py --runpod
EXP073_ATTEMPT=plastic_wee EXP073_STAGE=final40 uv run python experiments/exp073.py --runpod --collect
EXP073_ATTEMPT=plastic_wee EXP073_STAGE=final40 uv run python experiments/exp073.py --skip-training
