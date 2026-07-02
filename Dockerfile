# Pre-baked training image for the RunPod fan-out (see nb022 "Compute options").
#
# Built once by GitHub Actions on amd64 (RunPod's architecture) and pushed to
# ghcr.io/eoinmurray/pinglab. It bakes the Python deps + a CUDA-12.8 (cu128)
# torch build so pods boot ready-to-train — a per-pod `uv sync` hung on a fresh
# box and is the entire reason this image exists.
#
# Code is NOT baked: pods `git pull` the latest at launch, so only a dependency
# change requires a rebuild. Pods must run with `uv run --no-sync` so the pinned
# cu128 torch below is not reverted to the lockfile's default (cu130) build.
FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PATH="/root/.local/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates build-essential openssh-client \
    && rm -rf /var/lib/apt/lists/*

# uv manages Python + the locked dependency set.
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone once to resolve the lockfile into a warm venv layer. This code snapshot
# is throwaway — pods refresh it with `git pull` at launch.
WORKDIR /workspace
RUN git clone --depth 1 https://github.com/eoinmurray/pinglab.git
WORKDIR /workspace/pinglab

# Bake the deps (skip the dev group = ruff/ty), then swap torch to the cu128
# build RunPod's driver (CUDA 12.8) supports. cu128 carries the recompile_limit
# rename, so no source patch is needed (unlike the cu124 fallback for the older
# CUED drivers).
RUN uv sync --no-dev \
 && uv pip install --index-url https://download.pytorch.org/whl/cu128 \
        torch torchvision

# Fail the build early if the baked env is broken. CUDA availability itself is
# checked at runtime on a GPU host, not here (the builder has no GPU).
RUN uv run --no-sync python -c "import torch; print('baked torch', torch.__version__)"

CMD ["/bin/bash"]
