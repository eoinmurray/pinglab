# Pre-baked training image for the RunPod fan-out (see nb022 "Compute options").
#
# Built once by GitHub Actions on amd64 (RunPod's architecture) and pushed to
# ghcr.io/eoinmurray/pinglab. It bakes the Python deps + a CUDA-12.8 (cu128)
# torch build so pods boot ready-to-train — a per-pod `uv sync` hung on a fresh
# box and is the entire reason this image exists.
#
# Code is NOT baked: pods `git pull` the latest at launch, so only a dependency
# change requires a rebuild. Pods must run training with `uv run --no-sync` so
# the pinned cu128 torch below is not reverted to the lockfile's default (cu130).
FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PATH="/root/.local/bin:${PATH}"

# openssh-server + rsync: RunPod drives the pod over SSH and we rsync artifacts
# back, so the container must run its own sshd (the CUDA base has neither).
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates build-essential openssh-server rsync \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /run/sshd

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

# Start script: install RunPod's injected public key, then run sshd in the
# foreground (keeps the container alive and lets the launcher connect). Written
# at build time; $PUBLIC_KEY is expanded at runtime, not here.
RUN printf '%s\n' \
      '#!/bin/bash' \
      'mkdir -p /root/.ssh && chmod 700 /root/.ssh' \
      '[ -n "$PUBLIC_KEY" ] && echo "$PUBLIC_KEY" > /root/.ssh/authorized_keys && chmod 600 /root/.ssh/authorized_keys' \
      'ssh-keygen -A' \
      'exec /usr/sbin/sshd -D -e' \
    > /start.sh && chmod +x /start.sh

CMD ["/start.sh"]
