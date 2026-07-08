# Pre-baked training image for the RunPod fan-out (see nb022 "Compute options").
#
# Built once by GitHub Actions on amd64 (RunPod's architecture) and pushed to
# ghcr.io/eoinmurray/pinglab. It bakes the Python deps + a CUDA-12.8 (cu128)
# torch build so pods boot ready-to-train — a per-pod `uv sync` hung on a fresh
# box and is the entire reason this image exists.
#
# torch/torchvision resolve to the cu128 build straight from the lockfile:
# pyproject pins the PyTorch cu128 wheel index for Linux (`[tool.uv.sources]`),
# so `uv sync` installs the driver-correct build ONCE. There is no second
# `--reinstall` step and no throwaway PyPI/cu13 torch baked in a lower layer —
# that duplicate was ≈7 GB of the old image.
#
# Code is NOT baked: pods `git pull` the latest at launch, so only a dependency
# change requires a rebuild. Pods run training with `uv run --no-sync` to skip
# the (now redundant) resolve on boot — the lockfile already pins cu128, so this
# is a speed optimisation, not a correctness one.
FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04

# UV_NO_CACHE: never populate ~/.cache/uv, so no wheel cache bloats the layer
# (torch is multi-GB). UV_LINK_MODE=copy: the base image's / is a single fs, but
# copy avoids any cross-device hardlink warning during install.
ENV DEBIAN_FRONTEND=noninteractive \
    PATH="/root/.local/bin:${PATH}" \
    UV_NO_CACHE=1 \
    UV_LINK_MODE=copy

# openssh-server + rsync: the collect step rsyncs artifacts off the volume over
# SSH, so the container runs its own sshd (the CUDA base has neither).
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates build-essential openssh-server rsync \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /run/sshd

# runpodctl: a self-running pod removes ITSELF when its work is done (authing via
# the injected RUNPOD_API_KEY), so nothing depends on the laptop for teardown.
# uv manages Python + the locked dependency set. Both fetched in one layer.
RUN curl -fsSL https://github.com/runpod/runpodctl/releases/latest/download/runpodctl-linux-amd64 \
        -o /usr/local/bin/runpodctl \
    && chmod +x /usr/local/bin/runpodctl \
    && curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone once, bake the deps (skip the dev group = ruff/ty), and smoke-test the
# baked env — all in one layer so no intermediate cache survives. `uv sync`
# resolves the lockfile's cu128 torch directly (see header): a single torch
# install, driver-compatible with RunPod's CUDA 12.8. cu128 carries the
# recompile_limit rename, so no source patch is needed (unlike the cu124
# fallback for the older CUED drivers). The clone is throwaway — pods refresh it
# with `git pull` at launch. `import torch` verifies the env imports; CUDA
# availability is checked at runtime on a GPU host (the builder has no GPU).
WORKDIR /workspace
RUN git clone --depth 1 https://github.com/eoinmurray/pinglab.git \
    && cd pinglab \
    && uv sync --no-dev --locked \
    && uv run --no-sync python -c "import torch; print('baked torch', torch.__version__)" \
    && rm -rf /root/.cache
WORKDIR /workspace/pinglab

# Start script — two modes, chosen at runtime by the CELLS env var:
#   • CELLS set → fire-and-forget work. Start sshd in the background (for
#     debugging), check out the exact pinned commit ($PIN_SHA), then hand off to
#     experiments/${PINGLAB_POD_RUNNER}.py --pod-run (default exp022). Training
#     pods write to /shared/training; infer pods write to /shared/artifacts/<slug>.
#     Self-terminates when done; a backstop removes the pod after $MAX_RUNTIME.
#   • CELLS unset → just sshd in the foreground (the collect step / debugging).
# $PUBLIC_KEY / $PIN_SHA / $CELLS / $PINGLAB_POD_RUNNER / $MAX_RUNTIME at runtime.
RUN printf '%s\n' \
      '#!/bin/bash' \
      'mkdir -p /root/.ssh && chmod 700 /root/.ssh' \
      '[ -n "$PUBLIC_KEY" ] && echo "$PUBLIC_KEY" > /root/.ssh/authorized_keys && chmod 600 /root/.ssh/authorized_keys' \
      'ssh-keygen -A' \
      'if [ -z "$CELLS" ]; then exec /usr/sbin/sshd -D -e; fi' \
      '/usr/sbin/sshd' \
      'cd /workspace/pinglab' \
      'git fetch origin "$PIN_SHA" --depth 1 -q && git reset --hard "$PIN_SHA" -q' \
      '( sleep "${MAX_RUNTIME:-54000}"; runpodctl remove pod "$RUNPOD_POD_ID" ) &' \
      'RUNNER="${PINGLAB_POD_RUNNER:-exp022}"' \
      'exec uv run --no-sync python experiments/${RUNNER}.py --pod-run' \
    > /start.sh && chmod +x /start.sh

CMD ["/start.sh"]
