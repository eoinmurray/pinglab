# syntax=docker/dockerfile:1
# Build from the repository root: docker build -f experiments/exp152/Dockerfile .
FROM ros:humble-ros-base-jammy
COPY --from=ghcr.io/astral-sh/uv:0.9.11 /uv /uvx /usr/local/bin/
ENV DEBIAN_FRONTEND=noninteractive MPLBACKEND=Agg PYTHONUNBUFFERED=1 \
    UV_PYTHON_DOWNLOADS=never OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ros-humble-rclpy ros-humble-std-msgs \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY experiments/exp152/custom/requirements.txt ./requirements.txt
RUN uv venv --python /usr/bin/python3 /opt/venv \
    && uv pip install --python /opt/venv/bin/python \
       torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu \
    && uv pip install --python /opt/venv/bin/python -r requirements.txt
RUN apt-get update && apt-get install -y --no-install-recommends g++ ninja-build util-linux \
    && rm -rf /var/lib/apt/lists/*
ENV TORCHINDUCTOR_COMPILE_THREADS=1 PYTHONPATH=/workspace:/workspace/tools
WORKDIR /workspace
# Mount the checkout at execution so Pingstore captures Git provenance and owns outputs.
ENTRYPOINT ["/bin/bash", "-c", "source /opt/ros/humble/setup.bash && exec /opt/venv/bin/python experiments/exp152/compute.py \"$@\"", "--"]
