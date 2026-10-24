#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "$0")/../../.." && pwd)
context=${SNNLAB_DOCKER_CONTEXT:-colima-snnlab-ros2}
image=pinglab-exp152:local
docker --context "$context" build -t "$image" -f "$repo/experiments/exp152/Dockerfile" "$repo"
docker --context "$context" run --rm --init --cpus=4 --cpuset-cpus=0-3 --network=none \
  -e ROS_LOCALHOST_ONLY=1 -e ROS_DOMAIN_ID=73 \
  -v "$repo:/workspace" -v pinglab-exp152-torch-cache:/tmp/torchinductor_root \
  "$image" "$@"
