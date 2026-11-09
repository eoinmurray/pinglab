# exp152 — Spiking control of a rotary joint

Control a rotary joint using separate flexor, extensor and plant ROS2 processes. Record
neural activity, joint motion and execution timing; the streaming controller is not hard
real-time.

## Run

From the repository root, run these stages in order.
Compute runs a ROS2 acquisition and requires explicit authorisation.

```sh
bash experiments/exp152/custom/run.sh --duration 30 --max-lag-ms 8000 --compile
uv run python experiments/exp152/analyse.py --source COMPUTE_ID
uv run python experiments/exp152/present.py --source ANALYSE_ID
```

Replace `COMPUTE_ID` and `ANALYSE_ID` with the completed run IDs
from the preceding stages.

Acquisition requires Docker/Colima with four CPUs. `SNNLAB_DOCKER_CONTEXT` overrides the
default `colima-snnlab-ros2` context.

To run automated checks:

```sh
uv run pytest experiments/exp152/test.py -q
```

## Folder guide

1. `recipe.py` — configuration and model definition.
2. `compute.py` — acquire ROS2 recordings and save evidence.
3. `analyse.py` — calculate measurements from saved evidence.
4. `present.py` — generate figures and report assets.
5. `custom/` — experiment-specific models, helpers and execution tools.
6. `Dockerfile` — container environment for ROS2 acquisition.
7. `report/` — report, appendix and supporting components.
8. `test.py` — automated checks.
