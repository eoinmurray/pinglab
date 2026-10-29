# exp152 — Spiking control of a rotary joint

Reproduce the report's three-node ROS2 Humble recording: separate flexor,
extensor and plant processes, one Elevate neuron per controller, a 2 g tip mass
at 20 cm, and 1 ms model/control windows. The default acquisition is 30 s with
compiled SNNLab streams and an 8000 ms lag stop. The lag allowance permits
observing slow execution; deadline checks still use 1 ms.

1. `bash experiments/exp152/custom/run.sh --duration 30 --max-lag-ms 8000 --compile`
   builds the image and acquires the recording. Requires Docker/Colima with
   four CPUs. Optional: `--duration` (30 s, minimum 2), `--max-lag-ms` (8000),
   `--compile` (enabled by default), `--run-id`. Set `SNNLAB_DOCKER_CONTEXT`
   to override `colima-snnlab-ros2`. Scientific model settings live in `recipe.py`.
2. `uv run python experiments/exp152/analyse.py --source COMPUTE_ID` measures
   the explicitly selected acquisition and exports report traces and verdicts.
   Optional: `--run-id`.
3. `uv run python experiments/exp152/present.py --source ANALYSE_ID` exports
   replay data, the controller circuit diagram and report numbers. Optional:
   `--run-id`. Each stage prints its run ID; pass it to the next stage explicitly.
4. `uv run pytest experiments/exp152/test.py -q` validates neural dynamics,
   mechanics, stream continuation/reset, cycle checks, replay and CLI arguments.
   Run the same tests in the image for ROS clock and diagnostic-export coverage:

   ```sh
   docker --context colima-snnlab-ros2 run --rm --network=none \
     --cpus=4 --cpuset-cpus=0-3 -e PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
     -v "$PWD:/workspace" --entrypoint /bin/bash pinglab-exp152:local -c \
     'source /opt/ros/humble/setup.bash && /opt/venv/bin/python -m pytest experiments/exp152/test.py -q'
   ```

5. `cd site && bun run dev --background` serves `/exp152`. Reload after a new
   presentation so plots, report numbers and native parameters update together.
