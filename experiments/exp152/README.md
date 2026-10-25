# exp152 — Spiking control of a rotary joint

Use custom Hodgkin–Huxley flexor/extensor controllers and a simulated joint in
independent Pingstore stages. The HH equations are adapted from SNNLab’s example,
with conductance inputs, unscaled channel kinetics and a 1 ms neural/ROS update.
A 200 ms flexor starting pulse reduces settling from a centred, stationary joint.
The neurons use SNNLab's `extensions.register_neuron` and `lang.CustomNeuron`,
with a PyTorch tensor callback performing RK4 steps no larger than 0.025 ms.
SNNLab supports eager execution or optional compilation of that callback.
Retain raw ROS diagnostics, substep voltage means/spike counts,
topic events, timing, native bundles/start snapshots and logs for a simpler PixiJS replay.

1. `bash experiments/exp152/custom/run.sh`: build the experiment-local Dockerfile
   and run compute with four CPUs in the existing `colima-snnlab-ros2` context.
   Override the context with `SNNLAB_DOCKER_CONTEXT`. Optional compute arguments:
   `--duration` (8 s, minimum 2), `--dt-ms` (1), `--timer-ms` (1), `--neurons`
   (1 per population, 4 total), `--max-lag-ms` (20), `--max-spike-age-ms` (5),
   `--flexor-bias-ms` (200), `--flexor-initial-us` (0.02),
   `--extensor-initial-us` (0.005), `--compile` (optional SNNLab compilation),
   `--startup-source ANALYSE_ID` (validated tuning ancestry),
   `--profile-timing`, `--run-id`. Start Docker/Colima first.
2. `uv run python experiments/exp152/analyse.py --source COMPUTE_ID`: independently
   validate and analyse one acquisition; optional `--run-id` and
   `--baseline-source ANALYSE_ID` for an explicit startup comparison. Behaviour and timing
   verdicts stay separate; completed acquisition is not a passing verdict.
3. `uv run python experiments/exp152/present.py --source ANALYSE_ID`: export
   `circuit.json`, `circuit.svg` when available, and `numbers.json`;
   optional `--run-id`. Full-resolution arrays remain in the analysis ancestry.
4. `uv run pytest experiments/exp152/test.py -q`: portable protocol/export checks.
5. ROS clock tests inside the built image: `docker --context colima-snnlab-ros2 run --rm -e PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 -v "$PWD:/workspace" --entrypoint /bin/bash pinglab-exp152:local -c 'source /opt/ros/humble/setup.bash && /opt/venv/bin/python -m pytest experiments/exp152/test.py -q'`.
6. `cd site && bun run dev --background`: open `/exp152` to replay the selected
   presentation. Restart the dev server after a new presentation to refresh its
   selection. No historical sibling-repository artifacts are imported.

7. `uv run python experiments/exp152/compute.py --startup-sweep`: record a bounded
   deterministic starting-pulse sweep with a one-step transport delay and a resting
   plant; requires 1 ms updates and one cell per population. Run `analyse.py` with
   its explicit compute ID to measure early swings, sensory engagement and settling.
   The sweep contains the original 20 ms pulse and the selected 200 ms pulse.
8. The tensor-callback acquisition uses `--compile --flexor-bias-ms 200
   --flexor-initial-us 0.02 --max-lag-ms 200 --startup-source exp152-r039-analyse`.
   The larger abort threshold
   allows recording through timing excursions; it does not alter 1 ms releases or
   excuse missed deadlines. The default abort threshold remains 20 ms.
   Settling means extrema and same-direction periods remain within 20% of late-run
   medians, with at least three extrema left to assess. The startup comparison is
   retained in the analysis and presentation `numbers.json`.
