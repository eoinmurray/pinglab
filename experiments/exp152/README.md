# exp152 — Spiking control of a rotary joint

One Elevate neuron per flexor/extensor controller drives a simulated rotary
joint with a 2 g tip mass at 20 cm. Reciprocal inhibition and excitatory sensory
feedback couple three ROS2 Humble processes through complete logical spike
windows. Model and control steps are 1 ms; intrinsic neural steps are 0.025 ms.
The report separates repeatable movement from wall-clock timing success.

1. `bash experiments/exp152/custom/run.sh --compile --max-lag-ms 8000` builds
   the local image and acquires ROS data with the report's execution settings.
   Requires Docker/Colima with four CPUs. The recipe's ordinary lag stop remains
   20 ms; the displayed recording used 8000 ms and failed its timing checks.
   Optional arguments: `--duration` (30 s, minimum 2), `--max-lag-ms` (20),
   `--compile` / `--eager`, `--cycle-validation-source ANALYSE_ID`, `--run-id`.
   Set `SNNLAB_DOCKER_CONTEXT` to override `colima-snnlab-ros2`.
2. `uv run python experiments/exp152/compute.py --native --duration 40 --compile`
   reproduces the deterministic native validation from the fixed recipe, without
   requiring historical tuning runs. Separate controller states and fixed 1 ms
   transport are preserved. To select a recorded condition instead, replace
   `--native` with `--cycle-source ANALYSE_ID --condition-index INDEX`.
   Analyse the native compute, then pass its analysis to the ROS acquisition as
   `--cycle-validation-source ANALYSE_ID` for an exact trace comparison.
3. `uv run python experiments/exp152/analyse.py --source COMPUTE_ID` measures
   the explicit ROS acquisition or native validation. Optional: `--run-id`.
4. `uv run python experiments/exp152/present.py --source ANALYSE_ID` exports
   the animation data, native circuit diagram and report numbers from one ROS
   analysis. Optional: `--run-id`. Raw diagnostics, traces and native snapshots
   stay in the recorded ancestry.
5. `uv run pytest experiments/exp152/test.py -q` validates neural dynamics,
   mechanics, stream reset/continuation, cycle analysis and animation geometry.
   Run inside the image for ROS clock and reliable diagnostic-export coverage:

   ```sh
   docker --context colima-snnlab-ros2 run --rm \
     -e PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 -v "$PWD:/workspace" \
     --entrypoint /bin/bash pinglab-exp152:local -c \
     'source /opt/ros/humble/setup.bash && /opt/venv/bin/python -m pytest experiments/exp152/test.py -q'
   ```

6. `cd site && bun run dev --background` serves `/exp152`. Restart after a new
   presentation so the report, animation and parameter table select it together.
7. `uv run python experiments/exp152/compute.py --consolidate-source COMPUTE_ID`
   imports explicit fixed evidence without simulation. It preserves raw bytes,
   native snapshots and original execution timing, and embeds original lineage
   records, inventories and tuning summaries instead of retaining the historical
   runs as operational dependencies. First import and analyse the native compute;
   then import the ROS compute with `--cycle-validation-source NEW_ANALYSE_ID`
   and `--archive-source OLD_PRESENT_ID`. The replacement validation must match
   the original cycle results and native trace exactly. Analyse and present the
   imported ROS compute independently before pruning.
8. `uv run pingstore prune --experiment exp152 --dry-run` plans local cleanup;
   apply its exact plan with `--confirm PLAN_HASH`. The latest presentation's
   complete input ancestry is retained. Consolidation leaves five required runs:
   native compute/analysis and ROS compute/analysis/presentation.
   This does not delete R2 backups or rewrite completed runs.
9. `uv run pingstore prune-remote --experiment exp152 --dry-run` plans removal
   of unused R2 history after the current runset is backed up. Apply its exact
   plan with `--confirm PLAN_HASH`; required ancestry and other experiments are
   retained. Refresh any deployed report to the current presentation first.
10. `uv run python experiments/exp152/compute.py --adapt --compile` tests bounded
    online adaptation of tonic excitation and sensory weight in the native loop.
    Optional: `--target-frequency-hz` (0.65), `--target-excursion-deg` (60,
    peak-to-peak), `--adaptation-iterations` (16), `--eager`, `--run-id`.
    Baseline and signed probes start at 26 s, discard the first 15 s and require
    five repeatable cycles before their errors can guide updates. Unsettled
    segments extend in 5 s increments to 60 s; an unsettled baseline skips its
    update. Initial, frozen and fresh-reset validations each last 60 s.
    This makes no ROS timing claim.
    Analyse the explicit compute with `analyse.py --source COMPUTE_ID`, then
    use `present.py --source ANALYSE_ID --reference-present ROS_PRESENT_ID` to
    add the figures while retaining that explicit ROS replay and its snapshots.
