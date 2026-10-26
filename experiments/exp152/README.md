# exp152 — Spiking control of a rotary joint

Use one Elevate neuron per flexor/extensor controller, reciprocal inhibitory
spike connections and excitatory joint feedback. The third ROS2 node simulates
a rotary joint with gravity and a tip point mass. The validated neuron and synaptic settings use
0.025 ms internal exponential steps, a 20 ms starting E pulse and muscle spike
increment 0.003, ongoing E conductance 0.04 µS and sensory increments 0.035 µS. Default external updates remain 1 ms, from a stationary
90° joint (zero downward). A 2 g mass at 20 cm adds 0.00008 kg m² inertia and
0.003924 N m peak gravity torque; sensory centre, spring rest and ±1 rad stops
are relative to 90°. Flexor raises against gravity, extensor lowers with it. Require ten repeatable cycles after startup; record cycle, behaviour and timing verdicts independently. Historical HH callback
code remains available for loading immutable historical native bundles.

1. `bash experiments/exp152/custom/run.sh`: build the local Dockerfile and run
   compute with four CPUs in `colima-snnlab-ros2`. Override the context with
   `SNNLAB_DOCKER_CONTEXT`. Optional arguments: `--duration` (30 s, minimum 2),
   `--dt-ms` (1), `--timer-ms` (1), `--neurons` (1 per node, 2 total; larger
   values replicate the single population for load testing), `--max-lag-ms`
   (20), `--max-spike-age-ms` (5), `--flexor-bias-ms` (20),
   `--flexor-initial-us` (0.5), `--extensor-initial-us` (0), `--compile`,
   `--startup-source ANALYSE_ID`, `--controller-warmup-ms` (1000 ms simulated
   before arming ROS, then all controller state and inputs reset),
   `--profile-timing`, `--initial-angle-deg` (90; also sets the spring/sensory
   reference), `--tip-mass-kg` (0.002), `--arm-length-m` (0.2),
   `--gravity-m-s2` (9.81), `--run-id`.
   Start Docker/Colima first. Parameter-selection ancestry is an explicit,
   validated Pingstore input; execution imports only experiment-local source.
2. `uv run python experiments/exp152/analyse.py --source COMPUTE_ID`: analyse
   one explicit acquisition; optional `--run-id`, `--baseline-source ANALYSE_ID`.
   A completed acquisition may contain safety stops or failed timing checks.
3. `uv run python experiments/exp152/present.py --source ANALYSE_ID`: export
   the recorded circuit replay, native circuit SVG and numbers; optional
   `--run-id`. Full-resolution traces remain in its analysis ancestry.
4. `uv run pytest experiments/exp152/test.py -q`: native dynamics, continuity,
   snapshot, closed-loop, compiled/eager fidelity and animation checks.
5. Run the same tests inside the built image for ROS clock coverage:
   `docker --context colima-snnlab-ros2 run --rm -e PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
   -v "$PWD:/workspace" --entrypoint /bin/bash pinglab-exp152:local -c
   'source /opt/ros/humble/setup.bash && /opt/venv/bin/python -m pytest
   experiments/exp152/test.py -q'`.
6. `cd site && bun run dev --background`: open `/exp152`. Restart the server
   after a new presentation. The replay shows one cell per side, reciprocal I,
   muscle activation, sensory E events, two voltage traces and all callback timings.
7. `uv run python experiments/exp152/compute.py --startup-sweep`: compare four
   bounded starting pulses with deterministic 1 ms spike transport; requires
   one cell per node and 1 ms updates. Analyse its explicit compute ID separately.
8. To inspect timing failures over a longer acquisition, an explicit
   `--max-lag-ms` override can retain more diagnostics. It is recorded in the
   configuration and does not change releases, deadlines or the timing verdict.
   Such a diagnostic recording must not be presented as meeting the default
   20 ms lag stop or completing all requested model windows.

9. `uv run python experiments/exp152/compute.py --load-sweep`: compare tip
   masses 1, 2, 5, 10 and 50 g at the recorded arm length and gravity, with
   identical neural settings and deterministic 1 ms transport. Analyse the
   explicit compute ID separately; `startup.json` contains each condition's
   recorded mechanics, stops, extrema, spike counts and direction reversals.

10. `uv run python experiments/exp152/compute.py --cycle-sweep --cycle-grid synaptic --duration 30`:
    screen 84 paired-controller conditions in independent native SNNLab batches.
    The `actuation` grid screens 147 conditions around the periodic operating point.
    Input gains represent linear synaptic weight changes, including the native
    initializer's float32 constants before float64 execution. These are screening
    runs; a chosen condition requires individual native-weight validation.
11. `uv run python experiments/exp152/compute.py --cycle-source ANALYSE_ID --condition-index INDEX --duration 40`:
    validate one explicitly selected cycle condition with two separate native
    controllers and fixed 1 ms delivery. Add `--compile` to validate the compiled
    execution used by ROS. Analyse its compute ID independently.
    Success requires the final ten complete upward reference-angle crossings
    after 5 s to have <1% period and turning-point variation and <1% normalized
    angle/velocity RMS waveform error, both late sensory streams, excursion
    beyond both thresholds, and zero mechanical stop hits.
12. Neural and actuator overrides: `--tonic-us` (0.04), `--sensory-drive-us`
    (0.035), `--peer-inhibition-us` (0.2), `--sensory-tau-ms` (10),
    `--peer-tau-ms` (50), `--activation-increment` (0.003).
    `--cycle-validation-source ANALYSE_ID` requires matching settings from a
    passing individual validation. The default recipe names its validation;
    exploratory overrides retain selection ancestry without inheriting its pass.

13. `--spike-transport logical` (default) exchanges complete sparse spike windows,
    including silence, and applies the preceding window to each model step.
    Nodes wait for missing windows while nominal 1 ms wall-clock deadlines keep
    running. The model may finish later than its requested simulation duration;
    lag still stops execution at the recorded budget. `arrival` preserves the
    earlier callback-arrival semantics. Ordinary spike topics retain delivery
    diagnostics in both modes. Post-run diagnostic batches use reliable KEEP_ALL
    queues; sequence validation fails incomplete diagnostic exports.
