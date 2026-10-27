"""Validate protocol resolution, continuation, clock scheduling and replay alignment.

ROS clock tests run in the experiment container and skip on hosts without ROS.
Portable tests exercise native controller state and lossless replay exports.
"""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from experiments.exp152.analyse import collect
from experiments.exp152.custom.models import Controller, Joint
from experiments.exp152.present import NODES, recording
from experiments.exp152.recipe import author_network, configuration


class ReplayTests(unittest.TestCase):
    def fixture(self):
        values = {}
        for name in NODES:
            for field, array in dict(
                time_s=[0.0011, 0.0032],
                callback_ms=[0.2, 0.3],
                lag_ms=[0.9, 0.9],
                lateness_ms=[0.1, 0.2],
                model_time_s=[0.001, 0.002],
                voltage=[[-65], [-63]],
                spikes=[[1], [0]],
            ).items():
                values[name + "_" + field] = np.asarray(array)
        values.update(
            angle=np.asarray([0.1, 0.2]),
            velocity=np.asarray([0.2, 0.3]),
            activation=np.asarray([[0.1, 0.2], [0.3, 0.4]]),
            gravity_torque=np.asarray([-0.003, -0.004]),
            sensory_flexor_time_s=np.asarray([]),
            sensory_extensor_time_s=np.asarray([]),
        )
        summary = dict(
            mode="ros2",
            configuration=configuration(duration=8),
            controller_populations=["motor"],
            duration_seconds=8,
            timer_ms=1,
            neurons_per_population=1,
            snnlab_ref="test",
            passed=False,
            timing_passed=False,
            stop_reasons=dict.fromkeys(NODES, "excessive_lag"),
            full_callback_ms=dict.fromkeys(NODES, dict(n=2, mean=0.25, sd=0.07)),
        )
        return values, summary

    def test_completion_times_and_short_but_late_callbacks_are_preserved(self):
        values, summary = self.fixture()
        values["sensory_flexor_time_s"] = np.asarray([0.0015])
        values["sensory_extensor_time_s"] = np.asarray([0.0025, 0.0035, 0.0045])
        payload = recording(values, summary)
        self.assertEqual(
            payload["sensory_time_ms"], dict(flexor=[1.5], extensor=[2.5, 3.5, 4.5])
        )
        for name in NODES:
            node = payload["nodes"][name]
            np.testing.assert_allclose(node["time_ms"], [1.1, 3.2])
            self.assertEqual(sum(v > 0 for v in node["lateness_ms"]), 2)
            self.assertTrue(all(v < payload["period_ms"] for v in node["callback_ms"]))
            self.assertEqual(node["model_s"][-1], 0.002)
        self.assertEqual(payload["duration_ms"], 8000)
        self.assertEqual(payload["controller_populations"], ["motor"])
        self.assertEqual(payload["schema"], "pinglab.exp152.replay/v2")
        self.assertEqual(len(payload["nodes"]["flexor"]["voltage"][0]), 1)
        self.assertEqual(payload["nodes"]["plant"]["angle"], [0.1, 0.2])

    def test_rejects_unsorted_completion_times(self):
        values, summary = self.fixture()
        values["flexor_time_s"] = np.asarray([0.003, 0.001])
        with self.assertRaisesRegex(ValueError, "Invalid callback timeline"):
            recording(values, summary)


class ClockTests(unittest.TestCase):
    def run_schedule(self, jump_ns):
        pytest.importorskip("rclpy")
        from experiments.exp152.custom import node_flexor as runtime

        state = SimpleNamespace(now=0, active=True)
        epoch = 1_000_000_000_000
        records = []

        class Executor:
            def add_node(self, node):
                pass

            def spin_once(self, timeout_sec):
                state.now += round(timeout_sec * 1e9)
                if records:
                    state.active = False

        class Node(runtime.RuntimeNode):
            def __init__(self):
                self.duration = 0.004
                self.capacity = 4
                self.period_ns = 1_000_000
                self.timer_ms = 1
                self.dt_ms = 1.0
                self.substeps = 1
                self.max_lag_ms = 20
                self.updates = self.steps = 0
                self.frame_wait_ms = np.zeros(4)
                self.frame_inbox = runtime.SpikeWindowInbox(["peer", "plant"], 1)
                for index in range(3):
                    for source in ("peer", "plant"):
                        self.frame_inbox.receive(source, [index, 1])
                self.frame_publisher = SimpleNamespace(publish=lambda msg: None)
                self.timing = np.zeros((4, 5))
                self.stamps = np.zeros(4, dtype=np.int64)
                self.schedule_time = np.zeros(4)
                self.clock_offset_change = np.zeros(4)
                self.clock_sample_span = np.zeros(4)
                self.clock = SimpleNamespace(now=self.ros_now)
                self.cpu_affinity = [0]
                self.finished = False
                self.deliveries = []
                self.diagnostics = SimpleNamespace(
                    publish=lambda msg: records.extend(json.loads(msg.data))
                )

            def ros_now(self):
                offset = jump_ns if self.updates >= 1 else 0
                return SimpleNamespace(nanoseconds=epoch + state.now + offset)

            def get_parameter(self, name):
                return SimpleNamespace(value=epoch + 11_000_000)

            def tick(self):
                # One callback misses its deadline; later callbacks catch up.
                state.now += 1_200_000 if self.updates == 1 else 200_000
                return 1.2 if self.updates == 1 else 0.2

            def diagnostic_values(self, index):
                return {}

        node = Node()
        with (
            patch.object(runtime, "SingleThreadedExecutor", Executor),
            patch.object(runtime.rclpy, "ok", lambda: state.active),
            patch.object(runtime.time, "perf_counter_ns", lambda: state.now),
        ):
            node.execute()
        self.assertEqual(node.stop_reason, "completed")
        self.assertEqual(node.steps, 4)
        self.assertEqual(len(records), 5)
        ticks = records[:-1]
        np.testing.assert_allclose([r["wall_lag_ms"] for r in ticks], [0, 0, 0.2, 0])
        np.testing.assert_allclose(
            [r["deadline_lateness_ms"] for r in ticks], [-0.8, 0.2, -0.6, -0.8]
        )
        np.testing.assert_allclose(
            [r["schedule_time_s"] for r in ticks], [0.0002, 0.0022, 0.0024, 0.0032]
        )
        np.testing.assert_allclose(
            [r["clock_offset_change_ms"] for r in ticks], [0] + [jump_ns / 1e6] * 3
        )
        self.assertEqual(ticks[0]["clock_mapping"]["sample_span_ns"], 0)
        return node.timing

    def test_ros_clock_jumps_do_not_change_schedule_or_deadlines(self):
        baseline = self.run_schedule(0)
        for jump in (-108_000_000, 108_000_000):
            with self.subTest(jump=jump):
                np.testing.assert_array_equal(self.run_schedule(jump), baseline)


@pytest.mark.parametrize(
    "settings",
    [
        dict(dt_ms=0.3),
        dict(timer_ms=0.15),
        dict(duration=float("nan")),
        dict(neurons=1.5),
        dict(neurons=2),
        dict(dt_ms=0.5),
        dict(timer_ms=2),
        dict(max_lag_ms=0),
        dict(controller_warmup_ms=0),
        dict(controller_warmup_ms=float("nan")),
    ],
)
def test_invalid_protocol(settings):
    with pytest.raises(ValueError):
        configuration(**settings)


def test_fresh_configuration():
    cfg = configuration()
    cfg["joint"]["inertia"] = 99
    assert configuration()["joint"]["inertia"] == 0.005


def test_native_continuation_and_reset():
    controller = Controller("flexor", compile_mode=None)
    controller.window(1, sensory=1)
    state = controller.stream.export_state()
    step = controller.steps
    first = controller.window(1, peer=1)
    controller.stream.reset(state)
    controller.steps = step
    second = controller.window(1, peer=1)
    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left, right)
    controller.stream.close()


def test_long_warmup_resets_state_inputs_and_simulation_clock():
    controllers = [
        Controller(
            "flexor",
            configuration_overrides={"controller_warmup_ms": duration},
        )
        for duration in (2.0, 1000.0)
    ]
    try:
        for controller, windows in zip(controllers, (2, 1000), strict=True):
            assert controller.steps == 0
            assert controller.info["warmup_windows"] == windows
            assert controller.info["warmup_state_reset"] is True
            assert controller.info["warmup_model_ms"] == windows
            assert all(not value.any().item() for value in controller.inputs.values())
        for tick in range(40):
            inputs = dict(sensory=int(tick % 11 == 0), peer=int(tick % 19 == 0))
            short, long = [controller.window(1, **inputs) for controller in controllers]
            np.testing.assert_array_equal(short[0], long[0])
            np.testing.assert_array_equal(short[1], long[1])
            assert short[2] == long[2]
    finally:
        for controller in controllers:
            controller.stream.close()


@pytest.mark.parametrize(
    "neurons,length,firing",
    [(1, 1, False), (1, 1, True)],
)
def test_output_extraction_preserves_values_events_and_array_ownership(
    neurons, length, firing
):
    """Compare the fast path with the former reduction and reused native outputs."""
    import torch

    voltage = np.arange(length * neurons, dtype=float).reshape(length, neurons) - 65
    spikes = (
        (np.arange(length * neurons).reshape(length, neurons) % 3 == 0) & firing
    ).astype(float)
    expected_voltage = np.column_stack(
        [voltage[:, 0] if neurons == 1 else voltage.mean(axis=1)]
    )
    expected_spikes = np.column_stack(
        [spikes[:, 0] if neurons == 1 else spikes.sum(axis=1)]
    ).astype(int)
    expected_events = np.argwhere(spikes != 0).tolist()
    outputs = dict(
        motor_voltage=torch.tensor(voltage[:, None]),
        motor_spikes=torch.tensor(spikes[:, None]),
    )
    controller = object.__new__(Controller)
    controller.torch, controller.steps, controller.neurons = torch, 0, neurons
    controller.inputs = {}
    controller.stream = SimpleNamespace(
        advance=lambda _inputs: SimpleNamespace(outputs=outputs)
    )
    actual_voltage, actual_spikes, actual_events = controller._execute(length)
    np.testing.assert_array_equal(actual_voltage, expected_voltage)
    np.testing.assert_array_equal(actual_spikes, expected_spikes)
    assert actual_events == expected_events
    assert controller.steps == length
    # Simulate streaming reusing its output tensors on a subsequent advance.
    outputs["motor_voltage"].fill_(999)
    outputs["motor_spikes"].fill_(0)
    np.testing.assert_array_equal(actual_voltage, expected_voltage)
    np.testing.assert_array_equal(actual_spikes, expected_spikes)


def test_joint_order_and_normalized_impulse():
    joint = Joint()
    torque = joint.advance([1, 0], 0.0001)
    assert torque == pytest.approx(
        0.8 * configuration()["joint"]["activation_increment"]
    )
    acceleration = (torque - 0.002 * 9.81 * 0.2) / 0.00508
    assert joint.velocity == pytest.approx(0.0001 * acceleration)
    assert joint.angle == pytest.approx(np.pi / 2 + 1e-8 * acceleration)


def test_native_start_snapshot(tmp_path):
    import torch
    from experiments.exp152.custom.snapshot import capture_start
    from snnlab.sim import Performance, prepare_stream

    cfg = configuration()
    bundle = author_network(cfg)
    stream = prepare_stream(
        bundle,
        window_steps=1,
        dtype=torch.float64,
        device="cpu",
        performance=Performance(compile_mode=None),
    )
    capture_start(stream, bundle, tmp_path / "flexor", cfg)
    snapshot = json.loads((tmp_path / "flexor/start/snapshot.json").read_text())
    assert snapshot["stage"] == "start"
    assert {row["id"] for row in snapshot["network"]["populations"]} == {
        "motor",
    }
    assert (tmp_path / "flexor/circuit.svg").is_file()
    stream.close()


def test_empty_failed_acquisition_retains_verdicts():
    cfg = configuration()
    args = SimpleNamespace(
        **cfg,
        cpu_allocation=dict(flexor=0, extensor=1, plant=2, recorder=3),
        observed_recorder_affinity=[3],
    )
    values, summary = collect(
        dict.fromkeys(NODES, []),
        {
            f"/{kind}/{side}": []
            for kind in ("motor", "sensory")
            for side in ("flexor", "extensor")
        },
        0,
        args,
        False,
        "startup failure",
    )
    assert summary["total_neurons"] == 2
    assert not summary["passed"]
    assert not summary["timing_passed"]
    assert values["flexor_substep_spikes"].shape == (0, cfg["substeps"], 1)
    assert (
        recording(values, dict(configuration=cfg, **summary))["nodes"]["plant"][
            "time_ms"
        ]
        == []
    )


def test_browser_pulses_reach_targets_without_reset_and_playback_loops():
    """A later event must not erase an earlier spike halfway along its path."""
    import shutil
    import subprocess
    from pathlib import Path

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the browser animation regression")
    source = Path(__file__).parent / "report/js/replay.js"
    script = r"""
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
const code = readFileSync(process.argv[1], 'utf8');
const {pulseSamples, routePoint, loopTime, toDegrees, voltageFraction, peerInhibitionRoutes, sensoryExcitationRoutes, controllerPopulations, arrowHead} = await import(
  'data:text/javascript;base64,' + Buffer.from(code).toString('base64'));
assert.deepEqual(controllerPopulations({schema:'pinglab.exp152.replay/v2', controller_populations:['motor'], neuron_model:'elevate', neurons_per_population:1, mechanics:{angle_convention:'zero_downward_positive_right'}}), ['motor']);
assert.throws(() => controllerPopulations({schema:'pinglab.exp152.replay/v1'}));
assert.throws(() => controllerPopulations({schema:'pinglab.exp152.replay/v2'}));
assert.equal(sensoryExcitationRoutes().length, 2);
for (const route of sensoryExcitationRoutes()) {
  assert.equal(route.inhibitory, false);
  assert.equal(route.sensory, true);
  const center = route.name === 'flexor' ? [205,240] : [1035,240];
  const end = route.points.at(-1);
  assert.ok(Math.hypot(end[0]-center[0],end[1]-center[1]) > 27);
}
const [forward, backward] = peerInhibitionRoutes();
for (const route of [forward, backward]) {
  const end = route.points.at(-1);
  const center = route.name === 'flexor' ? [1035, 240] : [205, 240];
  assert.ok(Math.hypot(end[0] - center[0], end[1] - center[1]) > 34);
}
const triangle = arrowHead([[0, 0], [100, 0]]);
assert.deepEqual(triangle, [100, 0, 89, 5, 89, -5]);
for (let step = 0; step <= 100; step++) {
  const t = step / 100;
  const a = routePoint(forward.points, t), b = routePoint(backward.points, 1 - t);
  assert.ok(Math.abs(a[0] - b[0]) < 1e-9);
  assert.ok(a[1] < b[1], 'peer routes must remain vertically separated');
}
assert.equal(toDegrees(Math.PI), 180);
assert.equal(toDegrees(Math.PI / 2), 90);
assert.deepEqual([-80, -20, 40, null].map(v => voltageFraction(v)), [0, .5, 1, 0]);
assert.deepEqual([-80, -20, 40].map(v => voltageFraction(v, -80, 40)), [0, .5, 1]);
const pulses = pulseSamples([0, 10, 20], 29);
assert.equal(pulses.length, 3);
assert.ok(pulses[0].progress > .95);
assert.equal(pulses[0].opacity, 1);
const arrived = pulseSamples([0, 10, 20, 30], 31)[0];
assert.equal(arrived.progress, 1);
assert.ok(arrived.opacity > .8);
for (const path of [[[2, 4], [80, 20]], [[2, 4], [10, 80], [60, 70], [80, 20]]]) {
  assert.deepEqual(routePoint(path, 0), path[0]);
  assert.deepEqual(routePoint(path, arrived.progress), path.at(-1));
}
assert.deepEqual(pulseSamples([0], 39), []);
assert.equal(loopTime(7999, 8, .25, 8000), 1);
assert.equal(loopTime(7990, 10, 1, 8000), 0);
assert.equal(loopTime(0, -.1, 1, 8000), 0);
"""
    subprocess.run(
        [node, "--input-type=module", "-e", script, str(source)],
        check=True,
        capture_output=True,
        text=True,
    )


def test_elevate_converges_to_independent_scipy_ode():
    """Check the complete six-state continuous model with an adaptive solver."""
    import torch
    from experiments.exp152.custom.elevate import KEYS, initialize, step
    from scipy.integrate import solve_ivp
    from scipy.special import expit

    cfg = configuration()["neuron"]

    def derivative(_time, state):
        v, filtered = state[0], state[1:]
        m, h, n, ca, kca = expit(
            (filtered + [25, 40, 15, 45, 30]) * [0.2, -0.1, 0.1, 0.2, 0.1]
        )
        current = (
            100 * m * h * (40 - v)
            + (65 * n + 10 * kca) * (-90 - v)
            + 2 * ca * (120 - v)
            + 2 * (-50 - v)
            - 5
        )
        return np.r_[current, (v - filtered) / [0.1, 1, 1, 1, 75]]

    truth = solve_ivp(
        derivative,
        (0, 2),
        np.full(6, -60.0),
        method="DOP853",
        rtol=1e-11,
        atol=1e-12,
        max_step=0.001,
    ).y[:, -1]
    errors = []
    for dt in [0.04, 0.02, 0.01, 0.005]:
        state = initialize(
            SimpleNamespace(shape=(1, 1), device="cpu", dtype=torch.float64, config=cfg)
        )
        for _ in range(round(2 / dt)):
            state, spikes = step(
                SimpleNamespace(
                    state=state,
                    dt_ms=dt,
                    config={**cfg, "max_step_ms": dt},
                    excitatory=torch.zeros((1, 1), dtype=torch.float64),
                    inhibitory=torch.zeros((1, 1), dtype=torch.float64),
                )
            )
            assert spikes.item() in (0, 1)
        errors.append(
            np.linalg.norm(np.array([state[key].item() for key in KEYS]) - truth)
        )
    assert all(b < 0.7 * a for a, b in zip(errors, errors[1:], strict=False)), errors
    assert errors[-1] < 0.03


def test_elevate_definition_has_conductance_ports_and_gate_state():
    cfg = configuration()
    bundle = author_network(cfg)
    assert cfg["neuron_model"] == "elevate"
    populations = bundle.graph["populations"]
    assert len(populations) == 1
    assert populations[0]["id"] == "motor"
    assert populations[0]["size"] == 1
    assert populations[0]["neuron"]["kind"] == "custom_neuron"
    assert {row["id"] for row in bundle.graph["projections"]} == {
        "local_bias",
        "sensory_drive",
        "peer_inhibition",
    }


def test_elevate_compiled_controller_matches_eager_with_spike_inputs():
    """Compilation must preserve voltage, spikes and their within-window offsets."""
    eager = Controller("flexor", dt_ms=1.0, warm_steps=1, compile_mode=None)
    compiled = Controller("flexor", dt_ms=1.0, warm_steps=1, compile_mode="default")
    try:
        for tick in range(500):
            inputs = dict(sensory=int(tick % 17 == 0), peer=int(tick % 29 == 0))
            expected = eager.window(1, **inputs)
            actual = compiled.window(1, **inputs)
            # Kernel fusion changes floating-point operation order. Keep voltage
            # agreement below 0.001 mV and require exact event identity/offsets.
            np.testing.assert_allclose(actual[0], expected[0], atol=1e-3, rtol=0)
            np.testing.assert_array_equal(actual[1], expected[1])
            assert actual[2] == expected[2]
    finally:
        eager.stream.close()
        compiled.stream.close()


def test_elevate_closed_loop_reaches_both_sensors_without_joint_stops():
    """A deterministic one-window transport delay isolates the neural controller."""
    cfg = configuration(duration=30)
    controllers = [
        Controller(side, dt_ms=cfg["dt_ms"], warm_steps=cfg["substeps"])
        for side in ("flexor", "extensor")
    ]
    joint = Joint()
    motor, sensory = np.zeros(2), np.zeros(2, dtype=int)
    sensory_counts = np.zeros(2, dtype=int)
    late_sensory_counts = np.zeros(2, dtype=int)
    velocities, angles = [], []
    try:
        for tick in range(round(cfg["duration"] * 1000 / cfg["timer_ms"])):
            output = [
                controller.window(
                    cfg["substeps"], sensory=int(sensory[i]), peer=int(motor[1 - i])
                )
                for i, controller in enumerate(controllers)
            ]
            for substep in range(cfg["substeps"]):
                now = (tick * cfg["timer_ms"] + substep * cfg["dt_ms"]) / 1000
                sensory = joint.encode(now)
                sensory_counts += sensory
                if now >= cfg["duration"] / 2:
                    late_sensory_counts += sensory
                joint.advance(motor if substep == 0 else [0, 0], cfg["dt_ms"] / 1000)
            motor = np.array([row[1][:, 0].sum() for row in output])
            angles.append(joint.angle)
            velocities.append(joint.velocity)
        moving = np.sign(np.asarray(velocities)[np.abs(velocities) > 0.05])
        assert np.isfinite(angles).all()
        assert joint.limit_hits == 0
        # The unchanged cells must repeatedly drive both sides under gravity.
        tail = np.asarray(angles)[4000:]
        assert np.ptp(tail) > 2 * cfg["joint"]["sensory_limit"]
        assert (sensory_counts > 0).all(), (sensory_counts, min(angles), max(angles))
        assert (late_sensory_counts > 0).all(), late_sensory_counts
        from experiments.exp152.analyse import cycle_metrics

        assert cycle_metrics(
            angles,
            velocities,
            reference_angle=cfg["joint"]["reference_angle"],
            settings=cfg["analysis"],
        )["passed"]
        assert np.count_nonzero(np.diff(moving)) >= cfg["analysis"]["minimum_reversals"]
    finally:
        for controller in controllers:
            controller.stream.close()


def test_one_ms_protocol_has_no_neuron_clock_scaling():
    cfg = configuration()
    assert cfg["dt_ms"] == cfg["timer_ms"] == 1.0
    for population in author_network(cfg).graph["populations"]:
        assert "time_scale" not in population["neuron"]["config"]


@pytest.mark.parametrize("side", ["flexor", "extensor"])
def test_starting_pulse_replaces_tonic_then_restores_it(side):
    options = dict(flexor_bias_ms=20, flexor_initial_us=0.5, extensor_initial_us=0)
    controller = Controller(side, compile_mode=None, configuration_overrides=options)
    joint = Joint()
    assert joint.angle == np.pi / 2
    assert joint.velocity == 0
    np.testing.assert_array_equal(joint.activation, [0, 0])
    try:
        for tick in range(30):
            controller.window(1)
            expected = (
                options[side + "_initial_us"]
                if tick < 20
                else configuration()["tonic_us"]
            )
            assert controller.inputs["tonic"].item() == expected
    finally:
        controller.stream.close()


def test_loaded_joint_physics_against_independent_ode():
    from scipy.integrate import solve_ivp

    settings = configuration()["joint"]
    assert settings["total_inertia"] == pytest.approx(0.00508)
    joint = Joint(settings)
    assert joint.gravity_torque(np.pi / 2) == pytest.approx(-0.003924)
    assert joint.gravity_torque(0) == 0
    assert joint.gravity_torque(-np.pi / 2) == pytest.approx(0.003924)

    def rhs(time, state):
        angle, velocity = state
        return [
            velocity,
            (-0.003924 * np.sin(angle) - 0.02 * velocity - 0.02 * (angle - np.pi / 2))
            / 0.00508,
        ]

    expected = solve_ivp(rhs, [0, 0.1], [np.pi / 2, 0], rtol=1e-11, atol=1e-13)
    for _ in range(1000):
        joint.advance([0, 0], 0.0001)
    np.testing.assert_allclose(
        [joint.angle, joint.velocity], expected.y[:, -1], atol=5e-5
    )
    assert joint.limit_hits == 0
    # Equal opposing activation has unequal acceleration about the horizontal.
    raised, lowered = (
        Joint(dict(settings, activation_increment=0.01)),
        Joint(dict(settings, activation_increment=0.01)),
    )
    raised.advance([20, 0], 0.001)
    lowered.advance([0, 20], 0.001)
    assert raised.velocity > 0 > lowered.velocity
    assert abs(lowered.velocity) > abs(raised.velocity)


def test_joint_configuration_and_relative_sensors():
    cfg = configuration(initial_angle_deg=70, tip_mass_kg=0.1, arm_length_m=0.3)
    joint = Joint(cfg["joint"])
    assert joint.angle == pytest.approx(np.deg2rad(70))
    assert joint.total_inertia == pytest.approx(0.014)
    np.testing.assert_array_equal(joint.encode(0), [0, 0])
    joint.angle -= 0.26
    np.testing.assert_array_equal(joint.encode(0), [1, 0])
    joint.angle += 0.52
    np.testing.assert_array_equal(joint.encode(0.01), [0, 1])
    with pytest.raises(ValueError):
        configuration(tip_mass_kg=-1)
    with pytest.raises(ValueError):
        configuration(arm_length_m=0)
    with pytest.raises(ValueError):
        Joint(dict(cfg["joint"], initial_angle=10))


def test_replay_preserves_loaded_mechanics_and_rejects_other_models():
    values, summary = ReplayTests().fixture()
    settings = summary["configuration"]["joint"]
    values["gravity_torque"] = np.array([-0.003, -0.004])
    payload = recording(values, summary)
    assert payload["mechanics"] == settings
    assert payload["mechanics"]["initial_angle"] == np.pi / 2
    assert payload["nodes"]["plant"]["gravity_torque"] == [-0.003, -0.004]
    summary["configuration"]["neuron_model"] = "unsupported"
    with pytest.raises(ValueError, match="single-cell Elevate"):
        recording(values, summary)


def test_cycle_metrics_accepts_periodic_asymmetry_and_rejects_drift():
    from experiments.exp152.analyse import cycle_metrics

    time = np.arange(1, 20001) / 1000
    phase = 2 * np.pi * time
    reference = np.pi / 2
    angle = reference + 0.3 * np.sin(phase) + 0.03 * (np.cos(2 * phase) - 1)
    velocity = 2 * np.pi * (0.3 * np.cos(phase) - 0.06 * np.sin(2 * phase))
    result = cycle_metrics(angle, velocity, reference_angle=reference)
    assert result["passed"]
    assert result["assessed_cycles"] == 10
    assert result["mean_period_s"] == pytest.approx(1, abs=1e-5)
    assert not cycle_metrics(
        np.full_like(time, reference), np.zeros_like(time), reference_angle=reference
    )["passed"]
    drift = reference + (0.3 + 0.01 * time) * np.sin(phase)
    assert not cycle_metrics(drift, velocity, reference_angle=reference)["passed"]
    assert not cycle_metrics(angle[:6000], velocity[:6000], reference_angle=reference)[
        "passed"
    ]


def test_recorded_controller_settings_survive_runtime_adapter():
    from experiments.exp152.recipe import controller_settings

    cfg = configuration(
        duration=40,
        tonic_us=0.06,
        sensory_drive_us=0.075,
        peer_inhibition_us=0.3,
        peer_tau_ms=150,
        activation_increment=0.004,
    )
    restored = configuration(**controller_settings(cfg))
    for key in ("duration", "tonic_us", "weights", "synapse_tau_ms", "joint"):
        assert restored[key] == cfg[key]


def test_spike_windows_wait_for_silence_and_apply_only_the_requested_step():
    from experiments.exp152.custom.models import SpikeWindowInbox

    inbox = SpikeWindowInbox(["peer", "plant"], 1)
    assert inbox.take(-1) == dict(peer=[], plant=[])
    inbox.receive("peer", [1, 1, 0, 0])
    inbox.receive("plant", [0, 1])
    assert inbox.take(0) is None
    inbox.receive("peer", [0, 1, 0, 0])
    assert inbox.take(0) == dict(peer=[(0, 0)], plant=[])
    assert inbox.take(1) is None
    inbox.receive("plant", [1, 1, 0, 1])
    assert inbox.take(1) == dict(peer=[(0, 0)], plant=[(0, 1)])
    inbox.receive("peer", [0, 1, 0, 0])
    assert not inbox.pending["peer"]
    with pytest.raises(ValueError):
        inbox.receive("peer", [2, 1, 1, 0])


def test_reliable_diagnostics_keep_all_batches_beyond_previous_history_depth():
    import time

    rclpy = pytest.importorskip("rclpy")
    from experiments.exp152.compute import _recorder_type
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from std_msgs.msg import Int64MultiArray, String

    qos = QoSProfile(
        history=HistoryPolicy.KEEP_ALL, reliability=ReliabilityPolicy.RELIABLE
    )
    rclpy.init()
    recorder = _recorder_type(
        Node=Node, String=String, Int64MultiArray=Int64MultiArray, diagnostic_qos=qos
    )()
    sender = Node("test_diagnostic_export")
    publisher = sender.create_publisher(String, "/diagnostics/flexor", qos)
    try:
        until = time.monotonic() + 10
        while publisher.get_subscription_count() == 0 and time.monotonic() < until:
            rclpy.spin_once(recorder, timeout_sec=0.01)
        assert publisher.get_subscription_count() > 0
        for index in range(400):
            publisher.publish(String(data=json.dumps([dict(index=index)])))
        while len(recorder.records["flexor"]) < 400 and time.monotonic() < until:
            rclpy.spin_once(recorder, timeout_sec=0.01)
        assert recorder.records["flexor"] == [dict(index=i) for i in range(400)]
    finally:
        sender.destroy_node()
        recorder.destroy_node()
        rclpy.shutdown()


def test_cycle_metrics_keeps_model_and_wall_clock_variation_separate():
    from experiments.exp152.analyse import cycle_metrics

    time = np.arange(1, 20001) / 1000
    angle, velocity = (
        0.3 * np.sin(2 * np.pi * time),
        0.6 * np.pi * np.cos(2 * np.pi * time),
    )
    assert cycle_metrics(angle, velocity)["passed"]
    irregular_wall_time = time + 0.1 * np.sin(2.4 * time)
    assert np.all(np.diff(irregular_wall_time) > 0)
    assert not cycle_metrics(angle, velocity, times=irregular_wall_time)["passed"]
