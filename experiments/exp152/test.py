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
                voltage=[[-65, -64], [-63, -62]],
                spikes=[[1, 0], [0, 1]],
            ).items():
                values[name + "_" + field] = np.asarray(array)
        values.update(
            angle=np.asarray([0.1, 0.2]),
            velocity=np.asarray([0.2, 0.3]),
            activation=np.asarray([[0.1, 0.2], [0.3, 0.4]]),
        )
        summary = dict(
            mode="ros2",
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
        payload = recording(values, summary)
        for name in NODES:
            node = payload["nodes"][name]
            np.testing.assert_allclose(node["time_ms"], [1.1, 3.2])
            self.assertEqual(sum(v > 0 for v in node["lateness_ms"]), 2)
            self.assertTrue(all(v < payload["period_ms"] for v in node["callback_ms"]))
            self.assertEqual(node["model_s"][-1], 0.002)
        self.assertEqual(payload["duration_ms"], 8000)
        self.assertEqual(payload["nodes"]["plant"]["angle"], [0.1, 0.2])

    def test_rejects_unsorted_completion_times(self):
        values, summary = self.fixture()
        values["flexor_time_s"] = np.asarray([0.003, 0.001])
        with self.assertRaisesRegex(ValueError, "Invalid callback timeline"):
            recording(values, summary)


class ClockTests(unittest.TestCase):
    def run_schedule(self, jump_ns, profile):
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
                self.dt_ms = 0.1
                self.substeps = 10
                self.max_lag_ms = 20
                self.updates = self.steps = 0
                self.profile_timing = profile
                self.stage_timing = np.zeros((4, 1, 2)) if profile else None
                self.timing = np.zeros((4, 5))
                self.stamps = np.zeros(4, dtype=np.int64)
                self.schedule_time = np.zeros(4)
                self.clock_offset_change = np.zeros(4)
                self.clock_sample_span = np.zeros(4)
                self.clock = SimpleNamespace(now=self.ros_now)
                self.clock_probe = {}
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
        self.assertEqual(node.steps, 40)
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
        for profile in (False, True):
            baseline = self.run_schedule(0, profile)
            for jump in (-108_000_000, 108_000_000):
                with self.subTest(profile=profile, jump=jump):
                    np.testing.assert_array_equal(
                        self.run_schedule(jump, profile), baseline
                    )


@pytest.mark.parametrize(
    "settings",
    [
        dict(dt_ms=0.3),
        dict(timer_ms=0.15),
        dict(duration=float("nan")),
        dict(neurons=1.5),
        dict(max_lag_ms=0),
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
    controller = Controller("flexor", warm_steps=10, compile_mode=None)
    controller.window(10, sensory=1)
    state = controller.stream.export_state()
    step = controller.steps
    first = controller.window(10, peer=1)
    controller.stream.reset(state)
    controller.steps = step
    second = controller.window(10, peer=1)
    for left, right in zip(first, second):
        np.testing.assert_array_equal(left, right)
    controller.stream.close()


def test_joint_order_and_normalized_impulse():
    joint = Joint()
    torque = joint.advance([1, 0], 0.0001)
    assert torque == pytest.approx(0.064)
    assert joint.velocity == pytest.approx(0.00128)
    assert joint.angle == pytest.approx(1.28e-7)


def test_native_start_snapshot(tmp_path):
    import torch
    from experiments.exp152.custom.snapshot import capture_start
    from snnlab.sim import Performance, prepare_stream

    cfg = configuration()
    bundle = author_network(cfg)
    stream = prepare_stream(
        bundle,
        window_steps=10,
        dtype=torch.float64,
        device="cpu",
        performance=Performance(compile_mode=None),
    )
    capture_start(stream, bundle, tmp_path / "flexor", cfg)
    snapshot = json.loads((tmp_path / "flexor/start/snapshot.json").read_text())
    assert snapshot["stage"] == "start"
    assert {row["id"] for row in snapshot["network"]["populations"]} == {
        "motor",
        "slow",
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
    assert not summary["passed"]
    assert not summary["timing_passed"]
    assert values["flexor_substep_spikes"].shape == (0, cfg["substeps"], 2)
    assert recording(values, summary)["nodes"]["plant"]["time_ms"] == []


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
const {pulseSamples, routePoint, loopTime, toDegrees, voltageFraction, peerInhibitionRoutes, arrowHead} = await import(
  'data:text/javascript;base64,' + Buffer.from(code).toString('base64'));
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
assert.deepEqual([-100, -75, -60, -45, 0, null].map(v => voltageFraction(v)), [0, 0, .5, 1, 1, 0]);
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


@pytest.mark.parametrize("capacitance, dt_ms", [(0.01, 0.1), (0.01, 1.0), (0.02, 1.0)])
def test_hh_conductance_drive_matches_independent_scipy_solution(capacitance, dt_ms):
    """Validate ionic dynamics and conductance reversal currents independently."""
    import torch
    from experiments.exp152.custom.hodgkin_huxley import initial_state, register, step
    from scipy.integrate import solve_ivp
    from scipy.special import exprel

    def derivative(t, state):
        v, m, h, n = state
        alpha = np.array(
            [
                1 / exprel(-(v + 40) / 10),
                0.07 * np.exp(-(v + 65) / 20),
                0.1 / exprel(-(v + 55) / 10),
            ]
        )
        beta = np.array(
            [
                4 * np.exp(-(v + 65) / 18),
                1 / (1 + np.exp(-(v + 35) / 10)),
                0.125 * np.exp(-(v + 65) / 80),
            ]
        )
        ionic = (
            1.2 * m**3 * h * (v - 50) + 0.36 * n**4 * (v + 77) + 0.003 * (v + 54.387)
        )
        current = 0.005 * (0 - v) + 0.001 * (-80 - v)
        return np.r_[
            (current - ionic) / capacitance, alpha * (1 - state[1:]) - beta * state[1:]
        ]

    state = initial_state(
        SimpleNamespace(shape=(1, 1), device="cpu", dtype=torch.float64)
    )
    keys = ("voltage", "m", "h", "n")
    initial = np.array([state[k].item() for k in keys])
    register()
    duration_ms = 35
    times = np.arange(1, round(duration_ms / dt_ms) + 1) * dt_ms
    reference = solve_ivp(
        derivative,
        (0, duration_ms),
        initial,
        t_eval=times,
        method="DOP853",
        rtol=1e-10,
        atol=1e-12,
    )
    actual = []
    for _ in times:
        state, spikes = step(
            SimpleNamespace(
                state=state,
                dt_ms=dt_ms,
                excitatory=torch.tensor([[0.005]], dtype=torch.float64),
                inhibitory=torch.tensor([[0.001]], dtype=torch.float64),
                config={"capacitance_nf": capacitance},
            )
        )
        actual.append([state[k].item() for k in keys])
        assert spikes.item() in (0, 1)
    np.testing.assert_allclose(
        np.asarray(actual)[:, 0], reference.y[0], atol=0.05, rtol=0
    )
    np.testing.assert_allclose(
        np.asarray(actual)[:, 1:], reference.y[1:].T, atol=0.001, rtol=0
    )


def test_hh_definition_has_conductance_ports_and_gate_state():
    cfg = configuration()
    bundle = author_network(cfg)
    assert cfg["neuron_model"] == "hodgkin_huxley"
    populations = bundle.graph["populations"]
    assert all(row["neuron"]["kind"] == "custom_neuron" for row in populations)


def test_hh_compiled_controller_matches_eager_with_spike_inputs():
    """Compilation must preserve voltage, spikes and their within-window offsets."""
    eager = Controller("flexor", dt_ms=1.0, warm_steps=1, compile_mode=None)
    compiled = Controller("flexor", dt_ms=1.0, warm_steps=1, compile_mode="default")
    try:
        for tick in range(120):
            inputs = dict(sensory=int(tick % 17 == 0), peer=int(tick % 29 == 0))
            expected = eager.window(1, **inputs)
            actual = compiled.window(1, **inputs)
            np.testing.assert_allclose(actual[0], expected[0], atol=1e-7, rtol=1e-8)
            np.testing.assert_array_equal(actual[1], expected[1])
            assert actual[2] == expected[2]
    finally:
        eager.stream.close()
        compiled.stream.close()


def test_hh_closed_loop_reaches_both_sensors_without_joint_stops():
    """A deterministic one-window transport delay isolates the neural controller."""
    cfg = configuration()
    controllers = [
        Controller(side, dt_ms=cfg["dt_ms"], warm_steps=cfg["substeps"])
        for side in ("flexor", "extensor")
    ]
    joint = Joint()
    motor, sensory = np.zeros(2), np.zeros(2, dtype=int)
    sensory_counts = np.zeros(2, dtype=int)
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
                joint.advance(motor if substep == 0 else [0, 0], cfg["dt_ms"] / 1000)
            motor = np.array([row[1][:, 0].sum() for row in output])
            angles.append(joint.angle)
            velocities.append(joint.velocity)
        moving = np.sign(np.asarray(velocities)[np.abs(velocities) > 0.05])
        assert np.isfinite(angles).all()
        assert joint.limit_hits == 0
        assert (sensory_counts > 0).all(), (sensory_counts, min(angles), max(angles))
        assert np.count_nonzero(np.diff(moving)) >= cfg["analysis"]["minimum_reversals"]
    finally:
        for controller in controllers:
            controller.stream.close()


def test_one_ms_protocol_has_no_neuron_clock_scaling():
    cfg = configuration()
    assert cfg["dt_ms"] == cfg["timer_ms"] == 1.0
    for population in author_network(cfg).graph["populations"]:
        assert "time_scale" not in population["neuron"]["config"]


def test_native_hh_preserves_python_rk4_with_internal_parameter_changes():
    import torch
    from experiments.exp152.custom.hodgkin_huxley import (
        initial_state,
        reference_step,
        register,
        step,
    )

    register()
    config = dict(capacitance_nf=0.02, g_k_us=0.45, g_na_us=1.1)
    state = initial_state(
        SimpleNamespace(shape=(2, 3), device="cpu", dtype=torch.float64, config=config)
    )
    expected = {key: value.clone() for key, value in state.items()}
    for tick in range(40):
        ge = torch.full((2, 3), 0.005 if tick < 25 else 0.0, dtype=torch.float64)
        gi = torch.full((2, 3), 0.001 if tick > 15 else 0.0, dtype=torch.float64)
        state, spikes = step(
            SimpleNamespace(
                state=state, dt_ms=1.0, config=config, excitatory=ge, inhibitory=gi
            )
        )
        expected, reference_spikes = reference_step(
            SimpleNamespace(
                state=expected, dt_ms=1.0, config=config, excitatory=ge, inhibitory=gi
            )
        )
        for key in state:
            torch.testing.assert_close(state[key], expected[key], atol=1e-7, rtol=1e-9)
        torch.testing.assert_close(spikes, reference_spikes, atol=0, rtol=0)


def test_startup_estimator_distinguishes_stationary_onset_from_settling():
    from experiments.exp152.analyse import startup_metrics

    time = np.arange(8000) / 1000
    regular = 0.28 * np.sin(2 * np.pi * time / 0.8)
    ramped = regular * np.minimum(1, time / 3)
    steady = startup_metrics(regular)
    transient = startup_metrics(ramped)
    assert transient["settling_s"] > steady["settling_s"] + 1
    assert transient["early_mean_amplitude_error"] > 0.2
    assert steady["early_mean_amplitude_error"] < 0.01


@pytest.mark.parametrize("side", ["flexor", "extensor"])
def test_starting_pulse_preserves_rest_state_and_window_equivalence(side):
    options = dict(flexor_bias_ms=50, flexor_initial_us=0.02, extensor_initial_us=0)
    controllers = [
        Controller(side, compile_mode=None, configuration_overrides=options)
        for _ in range(2)
    ]
    joint = Joint()
    assert joint.angle == joint.velocity == 0
    np.testing.assert_array_equal(joint.activation, [0, 0])
    try:
        for tick in range(60):
            a = controllers[0].window(1)
            b = controllers[1].advance([0], [0])
            np.testing.assert_array_equal(a[0], b[0])
            np.testing.assert_array_equal(a[1], b[1])
            expected = (
                options[side + "_initial_us"]
                if tick < 50
                else configuration()["tonic_us"]
            )
            assert controllers[0].inputs["tonic"].item() == expected
    finally:
        for controller in controllers:
            controller.stream.close()


def test_startup_estimator_preserves_empty_and_stationary_failures():
    from experiments.exp152.analyse import startup_metrics

    assert startup_metrics([])["settling_s"] is None
    assert startup_metrics(np.zeros(8000))["settling_s"] is None
    time = np.arange(4000) * 0.002
    metrics = startup_metrics(0.28 * np.sin(2 * np.pi * time / 0.8), dt_s=0.002)
    assert metrics["first_extremum_s"] == pytest.approx(0.2)
