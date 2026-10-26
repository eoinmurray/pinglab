"""Adapt the prepared-stream lifecycle and mechanical plant to ROS callbacks.

The network is authored in recipe.py; ionic equations live in the Elevate
extension. Timing probes, input buffering and the spike-driven mechanical
integrator retain the source update order.
"""

import json
import math
import os
import time

import numpy as np
from experiments.exp152.custom.snapshot import capture_start
from experiments.exp152.recipe import DT_CHOICES, SIDES, author_network, configuration


def start_measurement():
    """Pair elapsed wall time with CPU time consumed by this calling thread."""
    return time.perf_counter_ns(), time.thread_time_ns()


def finish_measurement(start, target):
    cpu = time.thread_time_ns()
    wall = time.perf_counter_ns()
    target[0] = (wall - start[0]) / 1e6
    target[1] = (cpu - start[1]) / 1e6


class SpikeWindowInbox:
    """Buffer complete sparse spike windows; empty frames make silence explicit."""

    def __init__(self, sources, substeps):
        self.pending = {source: {} for source in sources}
        self.substeps = substeps
        self.consumed = -1

    def receive(self, source, payload):
        if (
            len(payload) < 2
            or payload[0] < 0
            or payload[1] != self.substeps
            or (len(payload) - 2) % 2
        ):
            raise ValueError("Invalid spike window header or sparse event pairs")
        index = payload[0]
        events = list(zip(payload[2::2], payload[3::2], strict=True))
        if any(not 0 <= step < self.substeps or unit < 0 for step, unit in events):
            raise ValueError("Invalid spike window event")
        if index > self.consumed:
            self.pending[source][index] = events

    def take(self, index):
        if index < 0:
            return {source: [] for source in self.pending}
        if not all(index in rows for rows in self.pending.values()):
            return None
        self.consumed = index
        return {source: rows.pop(index) for source, rows in self.pending.items()}


class Controller:
    """One motor population, preserving SNNLab state each step.

    Population size one gives one neuron per controller. Larger populations
    replicate identical cells with normalized dense connectivity: a load test,
    not a claim of additional biological complexity. Only spike events connect
    this model to its peer and the plant. Tonic excitation is local.
    """

    def __init__(
        self,
        side,
        dt_ms=1.0,
        neurons=1,
        *,
        warm_steps=1,
        compile_mode=None,
        profile_timing=False,
        evidence_dir=None,
        configuration_overrides=None,
    ):
        import torch
        from snnlab.sim import Performance, prepare_stream

        if (
            side not in SIDES
            or dt_ms not in DT_CHOICES
            or neurons < 1
            or warm_steps < 1
        ):
            raise ValueError("Invalid side, timestep or population size")
        torch.set_num_threads(1)
        self.side, self.dt_ms, self.neurons = side, dt_ms, neurons
        self.profile_timing = profile_timing
        self.stage_timing = np.zeros((3, 2))
        self.torch, self.steps = torch, 0
        cfg = configuration(
            dt_ms=dt_ms,
            neurons=neurons,
            timer_ms=warm_steps * dt_ms,
            compile_mode=compile_mode,
            **(configuration_overrides or {}),
        )
        self.cfg = cfg
        bundle = author_network(cfg)
        # Preparation privately owns the model/state and caches fixed graph metadata.
        # Keep float64 CPU dynamics; the tensor Elevate callback integrates each step.
        started = time.perf_counter()
        self.stream = prepare_stream(
            bundle,
            window_steps=warm_steps,
            batch_size=1,
            device="cpu",
            dtype=torch.float64,
            performance=Performance(
                compile_mode=compile_mode,
                compile_block_steps=min(warm_steps, 64),
                cuda_graphs="off",
                precision="highest",
                projection_mode="fast",
            ),
        )
        self.inputs = {
            name: torch.zeros(warm_steps, 1, 1, dtype=torch.float64)
            for name in ("tonic", "sensory", "peer")
        }
        path = self.stream.info["execution_path"]
        if compile_mode is not None and path["selected"] not in {
            "compiled_blocks",
            "kernel_loop",
        }:
            raise RuntimeError(f"Compiled blocks unavailable: {path}")
        self.info = dict(
            prepared_stream=True,
            persistent_state=self.stream.info.get("persistent_state", False),
            compile_mode=compile_mode,
            execution_path=path["selected"],
            block_steps=path["block_steps"],
            warm_input_steps=warm_steps,
            warmup_seconds=time.perf_counter() - started,
            torch_version=torch.__version__,
        )
        # Exercise quiet, sensory and peer-input windows before the common ROS epoch.
        warmup_windows = max(
            2, math.ceil(cfg["controller_warmup_ms"] / (warm_steps * dt_ms))
        )
        self.stream.reset()
        self.steps = 0
        for index in range(warmup_windows):
            self.window(
                warm_steps, sensory=int(index % 17 == 0), peer=int(index % 29 == 0)
            )
        # Warm-up must never become a scientific burn-in or emit ROS spikes.
        self.stream.reset()
        self.steps = 0
        for values in self.inputs.values():
            values.zero_()
        self.stage_timing.fill(0)
        self.info.update(
            warmup_seconds=time.perf_counter() - started,
            warmup_windows=warmup_windows,
            warmup_model_ms=warmup_windows * warm_steps * dt_ms,
            warmup_state_reset=True,
        )
        if evidence_dir is not None:
            capture_start(
                self.stream,
                bundle,
                evidence_dir,
                json.loads(os.environ["EXP152_CONFIGURATION"]),
            )

    def advance(self, sensory, peer):
        """Execute a window in one SNNLab call; inputs are counts per substep."""
        length = len(sensory)
        if length < 1 or len(peer) != length:
            raise ValueError("Input windows must have the same positive length")
        with self.torch.inference_mode():
            if self.inputs["tonic"].shape[0] != length:
                raise ValueError("Input window must match the prepared stream shape")
            self.inputs["tonic"].fill_(self.cfg["tonic_us"])
            if self.steps < round(self.cfg["flexor_bias_ms"] / self.dt_ms):
                bias_steps = max(
                    0,
                    min(
                        length,
                        round(self.cfg["flexor_bias_ms"] / self.dt_ms) - self.steps,
                    ),
                )
                self.inputs["tonic"][:bias_steps] += (
                    self.cfg[self.side + "_initial_us"] - self.cfg["tonic_us"]
                )
            for name, values in [("sensory", sensory), ("peer", peer)]:
                self.inputs[name].copy_(
                    self.torch.as_tensor(values, dtype=self.torch.float64).reshape(
                        length, 1, 1
                    )
                )
            return self._execute(length)

    def _execute(self, length):
        with self.torch.inference_mode():
            measured = start_measurement() if self.profile_timing else None
            result = self.stream.advance(self.inputs)
            if measured is not None:
                finish_measurement(measured, self.stage_timing[1])
            measured = start_measurement() if self.profile_timing else None
            self.steps += length
            voltages = result.outputs["motor_voltage"][:, 0].numpy()
            spike_array = result.outputs["motor_spikes"][:, 0].numpy()
            if self.neurons == 1:
                # Returned diagnostics own their arrays even if streaming reuses outputs.
                voltage = voltages.copy()
                spikes = spike_array.astype(int)
                if length == 1:
                    motor_events = [[0, 0]] if spikes[0, 0] else []
                else:
                    motor_events = [[int(i), 0] for i in np.flatnonzero(spikes[:, 0])]
            else:
                voltage = voltages.mean(axis=1, keepdims=True)
                spikes = spike_array.sum(axis=1, keepdims=True).astype(int)
                # Retain every substep index and neuron identity in load tests.
                motor_events = np.argwhere(spike_array != 0).tolist()
            if measured is not None:
                finish_measurement(measured, self.stage_timing[2])
        return voltage, spikes, motor_events

    def step(self, sensory=0, peer=0):
        voltage, spikes, events = self.advance([sensory], [peer])
        return voltage[0], spikes[0], [neuron for _, neuron in events]

    def window(self, steps, sensory=0, peer=0):
        # ROS uses a fixed shape: reuse tensors and avoid NumPy-to-Torch conversion.
        measured = start_measurement() if self.profile_timing else None
        if steps != self.inputs["tonic"].shape[0]:
            raise ValueError("Window must match the warmed callback shape")
        with self.torch.inference_mode():
            self.inputs["tonic"].fill_(self.cfg["tonic_us"])
            if self.steps < round(self.cfg["flexor_bias_ms"] / self.dt_ms):
                bias_steps = min(
                    steps, round(self.cfg["flexor_bias_ms"] / self.dt_ms) - self.steps
                )
                self.inputs["tonic"][:bias_steps].fill_(
                    self.cfg[self.side + "_initial_us"]
                )
            self.inputs["sensory"].zero_()
            self.inputs["peer"].zero_()
            self.inputs["sensory"][0, 0, 0] = sensory
            self.inputs["peer"][0, 0, 0] = peer
            if measured is not None:
                finish_measurement(measured, self.stage_timing[0])
            return self._execute(steps)


class Joint:
    """Damped rotary joint with opposing spike-driven virtual muscles."""

    def __init__(self, settings=None):
        for name, value in (
            configuration()["joint"] if settings is None else settings
        ).items():
            setattr(self, name, value)
        self.angle, self.velocity = self.initial_angle, 0.0
        self.activation = np.zeros(2)
        self.next_sensory = np.zeros(2)
        self.limit_hits = 0
        if abs(self.angle - self.reference_angle) > self.hard_limit:
            raise ValueError("Initial angle lies outside the mechanical stops")

    def gravity_torque(self, angle):
        return (
            -self.tip_mass_kg * self.gravity_m_s2 * self.arm_length_m * math.sin(angle)
        )

    def encode(self, now, feedback=True):
        spikes = np.zeros(2, dtype=int)
        if feedback:
            for i, active in enumerate(
                (
                    self.angle < self.reference_angle - self.sensory_limit,
                    self.angle > self.reference_angle + self.sensory_limit,
                )
            ):
                if active and now + 1e-12 >= self.next_sensory[i]:
                    spikes[i] = 1
                    self.next_sensory[i] = now + self.sensory_interval
        return spikes

    def advance(self, motor_spikes, dt):
        # Callers normalize motor counts by population size.
        self.activation = np.minimum(
            1.0,
            self.activation * math.exp(-dt / self.activation_tau)
            + self.activation_increment * np.asarray(motor_spikes),
        )
        torque = self.max_torque * (self.activation[0] - self.activation[1])
        acceleration = (
            torque
            + self.gravity_torque(self.angle)
            - self.damping * self.velocity
            - self.stiffness * (self.angle - self.spring_rest_angle)
        ) / self.total_inertia
        self.velocity += dt * acceleration
        self.angle += dt * self.velocity
        if abs(self.angle - self.reference_angle) > self.hard_limit:
            self.limit_hits += 1
            self.angle = float(
                np.clip(
                    self.angle,
                    self.reference_angle - self.hard_limit,
                    self.reference_angle + self.hard_limit,
                )
            )
            self.velocity = 0.0
        return torque
