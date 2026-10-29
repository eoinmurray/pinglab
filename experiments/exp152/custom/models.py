"""Adapt the prepared-stream lifecycle and mechanical plant to ROS callbacks.

The network is authored in recipe.py; ionic equations live in the Elevate
extension. Input buffering and the spike-driven mechanical integrator retain
the recorded update order.
"""

import math
import time

import numpy as np
from experiments.exp152.custom.snapshot import capture_start
from experiments.exp152.recipe import SIDES, author_network, configuration


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

    Each controller has one Elevate neuron. Only spike events connect it to
    its peer and the plant; tonic excitation is local.
    """

    def __init__(self, side, cfg, *, evidence_dir=None):
        import torch
        from snnlab.sim import Performance, prepare_stream

        if side not in SIDES:
            raise ValueError("Choose flexor or extensor")
        self.side = side
        self.dt_ms, self.neurons = cfg["dt_ms"], cfg["neurons"]
        self.torch, self.steps = torch, 0
        self.cfg = cfg
        warm_steps = cfg["substeps"]
        dt_ms, compile_mode = cfg["dt_ms"], cfg["compile_mode"]
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
                cfg,
            )

    def _execute(self, length):
        with self.torch.inference_mode():
            result = self.stream.advance(self.inputs)
            self.steps += length
            voltages = result.outputs["motor_voltage"][:, 0].numpy()
            spike_array = result.outputs["motor_spikes"][:, 0].numpy()
            # Returned diagnostics own their arrays even if streaming reuses outputs.
            voltage = voltages.copy()
            spikes = spike_array.astype(int)
            motor_events = [[0, 0]] if spikes[0, 0] else []
        return voltage, spikes, motor_events

    def window(self, steps, sensory=0, peer=0):
        # ROS uses a fixed shape: reuse tensors and avoid NumPy-to-Torch conversion.
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

    def encode(self, now):
        spikes = np.zeros(2, dtype=int)
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
