"""Flexor node and shared ROS runtime; node boundaries carry timestamped spikes.

Run as a module inside the local experiment container. The runtime preserves
the source monotonic scheduling and buffered diagnostic protocol.
"""

import gc
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import rclpy
from experiments.exp152.custom.models import Controller, SpikeWindowInbox
from experiments.exp152.recipe import SIDES, window_steps
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Int64MultiArray, String


class RuntimeNode(Node):
    """Absolute release schedule and buffered diagnostics, exported after control."""

    def __init__(self, name):
        super().__init__(name)
        self.configuration = json.loads(os.environ["EXP152_CONFIGURATION"])
        cfg = self.configuration
        self.dt_ms, self.timer_ms = cfg["dt_ms"], cfg["timer_ms"]
        self.substeps = window_steps(self.dt_ms, self.timer_ms)
        self.duration, self.neurons = cfg["duration"], cfg["neurons"]
        self.max_lag_ms = cfg["max_lag_ms"]
        self.declare_parameter("start_ns", 0)
        self.cpu_affinity = sorted(os.sched_getaffinity(0))
        self.steps = 0
        self.updates = 0
        self.capacity = math.ceil(self.duration * 1000 / self.timer_ms)
        # Fixed numeric storage avoids per-window dictionaries and JSON in control.
        self.timing = np.zeros((self.capacity, 5))
        self.stamps = np.zeros(self.capacity, dtype=np.int64)
        self.schedule_time = np.zeros(self.capacity)
        self.clock_offset_change = np.zeros(self.capacity)
        self.clock_sample_span = np.zeros(self.capacity)
        self.deliveries = []
        self.sequences = {}
        self.received_sequences = {}
        self.diagnostics = self.create_publisher(
            String,
            "/diagnostics/" + name,
            QoSProfile(
                history=HistoryPolicy.KEEP_ALL, reliability=ReliabilityPolicy.RELIABLE
            ),
        )
        self.frame_wait_ms = np.zeros(self.capacity)
        sources = [node for node in (*SIDES, "plant") if node != name]
        self.frame_inbox = SpikeWindowInbox(sources, self.substeps)
        self.frame_publisher = self.create_publisher(
            Int64MultiArray, "/frames/" + name, 1000
        )
        self.frame_subscriptions = [
            self.create_subscription(
                Int64MultiArray,
                "/frames/" + source,
                lambda msg, source=source: self.frame_inbox.receive(source, msg.data),
                1000,
            )
            for source in sources
        ]
        self.finished = False
        self.stop_reason = "not_started"
        self.clock = self.get_clock()
        self.period_ns = round(self.timer_ms * 1e6)

    def wait_for_frames(self, executor, release):
        waiting = time.perf_counter_ns()
        now = waiting
        self.incoming_frames = self.frame_inbox.take(self.updates - 1)
        while (
            self.incoming_frames is None
            and rclpy.ok()
            and now - release <= self.max_lag_ms * 1e6
        ):
            executor.spin_once(timeout_sec=self.timer_ms / 1000)
            now = time.perf_counter_ns()
            self.incoming_frames = self.frame_inbox.take(self.updates - 1)
        self.frame_wait_ms[self.updates] = (time.perf_counter_ns() - waiting) / 1e6
        return now

    def execute(self):
        """Service ROS while waiting for fixed releases; never shift the origin."""
        executor = SingleThreadedExecutor()
        executor.add_node(self)
        start_ns = 0
        while rclpy.ok() and not start_ns:
            executor.spin_once(timeout_sec=0.01)
            start_ns = int(self.get_parameter("start_ns").value)
        if not rclpy.ok():
            return
        # Map the common ROS wall-clock epoch to a local monotonic deadline once.
        before = time.perf_counter_ns()
        wall = self.clock.now().nanoseconds
        after = time.perf_counter_ns()
        origin = (before + after) // 2 + start_ns - wall
        initial_offset = wall - (before + after) // 2
        self.clock_mapping = dict(
            ros_start_ns=start_ns,
            monotonic_origin_ns=origin,
            ros_minus_monotonic_ns=initial_offset,
            sample_span_ns=after - before,
        )
        end = origin + round(self.duration * 1e9)
        self.stop_reason = "completed"
        # The bounded run buffers diagnostics; collect cyclic garbage outside control.
        gc_was_enabled = gc.isenabled()
        gc.collect()
        gc.disable()
        while rclpy.ok() and self.updates < self.capacity:
            release = origin + self.updates * self.period_ns
            now = time.perf_counter_ns()
            while now < release and rclpy.ok():
                # Service ROS at least once per period while armed, including startup.
                executor.spin_once(
                    timeout_sec=min((release - now) / 1e9, self.timer_ms / 1000)
                )
                now = time.perf_counter_ns()
            now = self.wait_for_frames(executor, release)
            if not rclpy.ok():
                self.stop_reason = "interrupted"
                break
            if now - release > self.max_lag_ms * 1e6:
                self.stop_reason = "excessive_lag"
                break
            # If late, preserve all model windows and the original deadlines.
            # Catch up immediately, but service queued spikes between late windows.
            if now - release >= self.period_ns:
                executor.spin_once(timeout_sec=0)
            started = time.perf_counter_ns()
            wall_start = self.clock.now().nanoseconds
            sampled = time.perf_counter_ns()
            index = self.updates
            self.clock_offset_change[index] = (
                wall_start - (started + sampled) // 2 - initial_offset
            ) / 1e6
            self.clock_sample_span[index] = (sampled - started) / 1e6
            self.frame_events = []
            compute_ms = self.tick()
            self.frame_publisher.publish(
                Int64MultiArray(
                    data=[
                        self.updates,
                        self.substeps,
                        *[int(value) for event in self.frame_events for value in event],
                    ]
                )
            )
            self.steps += self.substeps
            self.stamps[index] = self.clock.now().nanoseconds
            self.timing[index, 0] = (started - release) / 1e6
            self.timing[index, 1] = compute_ms
            self.timing[index, 2] = self.steps * self.dt_ms / 1000
            # Include model work, spike publication and writes to diagnostic buffers.
            self.timing[index, 3] = (time.perf_counter_ns() - started) / 1e6
            self.timing[index, 4] = (
                self.timing[index, 0] + self.timing[index, 3] - self.timer_ms
            )
            self.schedule_time[index] = (started - origin) / 1e9 + self.timing[
                index, 3
            ] / 1000
            self.updates += 1
        # Export only after the shared control interval, including when stopping early.
        while rclpy.ok() and time.perf_counter_ns() < end:
            executor.spin_once(timeout_sec=max(0, (end - time.perf_counter_ns()) / 1e9))
        if gc_was_enabled:
            gc.enable()
        self.flush()
        # Keep publishers alive while the recorder drains reliable diagnostics.
        while rclpy.ok():
            executor.spin_once(timeout_sec=0.1)

    def emit(self, publisher, neuron_id=0, model_time_ms=0.0):
        topic = publisher.topic_name
        seq = self.sequences.get(topic, 0)
        self.sequences[topic] = seq + 1
        publisher.publish(
            Int64MultiArray(
                data=[
                    seq,
                    self.clock.now().nanoseconds,
                    int(neuron_id),
                    round(model_time_ms * 1e6),
                ]
            )
        )

    def receive(self, msg, topic):
        if len(msg.data) != 4:
            raise ValueError(
                "Spike payload must be [sequence, publication_ns, neuron_id, source_model_time_ns]"
            )
        sequence, stamp, _neuron_id, model_time_ns = msg.data
        now = self.clock.now().nanoseconds
        previous = self.received_sequences.get(topic, -1)
        self.deliveries.append(
            (
                now,
                stamp,
                topic,
                model_time_ns,
                (now - stamp) / 1e6,
                max(0, sequence - previous - 1),
            )
        )
        if sequence <= previous:
            return False
        self.received_sequences[topic] = sequence
        return True

    def diagnostic_values(self, index):
        raise NotImplementedError

    def flush(self):
        if self.finished:
            return
        records = []
        for index in range(self.updates):
            lag, compute, model_time, callback, lateness = self.timing[index]
            records.append(
                dict(
                    kind="tick",
                    stamp_ns=int(self.stamps[index]),
                    schedule_time_s=self.schedule_time[index],
                    clock_offset_change_ms=self.clock_offset_change[index],
                    clock_sample_span_ms=self.clock_sample_span[index],
                    model_time_s=model_time,
                    input_frame_wait_ms=self.frame_wait_ms[index],
                    wall_lag_ms=lag,  # Legacy field name; lag uses the monotonic release.
                    compute_ms=compute,
                    callback_ms=callback,
                    deadline_lateness_ms=lateness,
                    **self.diagnostic_values(index),
                    **(
                        {
                            "cpu_affinity": self.cpu_affinity,
                            "clock_mapping": self.clock_mapping,
                        }
                        if index == 0
                        else {}
                    ),
                )
            )
        records.extend(
            dict(
                kind="delivery",
                stamp_ns=now,
                sent_ns=stamp,
                topic=topic,
                source_model_time_ns=model,
                age_ms=age,
                missing=missing,
            )
            for now, stamp, topic, model, age, missing in self.deliveries
        )
        records.append(dict(kind="finished", stop_reason=self.stop_reason))
        for offset in range(0, len(records), 100):
            self.diagnostics.publish(
                String(data=json.dumps(records[offset : offset + 100]))
            )
        self.finished = True


class ControllerNode(RuntimeNode):
    def __init__(self, side):
        super().__init__(side)
        self.side = side
        self.model = Controller(
            side,
            self.configuration,
            evidence_dir=Path(os.environ["EXP152_EXPORT"]) / side,
        )
        self.voltage = np.zeros((self.capacity, self.substeps, 1))
        self.spikes = np.zeros((self.capacity, self.substeps, 1), dtype=int)
        self.publisher = self.create_publisher(Int64MultiArray, "/motor/" + side, 1000)
        self.sensory_topic = "/sensory/" + side
        self.peer_topic = "/motor/" + SIDES[1 - SIDES.index(side)]
        self.subscriptions_ = [
            self.create_subscription(
                Int64MultiArray, self.sensory_topic, self.sensory, 1000
            ),
            self.create_subscription(Int64MultiArray, self.peer_topic, self.peer, 1000),
        ]
        self.get_logger().info(
            f"Ready: {self.neurons} neurons; CPU {self.cpu_affinity}; {self.model.info}"
        )

    def sensory(self, msg):
        self.receive(msg, self.sensory_topic)

    def peer(self, msg):
        self.receive(msg, self.peer_topic)

    def tick(self):
        tick = time.perf_counter_ns()
        sensory = sum(
            unit == SIDES.index(self.side) for _, unit in self.incoming_frames["plant"]
        )
        peer = len(self.incoming_frames[SIDES[1 - SIDES.index(self.side)]])
        voltage, spikes, events = self.model.window(self.substeps, sensory, peer)
        for offset, neuron_id in events:
            self.emit(self.publisher, neuron_id, (self.steps + offset + 1) * self.dt_ms)
        compute_ms = (time.perf_counter_ns() - tick) / 1e6
        self.frame_events = events
        self.voltage[self.updates] = voltage
        self.spikes[self.updates] = spikes
        return compute_ms

    def diagnostic_values(self, index):
        return dict(
            voltage=self.voltage[index, -1].tolist(),
            spikes=self.spikes[index].sum(axis=0).tolist(),
            **({"model_info": self.model.info} if index == 0 else {}),
        )


def run(side="flexor"):
    rclpy.init()
    node = ControllerNode(side)
    try:
        node.execute()
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.flush()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    run("flexor")
