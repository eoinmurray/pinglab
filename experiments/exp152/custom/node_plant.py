"""Mechanical plant ROS node; all controller-facing messages are spikes.

Run as a module inside the local experiment container. This adapter applies
received motor events once per window and buffers mechanical state.
"""

import json
import os
import time

import numpy as np
import rclpy
from experiments.exp152.custom.models import Joint
from experiments.exp152.custom.node_flexor import RuntimeNode
from experiments.exp152.recipe import SIDES
from std_msgs.msg import Int64MultiArray


class PlantNode(RuntimeNode):
    def __init__(self):
        super().__init__("plant")
        self.plant = Joint(json.loads(os.environ["EXP152_CONFIGURATION"])["joint"])
        self.publishers_ = [
            self.create_publisher(Int64MultiArray, "/sensory/" + s, 1000) for s in SIDES
        ]
        self.subscriptions_ = [
            self.create_subscription(
                Int64MultiArray,
                "/motor/" + s,
                lambda msg, i=i: self.motor(i, msg),
                1000,
            )
            for i, s in enumerate(SIDES)
        ]
        self.state_buffer = np.zeros((self.capacity, 7))
        self.motor_buffer = np.zeros(2)
        self.get_logger().info(
            f"Ready: mechanical joint, CPU {self.cpu_affinity}, no neurons"
        )

    def motor(self, i, msg):
        self.receive(msg, "/motor/" + SIDES[i])

    def tick(self):
        tick = time.perf_counter_ns()
        motor = [len(self.incoming_frames[side]) for side in SIDES]
        events = []
        np.divide(motor, self.neurons, out=self.motor_buffer)
        for substep in range(self.substeps):
            model_ms = (self.steps + substep) * self.dt_ms
            sensory = self.plant.encode(model_ms / 1000)
            for i, count in enumerate(sensory):
                if count:
                    events.append((i, model_ms))
                    self.frame_events.append((substep, i))
            # Received motor events enter once, before the first decay/update.
            gravity_torque = self.plant.gravity_torque(self.plant.angle)
            torque = self.plant.advance(self.motor_buffer, self.dt_ms / 1000)
            self.motor_buffer.fill(0)
        for i, model_ms in events:
            self.emit(self.publishers_[i], model_time_ms=model_ms)
        compute_ms = (time.perf_counter_ns() - tick) / 1e6
        self.state_buffer[self.updates] = (
            self.plant.angle,
            self.plant.velocity,
            *self.plant.activation,
            torque,
            self.plant.limit_hits,
            gravity_torque,
        )
        return compute_ms

    def diagnostic_values(self, index):
        angle, velocity, flexor, extensor, torque, hits, gravity_torque = (
            self.state_buffer[index]
        )
        return dict(
            angle=angle,
            velocity=velocity,
            activation=[flexor, extensor],
            torque=torque,
            gravity_torque=gravity_torque,
            limit_hits=int(hits),
        )


def main():
    rclpy.init()
    node = PlantNode()
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
    main()
