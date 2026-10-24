"""Acquire three-process ROS joint-control diagnostics into immutable Pingstore.

Run inside the experiment Docker image with four allowed CPUs. No analysis or
presentation stage is invoked; raw callback and spike records remain available.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]
from experiments.exp152.recipe import SIDES, author_network, configuration
from pingstore.contracts import write_json_atomic
from pingstore.stages import source_run, stage_run

NODES = ("flexor", "extensor", "plant")
TOPICS = tuple(
    "/" + kind + "/" + side for kind in ("motor", "sensory") for side in SIDES
)


def stop(children):
    for child in children:
        if child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
    for child in children:
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()


def run_ros(args):
    available_cpus = sorted(os.sched_getaffinity(0))
    if len(available_cpus) < 4:
        raise RuntimeError(
            "Need four allowed CPUs: one each for flexor, extensor, plant and recorder"
        )
    args.cpu_allocation = dict(zip((*NODES, "recorder"), available_cpus[:4]))
    # Set affinity before ROS starts threads. taskset assigns each child its own CPU.
    os.sched_setaffinity(0, {args.cpu_allocation["recorder"]})
    import rclpy
    from rcl_interfaces.srv import SetParameters
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from std_msgs.msg import Int64MultiArray, String

    class Recorder(Node):
        def __init__(self):
            super().__init__("recorder")
            self.records = {name: [] for name in NODES}
            self.spikes = {topic: [] for topic in TOPICS}
            self.handles = [
                self.create_subscription(
                    String,
                    "/diagnostics/" + name,
                    lambda msg, name=name: self.records[name].extend(
                        json.loads(msg.data)
                    ),
                    100,
                )
                for name in NODES
            ]
            self.handles.extend(
                self.create_subscription(
                    Int64MultiArray,
                    topic,
                    lambda msg, topic=topic: self.spikes[topic].append(list(msg.data)),
                    1000,
                )
                for topic in TOPICS
            )

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    recorder = Recorder()
    children, logs = [], []
    error, ready, start_ns = None, False, 0
    try:
        for name in NODES:
            log = (output / f"node_{name}.log").open("w")
            logs.append(log)
            children.append(
                subprocess.Popen(
                    [
                        "taskset",
                        "--cpu-list",
                        str(args.cpu_allocation[name]),
                        sys.executable,
                        "-m",
                        "experiments.exp152.custom.node_" + name,
                        "--ros-args",
                        "-p",
                        f"dt_ms:={float(args.dt_ms)}",
                        "-p",
                        f"timer_ms:={float(args.timer_ms)}",
                        "-p",
                        f"duration:={float(args.duration)}",
                        "-p",
                        f"neurons:={args.neurons}",
                        "-p",
                        f"max_lag_ms:={float(args.max_lag_ms)}",
                        "-p",
                        f"profile_timing:={str(args.profile_timing).lower()}",
                    ],
                    cwd=REPO,
                    env={
                        **os.environ,
                        "EXP152_EXPORT": str(output),
                        "EXP152_CONFIGURATION": json.dumps(vars(args), default=str),
                    },
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            )
        clients = {
            name: recorder.create_client(SetParameters, f"/{name}/set_parameters")
            for name in NODES
        }
        deadline = time.monotonic() + 300
        while not ready:
            rclpy.spin_once(recorder, timeout_sec=0.01)
            if any(child.poll() is not None for child in children):
                raise RuntimeError("A node exited during startup; inspect node logs")
            ready = (
                set(NODES) <= set(recorder.get_node_names())
                and all(client.service_is_ready() for client in clients.values())
                and all(
                    recorder.count_publishers(topic) == 1
                    and recorder.count_subscribers(topic)
                    >= (3 if "/motor/" in topic else 2)
                    for topic in TOPICS
                )
            )
            if time.monotonic() > deadline:
                raise TimeoutError("Nodes or spike connections did not become ready")
        # One future wall-clock origin, after discovery and SNNLab warm-up.
        start_ns = recorder.get_clock().now().nanoseconds + 2_000_000_000
        futures = [
            client.call_async(
                SetParameters.Request(
                    parameters=[
                        Parameter("start_ns", value=start_ns).to_parameter_msg()
                    ]
                )
            )
            for client in clients.values()
        ]
        while not all(future.done() for future in futures):
            rclpy.spin_once(recorder, timeout_sec=0.01)
            if recorder.get_clock().now().nanoseconds >= start_ns:
                raise TimeoutError("Failed to arm every node before the shared start")
        for future in futures:
            response = future.result()
            if not all(result.successful for result in response.results):
                raise RuntimeError("A node rejected the shared start time")
        end_ns = start_ns + int((args.duration + 10) * 1e9)
        while not all(
            any(row["kind"] == "finished" for row in rows)
            for rows in recorder.records.values()
        ):
            rclpy.spin_once(recorder, timeout_sec=0.002)
            if any(child.poll() is not None for child in children):
                raise RuntimeError(
                    "A node exited during the experiment; inspect node logs"
                )
            if recorder.get_clock().now().nanoseconds > end_ns:
                raise TimeoutError(
                    "Nodes did not finish exporting buffered diagnostics"
                )
    except Exception:
        error = traceback.format_exc()
        print(error, file=sys.stderr)
    finally:
        stop(children)
        for log in logs:
            log.close()
        until = time.monotonic() + 0.2
        while time.monotonic() < until:
            rclpy.spin_once(recorder, timeout_sec=0.002)
        write_json_atomic(
            output / "diagnostics.json",
            dict(
                records=recorder.records,
                spikes=recorder.spikes,
                start_ns=start_ns,
                ready=ready,
                error=error,
                cpu_allocation=args.cpu_allocation,
                observed_recorder_affinity=sorted(os.sched_getaffinity(0)),
            ),
        )
        recorder.destroy_node()
        rclpy.shutdown()
    return error


def compute(*, run_id=None, startup_source=None, **overrides):
    cfg = configuration(**overrides)
    inputs = {}
    if startup_source is not None:
        selection = source_run(
            REPO / ".pingstore", startup_source, stage="analyse", experiment="exp152"
        )
        conditions = selection.record["execution"]["configuration"].get(
            "conditions", []
        )
        if not any(
            all(cfg[key] == value for key, value in row.items()) for row in conditions
        ):
            raise ValueError(
                "Starting pulse must match a condition in the selected sweep"
            )
        inputs["startup"] = selection
    with stage_run(
        REPO, "exp152", "compute", run_id=run_id, configuration=cfg, inputs=inputs
    ) as run:
        args = argparse.Namespace(**cfg, output_dir=run.export)
        error = run_ros(args)
        # Completed acquisition may contain a failed behaviour/timing verdict.
        # Preserve it for independent analysis instead of losing the diagnostics.
        run.record["execution"]["acquisition_error"] = error
        snapshots = {}
        for path in run.export.rglob("snapshot.json"):
            snapshots[path.parent.parent.relative_to(run.export).as_posix()] = ["start"]
        run.record.setdefault("scientific_artifacts", {})["snapshot_stages"] = snapshots
    return run.run_id


def startup_sweep(*, run_id=None, **overrides):
    """Record deterministic coupled trajectories; no ROS timing claims apply."""
    import numpy as np
    from experiments.exp152.custom.models import Controller, Joint
    from experiments.exp152.recipe import startup_conditions

    cfg = configuration(**{**overrides, "compile_mode": None})
    if cfg["dt_ms"] != 1 or cfg["timer_ms"] != 1 or cfg["neurons"] != 1:
        raise ValueError(
            "Startup sweep requires one cell per population and 1 ms updates"
        )
    conditions = startup_conditions()
    protocol = dict(
        cfg,
        protocol="startup_sweep",
        conditions=conditions,
        transport_delay_ms=1.0,
        initial_angle_rad=0.0,
        initial_velocity_rad_s=0.0,
    )
    with stage_run(
        REPO, "exp152", "compute", run_id=run_id, configuration=protocol
    ) as run:
        for index, condition in enumerate(conditions):
            controllers = [
                Controller(
                    side,
                    compile_mode=None,
                    configuration_overrides={"duration": cfg["duration"], **condition},
                )
                for side in SIDES
            ]
            joint = Joint()
            motor, sensory = np.zeros(2), np.zeros(2, dtype=int)
            trace = np.zeros((round(cfg["duration"] * 1000), 8))
            try:
                from experiments.exp152.custom.snapshot import capture_start

                for side, controller in zip(SIDES, controllers):
                    capture_start(
                        controller.stream,
                        author_network(controller.cfg),
                        run.export / f"condition-{index}" / side,
                        controller.cfg,
                        diagram=index == 0,
                    )
                for tick in range(len(trace)):
                    output = [
                        controller.window(
                            1, sensory=int(sensory[i]), peer=int(motor[1 - i])
                        )
                        for i, controller in enumerate(controllers)
                    ]
                    sensory = joint.encode(tick / 1000)
                    joint.advance(motor, 0.001)
                    motor = np.array([row[1][0, 0] for row in output])
                    trace[tick] = [
                        joint.angle,
                        joint.velocity,
                        *joint.activation,
                        *motor,
                        *sensory,
                    ]
                np.savez_compressed(run.export / f"condition-{index}.npz", trace=trace)
                print(
                    index,
                    condition,
                    "range",
                    trace[:, 0].min(),
                    trace[:, 0].max(),
                    "stops",
                    joint.limit_hits,
                    flush=True,
                )
            finally:
                for controller in controllers:
                    controller.stream.close()
        run.record.setdefault("scientific_artifacts", {})["snapshot_stages"] = {
            path.parent.parent.relative_to(run.export).as_posix(): ["start"]
            for path in run.export.rglob("snapshot.json")
        }
    print(run.run_id, flush=True)
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--startup-sweep", action="store_true")
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--dt-ms", type=float, default=1.0)
    parser.add_argument("--timer-ms", type=float, default=1.0)
    parser.add_argument("--neurons", type=int, default=1)
    parser.add_argument("--max-lag-ms", type=float, default=20.0)
    parser.add_argument("--max-spike-age-ms", type=float, default=5.0)
    parser.add_argument("--profile-timing", action="store_true")
    parser.add_argument(
        "--flexor-bias-ms", type=float, default=configuration()["flexor_bias_ms"]
    )
    parser.add_argument(
        "--flexor-initial-us", type=float, default=configuration()["flexor_initial_us"]
    )
    parser.add_argument("--extensor-initial-us", type=float, default=0.005)
    parser.add_argument(
        "--eager", dest="compile_mode", action="store_const", const=None, default=None
    )
    parser.add_argument(
        "--compile", dest="compile_mode", action="store_const", const="default"
    )
    parser.add_argument("--startup-source")
    parser.add_argument("--run-id")
    arguments = vars(parser.parse_args())
    sweep = arguments.pop("startup_sweep")
    if sweep:
        arguments.pop("startup_source")
    (startup_sweep if sweep else compute)(**arguments)
