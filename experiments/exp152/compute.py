"""Acquire the report’s three-node ROS joint-control diagnostics into immutable Pingstore.

Flexor, extensor and plant run as three independent ROS processes on separate
CPUs in the experiment image. Every node retains its own 1 ms spike-window
schedule and callback timings. Analysis and presentation remain separate stages.
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
sys.path[:0] = [str(REPO), str(REPO / "libraries")]
import contextlib

from experiments.exp152.recipe import SIDES, configuration
from pingstore.library import stage_run, write_json_atomic

NODES = ("flexor", "extensor", "plant")
TOPICS = tuple(
    "/" + kind + "/" + side for kind in ("motor", "sensory") for side in SIDES
)


def stop(children):
    for child in children:
        if child.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(child.pid, signal.SIGINT)
    for child in children:
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()


def _recorder_type(*, Node, String, Int64MultiArray, diagnostic_qos):
    """Build the recorder node after optional ROS types have been imported."""

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
                    diagnostic_qos,
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

    return Recorder


def _launch_ros_nodes(*, args, output, children, logs):
    """Launch the assigned node processes while retaining cleanup ownership."""
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


def run_ros(args):
    available_cpus = sorted(os.sched_getaffinity(0))
    if len(available_cpus) < 4:
        raise RuntimeError(
            "Need four allowed CPUs: one each for flexor, extensor, plant and recorder"
        )
    args.cpu_allocation = dict(
        zip((*NODES, "recorder"), available_cpus[:4], strict=True)
    )
    # Set affinity before ROS starts threads. taskset assigns each child its own CPU.
    os.sched_setaffinity(0, {args.cpu_allocation["recorder"]})
    import rclpy
    from rcl_interfaces.srv import SetParameters
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from std_msgs.msg import Int64MultiArray, String

    Recorder = _recorder_type(
        Node=Node,
        String=String,
        Int64MultiArray=Int64MultiArray,
        diagnostic_qos=QoSProfile(
            history=HistoryPolicy.KEEP_ALL, reliability=ReliabilityPolicy.RELIABLE
        ),
    )

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    recorder = Recorder()
    children, logs = [], []
    error, ready, start_ns = None, False, 0
    try:
        _launch_ros_nodes(args=args, output=output, children=children, logs=logs)
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
                and all(
                    recorder.count_publishers("/frames/" + name) == 1
                    and recorder.count_subscribers("/frames/" + name) == 2
                    for name in NODES
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


def compute(*, run_id=None, **overrides):
    cfg = configuration(**overrides)
    with stage_run(REPO, "exp152", "compute", run_id=run_id, configuration=cfg) as run:
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


def main(argv=None):
    """Acquire only the ROS report protocol with the documented arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = configuration()
    parser.add_argument("--duration", type=float, default=defaults["duration"])
    parser.add_argument("--max-lag-ms", type=float, default=defaults["max_lag_ms"])
    parser.add_argument(
        "--compile",
        dest="compile_mode",
        action="store_const",
        const="default",
        default=defaults["compile_mode"],
        help="Use the report's compiled SNNLab execution (default)",
    )
    parser.add_argument("--run-id")
    print(compute(**vars(parser.parse_args(argv))), flush=True)


if __name__ == "__main__":
    main()
