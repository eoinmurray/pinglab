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
import contextlib

from experiments.exp152.recipe import (
    SIDES,
    author_network,
    configuration,
    controller_settings,
)
from pingstore.contracts import load_json, write_json_atomic
from pingstore.stages import source_run, stage_run

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


def _recorder_type(*, Node, String, Int64MultiArray, diagnostic_qos=100):
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
                and (
                    args.spike_transport != "logical"
                    or all(
                        recorder.count_publishers("/frames/" + name) == 1
                        and recorder.count_subscribers("/frames/" + name) == 2
                        for name in NODES
                    )
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


def compute(
    *, run_id=None, startup_source=None, cycle_validation_source=None, **overrides
):
    cfg = configuration(**overrides)
    inputs = {
        "feasibility": source_run(
            REPO / ".pingstore",
            cfg["feasibility_source"]["run_id"],
            stage="analyse",
            experiment="exp152",
        )
    }
    inputs["load_selection"] = source_run(
        REPO / ".pingstore",
        cfg["load_selection_source"]["run_id"],
        stage="analyse",
        experiment="exp152",
    )
    validation_id = cycle_validation_source or cfg.get(
        "cycle_selection_source", {}
    ).get("run_id")
    if validation_id is not None:
        validation = source_run(
            REPO / ".pingstore", validation_id, stage="analyse", experiment="exp152"
        )
        if (
            validation.record["execution"]["configuration"].get("protocol")
            != "cycle_check"
        ):
            raise ValueError(
                "Cycle validation must use separate native controllers, not a batch screen"
            )
        selected = load_json(validation.file("cycles.json"))["conditions"][0]
        actual = controller_settings(cfg)
        validated_cfg = validation.record["execution"]["configuration"]
        matches = (
            selected["passed"]
            and all(
                actual[key] == value for key, value in selected["condition"].items()
            )
            and all(
                cfg[key] == validated_cfg[key]
                for key in (
                    "dt_ms",
                    "timer_ms",
                    "neurons",
                    "neuron",
                    "joint",
                    "weights",
                    "synapse_tau_ms",
                    "tonic_us",
                    "flexor_bias_ms",
                    "flexor_initial_us",
                    "extensor_initial_us",
                    "analysis",
                )
            )
        )
        if cycle_validation_source is not None and not matches:
            raise ValueError(
                "ROS settings must match the passing native cycle validation"
            )
        inputs["cycle_selection"] = validation
        if matches:
            inputs["cycle_validation"] = validation
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


def cycle_sweep(*, run_id=None, cycle_grid="synaptic", **overrides):
    """Screen independent native SNNLab batches with explicit linear input gains.

    AMPA/GABA filters and projection weights are linear in spike amplitude.
    Scaling each batch's event drive therefore screens different weights with a
    shared graph. Selected conditions must be checked again using actual weights.
    """
    import numpy as np
    import torch
    from experiments.exp152.custom.models import Joint
    from experiments.exp152.custom.snapshot import capture_start
    from experiments.exp152.recipe import cycle_conditions
    from snnlab.sim import Performance, prepare_stream

    cfg = configuration(**{**overrides, "compile_mode": None})
    if cfg["dt_ms"] != 1 or cfg["timer_ms"] != 1 or cfg["neurons"] != 1:
        raise ValueError("Cycle screen requires 1 ms updates and one cell per node")
    conditions = cycle_conditions(cycle_grid)
    protocol = dict(
        cfg,
        protocol="cycle_sweep",
        cycle_grid=cycle_grid,
        screening_weight_precision="float32 initializer constants, then float64 executor",
        conditions=conditions,
        transport_delay_ms=1.0,
        screening="Independent batch pairs; input event amplitudes scaled by candidate/base projection weight; selected condition requires native-weight validation",
        screening_warmup_windows=2,
    )
    inputs = {
        role: source_run(
            REPO / ".pingstore",
            cfg[key]["run_id"],
            stage="analyse",
            experiment="exp152",
        )
        for role, key in (
            ("feasibility", "feasibility_source"),
            ("load_selection", "load_selection_source"),
        )
    }
    torch.set_num_threads(1)
    with stage_run(
        REPO, "exp152", "compute", run_id=run_id, configuration=protocol, inputs=inputs
    ) as run:
        for tau in sorted({row["peer_tau_ms"] for row in conditions}):
            indices = [
                i for i, row in enumerate(conditions) if row["peer_tau_ms"] == tau
            ]
            rows = [conditions[i] for i in indices]
            base = configuration(
                **{
                    **overrides,
                    "compile_mode": None,
                    "peer_tau_ms": tau,
                    "sensory_drive_us": 0.025,
                    "peer_inhibition_us": 0.2,
                }
            )
            bundle = author_network(base)
            stream = prepare_stream(
                bundle,
                window_steps=1,
                batch_size=2 * len(rows),
                device="cpu",
                dtype=torch.float64,
                performance=Performance(compile_mode=None, projection_mode="fast"),
            )
            values = {
                key: torch.zeros(1, 2 * len(rows), 1, dtype=torch.float64)
                for key in ("tonic", "sensory", "peer")
            }
            try:
                for _ in range(2):
                    stream.advance(values)
                stream.reset()
                capture_start(
                    stream,
                    bundle,
                    run.export / f"batch-{int(tau)}" / "flexor",
                    base,
                    diagram=tau == 50,
                )
                joints = [
                    Joint(configuration(**{**overrides, **row})["joint"])
                    for row in rows
                ]
                motor, sensory = np.zeros((len(rows), 2)), np.zeros((len(rows), 2))
                trace = np.zeros((round(cfg["duration"] * 1000), len(rows), 10))
                sensory_gain = np.repeat(
                    [
                        float(np.float32(row["sensory_drive_us"]))
                        / float(np.float32(base["weights"]["sensory_drive"]))
                        for row in rows
                    ],
                    2,
                )
                peer_gain = np.repeat(
                    [
                        float(np.float32(row["peer_inhibition_us"]))
                        / float(np.float32(base["weights"]["peer_inhibition"]))
                        for row in rows
                    ],
                    2,
                )
                tonic = np.repeat([row["tonic_us"] for row in rows], 2)
                startup = np.tile(
                    [cfg["flexor_initial_us"], cfg["extensor_initial_us"]], len(rows)
                )
                with torch.inference_mode():
                    for tick in range(len(trace)):
                        values["tonic"][0, :, 0] = torch.from_numpy(
                            startup if tick < round(cfg["flexor_bias_ms"]) else tonic
                        )
                        values["sensory"][0, :, 0] = torch.from_numpy(
                            sensory.ravel() * sensory_gain
                        )
                        values["peer"][0, :, 0] = torch.from_numpy(
                            motor[:, ::-1].ravel() * peer_gain
                        )
                        result = stream.advance(values)
                        new_motor = (
                            result.outputs["motor_spikes"][0, :, 0]
                            .numpy()
                            .reshape(-1, 2)
                            .copy()
                        )
                        voltages = (
                            result.outputs["motor_voltage"][0, :, 0]
                            .numpy()
                            .reshape(-1, 2)
                        )
                        for j, joint in enumerate(joints):
                            sensory[j] = joint.encode(tick / 1000)
                            joint.advance(motor[j], 0.001)
                            trace[tick, j] = [
                                joint.angle,
                                joint.velocity,
                                *joint.activation,
                                *new_motor[j],
                                *sensory[j],
                                *voltages[j],
                            ]
                        motor = new_motor
                        if (tick + 1) % 5000 == 0:
                            print(
                                "tau",
                                tau,
                                "model seconds",
                                (tick + 1) / 1000,
                                "conditions",
                                len(rows),
                                flush=True,
                            )
                for j, index in enumerate(indices):
                    np.savez_compressed(
                        run.export / f"condition-{index}.npz", trace=trace[:, j]
                    )
                    write_json_atomic(
                        run.export / f"condition-{index}.json",
                        dict(
                            joint=configuration(**{**overrides, **rows[j]})["joint"],
                            limit_hits=joints[j].limit_hits,
                        ),
                    )
            finally:
                stream.close()
        run.record.setdefault("scientific_artifacts", {})["snapshot_stages"] = {
            path.parent.parent.relative_to(run.export).as_posix(): ["start"]
            for path in run.export.rglob("snapshot.json")
        }
    return run.run_id


def startup_sweep(
    *,
    run_id=None,
    load_sweep=False,
    cycle_source=None,
    condition_index=None,
    **overrides,
):
    """Record deterministic coupled trajectories; no ROS timing claims apply."""
    import numpy as np
    from experiments.exp152.custom.models import Controller, Joint
    from experiments.exp152.recipe import startup_conditions

    mode = overrides.get("compile_mode") if cycle_source is not None else None
    cfg = configuration(**{**overrides, "compile_mode": mode})
    if cfg["dt_ms"] != 1 or cfg["timer_ms"] != 1 or cfg["neurons"] != 1:
        raise ValueError(
            "Startup sweep requires one cell per population and 1 ms updates"
        )
    conditions = (
        [dict(tip_mass_kg=mass) for mass in (0.001, 0.002, 0.005, 0.01, 0.05)]
        if load_sweep
        else startup_conditions()
    )
    selection = None
    if cycle_source is not None:
        selection = source_run(
            REPO / ".pingstore", cycle_source, stage="analyse", experiment="exp152"
        )
        selected = load_json(selection.file("cycles.json"))["conditions"][
            condition_index
        ]
        conditions = [selected["condition"]]
        cfg = configuration(
            **{**overrides, **selected["condition"], "compile_mode": mode}
        )
    protocol = dict(
        cfg,
        protocol="cycle_check"
        if selection is not None
        else "load_sweep"
        if load_sweep
        else "startup_sweep",
        conditions=conditions,
        transport_delay_ms=1.0,
        initial_angle_rad=cfg["joint"]["initial_angle"],
        initial_velocity_rad_s=0.0,
    )
    with stage_run(
        REPO,
        "exp152",
        "compute",
        run_id=run_id,
        configuration=protocol,
        inputs={
            **({"cycle_selection": selection} if selection is not None else {}),
            "feasibility": source_run(
                REPO / ".pingstore",
                cfg["feasibility_source"]["run_id"],
                stage="analyse",
                experiment="exp152",
            ),
        },
    ) as run:
        for index, condition in enumerate(conditions):
            controllers = [
                Controller(
                    side,
                    compile_mode=cfg["compile_mode"],
                    configuration_overrides={**controller_settings(cfg), **condition},
                )
                for side in SIDES
            ]
            joint = Joint(controllers[0].cfg["joint"])
            motor, sensory = np.zeros(2), np.zeros(2, dtype=int)
            trace = np.zeros((round(cfg["duration"] * 1000), 8))
            try:
                from experiments.exp152.custom.snapshot import capture_start

                for side, controller in zip(SIDES, controllers, strict=True):
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
                write_json_atomic(
                    run.export / f"condition-{index}.json",
                    dict(
                        joint=controllers[0].cfg["joint"],
                        limit_hits=joint.limit_hits,
                    ),
                )
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
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--startup-sweep", action="store_true")
    modes.add_argument("--load-sweep", action="store_true")
    modes.add_argument("--cycle-sweep", action="store_true")
    parser.add_argument(
        "--cycle-grid", choices=("synaptic", "actuation"), default="synaptic"
    )
    parser.add_argument("--cycle-source")
    parser.add_argument("--condition-index", type=int)
    parser.add_argument("--duration", type=float, default=configuration()["duration"])
    parser.add_argument("--dt-ms", type=float, default=1.0)
    parser.add_argument("--timer-ms", type=float, default=1.0)
    parser.add_argument("--neurons", type=int, default=1)
    parser.add_argument("--max-lag-ms", type=float, default=20.0)
    parser.add_argument("--max-spike-age-ms", type=float, default=5.0)
    parser.add_argument("--profile-timing", action="store_true")
    parser.add_argument(
        "--controller-warmup-ms",
        type=float,
        default=configuration()["controller_warmup_ms"],
    )
    parser.add_argument(
        "--flexor-bias-ms", type=float, default=configuration()["flexor_bias_ms"]
    )
    parser.add_argument(
        "--flexor-initial-us", type=float, default=configuration()["flexor_initial_us"]
    )
    parser.add_argument(
        "--extensor-initial-us",
        type=float,
        default=configuration()["extensor_initial_us"],
    )
    parser.add_argument(
        "--eager", dest="compile_mode", action="store_const", const=None, default=None
    )
    parser.add_argument(
        "--compile", dest="compile_mode", action="store_const", const="default"
    )
    parser.add_argument("--initial-angle-deg", type=float, default=90.0)
    parser.add_argument("--tip-mass-kg", type=float, default=0.002)
    parser.add_argument("--arm-length-m", type=float, default=0.2)
    parser.add_argument("--gravity-m-s2", type=float, default=9.81)
    parser.add_argument("--tonic-us", type=float, default=configuration()["tonic_us"])
    parser.add_argument(
        "--sensory-drive-us",
        type=float,
        default=configuration()["weights"]["sensory_drive"],
    )
    parser.add_argument("--peer-inhibition-us", type=float, default=0.2)
    parser.add_argument("--sensory-tau-ms", type=float, default=10.0)
    parser.add_argument("--peer-tau-ms", type=float, default=50.0)
    parser.add_argument(
        "--activation-increment",
        type=float,
        default=configuration()["joint"]["activation_increment"],
    )
    parser.add_argument(
        "--spike-transport",
        choices=("arrival", "logical"),
        default=configuration()["spike_transport"],
    )
    parser.add_argument("--cycle-validation-source")
    parser.add_argument("--startup-source")
    parser.add_argument("--run-id")
    arguments = vars(parser.parse_args())
    cycle_grid = arguments.pop("cycle_grid")
    screen = arguments.pop("cycle_sweep")
    cycle_source = arguments.pop("cycle_source")
    condition_index = arguments.pop("condition_index")
    load_sweep = arguments.pop("load_sweep")
    sweep = arguments.pop("startup_sweep") or load_sweep or cycle_source is not None
    if (cycle_source is None) != (condition_index is None):
        parser.error("Use --cycle-source and --condition-index together")
    if sweep or screen:
        arguments.pop("startup_source")
        arguments.pop("cycle_validation_source")
    if screen:
        arguments["cycle_grid"] = cycle_grid
    if sweep:
        arguments.update(
            load_sweep=load_sweep,
            cycle_source=cycle_source,
            condition_index=condition_index,
        )
    print(
        (cycle_sweep if screen else startup_sweep if sweep else compute)(**arguments),
        flush=True,
    )
