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
import copy
import shutil

from experiments.exp152.recipe import (
    SIDES,
    adaptation_configuration,
    author_network,
    configuration,
    controller_settings,
)
from pingstore.library import (
    file_sha256,
    load_json,
    payload_inventory,
    source_run,
    stage_run,
    validated_run_graph,
    write_json_atomic,
)

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


def compute(*, run_id=None, cycle_validation_source=None, **overrides):
    cfg = configuration(**overrides)
    inputs = {}
    validation_id = cycle_validation_source
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


def cycle_check(source=None, *, condition_index=None, run_id=None, **overrides):
    """Validate one selected condition with separate native controller states.

    This is the deterministic 1 ms reference used by the report's ROS trace
    comparison. It makes no wall-clock timing claim and performs no search.
    """
    import numpy as np
    from experiments.exp152.custom.models import Controller, Joint
    from experiments.exp152.custom.snapshot import capture_start

    inputs = {}
    condition = {}
    if source is not None:
        selection = source_run(
            REPO / ".pingstore", source, stage="analyse", experiment="exp152"
        )
        condition = load_json(selection.file("cycles.json"))["conditions"][
            condition_index
        ]["condition"]
        inputs["cycle_selection"] = selection
    cfg = configuration(**{**overrides, **condition})
    condition = dict(
        activation_increment=cfg["joint"]["activation_increment"],
        peer_inhibition_us=cfg["weights"]["peer_inhibition"],
        peer_tau_ms=cfg["synapse_tau_ms"]["peer_inhibition"],
        sensory_drive_us=cfg["weights"]["sensory_drive"],
        tonic_us=cfg["tonic_us"],
    )
    protocol = dict(
        cfg,
        protocol="cycle_check",
        conditions=[condition],
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
        inputs=inputs,
    ) as run:
        controllers = []
        try:
            for side in SIDES:
                controller = Controller(
                    side,
                    compile_mode=cfg["compile_mode"],
                    configuration_overrides=controller_settings(cfg),
                )
                controllers.append(controller)
                capture_start(
                    controller.stream,
                    author_network(controller.cfg),
                    run.export / "condition-0" / side,
                    controller.cfg,
                )
            joint = Joint(cfg["joint"])
            motor, sensory = np.zeros(2), np.zeros(2, dtype=int)
            trace = np.zeros((round(cfg["duration"] * 1000), 8))
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
            np.savez_compressed(run.export / "condition-0.npz", trace=trace)
            write_json_atomic(
                run.export / "condition-0.json",
                dict(joint=cfg["joint"], limit_hits=joint.limit_hits),
            )
        finally:
            for controller in controllers:
                controller.stream.close()
        run.record.setdefault("scientific_artifacts", {})["snapshot_stages"] = {
            path.parent.parent.relative_to(run.export).as_posix(): ["start"]
            for path in run.export.rglob("snapshot.json")
        }
    return run.run_id


class AdaptiveJointSimulation:
    """Retain neural, mechanical and one-window transport state between probes."""

    def __init__(self, cfg):
        import numpy as np
        from experiments.exp152.custom.models import Joint

        self.cfg = cfg
        self.controllers = []
        self.joint = Joint(cfg["joint"])
        self.motor, self.sensory = np.zeros(2), np.zeros(2, dtype=int)
        self.tick = 0
        self.segment_index = 0

    def configure(self, parameters, directory):
        from experiments.exp152.custom.models import Controller
        from experiments.exp152.custom.snapshot import capture_start

        states = [c.stream.export_state() for c in self.controllers]
        replacement = []
        try:
            for index, side in enumerate(SIDES):
                controller = Controller(
                    side,
                    compile_mode=self.cfg["compile_mode"],
                    configuration_overrides={
                        **controller_settings(self.cfg),
                        **parameters,
                    },
                )
                replacement.append(controller)
                if states:
                    controller.stream.reset(states[index])
                    controller.steps = states[index].completed_steps
                capture_start(
                    controller.stream,
                    author_network(controller.cfg),
                    directory / side,
                    controller.cfg,
                    diagram=directory.name == "segment-000",
                )
        except BaseException:
            for controller in replacement:
                controller.stream.close()
            raise
        self.close()
        self.controllers = replacement

    def advance(self, seconds):
        import numpy as np

        trace = np.zeros((round(seconds * 1000), 8))
        for row in trace:
            output = [
                controller.window(
                    1, sensory=int(self.sensory[i]), peer=int(self.motor[1 - i])
                )
                for i, controller in enumerate(self.controllers)
            ]
            self.sensory = self.joint.encode(self.tick / 1000)
            self.joint.advance(self.motor, 0.001)
            self.motor = np.array([value[1][0, 0] for value in output])
            row[:] = [
                self.joint.angle,
                self.joint.velocity,
                *self.joint.activation,
                *self.motor,
                *self.sensory,
            ]
            self.tick += 1
        return trace

    def close(self):
        for controller in self.controllers:
            controller.stream.close()
        self.controllers = []


def _adaptation_segment(simulation, parameters, kind, run, cfg, *, validation=False):
    import numpy as np
    from experiments.exp152.analyse import adaptation_measure

    directory = run.export / f"segment-{simulation.segment_index:03d}"
    simulation.segment_index += 1
    simulation.configure(parameters, directory)
    start_tick, start_hits = simulation.tick, simulation.joint.limit_hits
    duration = cfg["adaptation"][
        "validation_duration_s" if validation else "segment_duration_s"
    ]
    trace = simulation.advance(duration)
    hits = simulation.joint.limit_hits - start_hits
    measurement = adaptation_measure(trace, cfg, validation=validation, limit_hits=hits)
    measurements = [dict(duration_s=duration, measurement=measurement)]
    maximum_duration = cfg["adaptation"].get("maximum_segment_duration_s", duration)
    while (
        not validation
        and not hits
        and not measurement["eligible"]
        and duration < maximum_duration
    ):
        extension = min(
            cfg["adaptation"]["measurement_extension_s"], maximum_duration - duration
        )
        trace = np.concatenate([trace, simulation.advance(extension)])
        duration += extension
        hits = simulation.joint.limit_hits - start_hits
        measurement = adaptation_measure(trace, cfg, limit_hits=hits)
        measurements.append(dict(duration_s=duration, measurement=measurement))
    np.savez_compressed(directory / "trace.npz", trace=trace)
    row = dict(
        kind=kind,
        parameters=dict(parameters),
        validation=validation,
        start_time_s=start_tick / 1000,
        duration_s=duration,
        limit_hits=hits,
        trace_file=str((directory / "trace.npz").relative_to(run.export)),
        online_measurement=measurement,
        measurement_checks=measurements,
        measurement_status=(
            "settled"
            if measurement["eligible"]
            else "timeout"
            if not hits
            else "limit_hit"
        ),
    )
    print(
        json.dumps(
            dict(
                segment=simulation.segment_index - 1,
                kind=kind,
                parameters=parameters,
                frequency_hz=measurement["frequency_hz"],
                excursion_deg=(
                    float(np.rad2deg(measurement["excursion_rad"]))
                    if measurement["valid"]
                    else None
                ),
                loss=measurement["loss"],
                eligible=measurement["eligible"],
                duration_s=duration,
                limit_hits=hits,
            )
        ),
        flush=True,
    )
    return row


def _adaptation_round(simulation, parameters, steps, iteration, run, cfg):
    """Compare a baseline with both bounded probes; accept only lower valid loss."""
    name = cfg["adaptation"]["parameters"][iteration % len(parameters)]
    lower, upper = cfg["adaptation"]["bounds_us"][name]
    rows = [_adaptation_segment(simulation, parameters, "baseline", run, cfg)]
    rows[0]["iteration"] = iteration
    if not rows[0]["online_measurement"]["eligible"]:
        decision = dict(
            iteration=iteration,
            parameter=name,
            before=dict(parameters),
            after=dict(parameters),
            accepted=False,
            relative_step=steps[name],
            reason="baseline_not_settled",
        )
        print(json.dumps(decision), flush=True)
        return dict(parameters), rows, decision
    # Reverse probe order on successive visits to the same coordinate.
    directions = (-1, 1) if (iteration // len(parameters)) % 2 == 0 else (1, -1)
    for direction in directions:
        candidate = dict(parameters)
        candidate[name] = min(
            upper, max(lower, parameters[name] * (1 + direction * steps[name]))
        )
        row = _adaptation_segment(simulation, candidate, "probe", run, cfg)
        row["iteration"] = iteration
        rows.append(row)
    valid = [row for row in rows if row["online_measurement"]["eligible"]]
    best = (
        min(valid, key=lambda row: row["online_measurement"]["loss"])
        if valid
        else rows[0]
    )
    baseline_loss = rows[0]["online_measurement"]["loss"]
    improvement = (
        best is not rows[0]
        and best["online_measurement"]["eligible"]
        and (
            baseline_loss is None
            or baseline_loss - best["online_measurement"]["loss"]
            > cfg["adaptation"]["minimum_loss_improvement"]
        )
    )
    updated = dict(best["parameters"] if improvement else parameters)
    decision = dict(
        iteration=iteration,
        parameter=name,
        before=dict(parameters),
        after=updated,
        accepted=bool(improvement),
        relative_step=steps[name],
        reason="lower_settled_loss" if improvement else "no_settled_improvement",
    )
    if not improvement:
        steps[name] = max(
            cfg["adaptation"]["minimum_relative_step"],
            steps[name] * cfg["adaptation"]["step_decay"],
        )
    print(json.dumps(decision), flush=True)
    return updated, rows, decision


def adaptation_check(*, run_id=None, **overrides):
    """Test bounded online coordinate adaptation and an independent fresh restart."""
    cfg = adaptation_configuration(**overrides)
    parameters = dict(
        tonic_us=cfg["tonic_us"], sensory_drive_us=cfg["weights"]["sensory_drive"]
    )
    steps = dict.fromkeys(parameters, cfg["adaptation"]["initial_relative_step"])
    segments, decisions = [], []
    with stage_run(REPO, "exp152", "compute", run_id=run_id, configuration=cfg) as run:
        simulation = AdaptiveJointSimulation(cfg)
        try:
            segments.append(
                _adaptation_segment(
                    simulation,
                    parameters,
                    "initial_validation",
                    run,
                    cfg,
                    validation=True,
                )
            )
            for iteration in range(cfg["adaptation"]["iterations"]):
                parameters, rows, decision = _adaptation_round(
                    simulation, parameters, steps, iteration, run, cfg
                )
                segments.extend(rows)
                decisions.append(decision)
            segments.append(
                _adaptation_segment(
                    simulation,
                    parameters,
                    "frozen_validation",
                    run,
                    cfg,
                    validation=True,
                )
            )
            next_index = simulation.segment_index
            simulation.close()
            simulation = AdaptiveJointSimulation(cfg)
            simulation.segment_index = next_index
            segments.append(
                _adaptation_segment(
                    simulation,
                    parameters,
                    "reset_validation",
                    run,
                    cfg,
                    validation=True,
                )
            )
        finally:
            simulation.close()
        write_json_atomic(
            run.export / "adaptation.json",
            dict(
                segments=segments,
                decisions=decisions,
                final_parameters=parameters,
            ),
        )
        run.record.setdefault("scientific_artifacts", {})["snapshot_stages"] = {
            path.parent.parent.relative_to(run.export).as_posix(): ["start"]
            for path in run.export.rglob("snapshot.json")
        }
    return run.run_id


def consolidate_compute(
    source, *, cycle_validation_source=None, archive_source=None, run_id=None
):
    """Import fixed evidence with archived lineage, without rerunning acquisition.

    Original records and inventories replace operational dependencies on tuning
    history. Raw outputs and native snapshots remain byte-identical; a replacement
    validation must itself contain exactly the original validation results.
    """
    upstream = source_run(
        REPO / ".pingstore", source, stage="compute", experiment="exp152"
    )
    records = validated_run_graph(REPO / ".pingstore/runs", {archive_source or source})
    if source not in records:
        raise ValueError(
            "Archive source must include the imported compute in its ancestry"
        )
    inputs = {}
    original_validation = upstream.record["inputs"].get("cycle_validation")
    if bool(original_validation) != bool(cycle_validation_source):
        raise ValueError(
            "Supply replacement validation exactly when the source has one"
        )
    if original_validation:
        validation = source_run(
            REPO / ".pingstore",
            cycle_validation_source,
            stage="analyse",
            experiment="exp152",
        )
        original = source_run(
            REPO / ".pingstore",
            original_validation["run_id"],
            reference=original_validation,
        )
        if file_sha256(validation.file("cycles.json")) != file_sha256(
            original.file("cycles.json")
        ):
            raise ValueError(
                "Replacement validation must preserve the original results"
            )
        original_native = source_run(
            REPO / ".pingstore",
            original.record["inputs"]["compute"]["run_id"],
            reference=original.record["inputs"]["compute"],
        )
        replacement_native = source_run(
            REPO / ".pingstore",
            validation.record["inputs"]["compute"]["run_id"],
            reference=validation.record["inputs"]["compute"],
        )
        if file_sha256(original_native.file("condition-0.npz")) != file_sha256(
            replacement_native.file("condition-0.npz")
        ):
            raise ValueError("Replacement validation must preserve the native trace")
        inputs = {"cycle_validation": validation, "cycle_selection": validation}
    archive = {
        identity: dict(
            record=record,
            inventory=payload_inventory(REPO / ".pingstore/runs" / identity),
            readme=(REPO / ".pingstore/runs" / identity / "README.md").read_text(),
        )
        for identity, record in records.items()
    }
    with stage_run(
        REPO,
        "exp152",
        "compute",
        inputs=inputs,
        run_id=run_id,
        configuration=upstream.record["execution"]["configuration"],
        operation="import-self-contained-evidence",
    ) as run:
        shutil.copytree(upstream.outputs, run.export, dirs_exist_ok=True)
        if payload_inventory(run.directory) != payload_inventory(upstream.directory):
            raise ValueError("Consolidation changed source bytes")
        run.record["scientific_artifacts"] = copy.deepcopy(
            upstream.record.get("scientific_artifacts", {})
        )
        run.record["scientific_execution"] = dict(
            upstream.record["execution"], origin=upstream.record["origin"]
        )
        run.record["consolidated_source"] = dict(
            **upstream.reference,
            archived_lineage="source-records.json",
            note="Byte-preserving import; original input records and checksums are embedded, not live run dependencies. No simulation was executed.",
        )
        write_json_atomic(run.export / "source-records.json", archive)
        # Retain the parameter-screen conclusions supporting the report's tuning account.
        tuning = {}
        for identity, record in records.items():
            if record["stage"] != "analyse":
                continue
            evidence = source_run(
                REPO / ".pingstore",
                identity,
                reference={
                    "run_id": identity,
                    "payload_digest": record["payload_digest"],
                },
            )
            for name in ("cycles.json", "startup.json", "metrics.json"):
                path = evidence.file(name)
                if path.exists():
                    data = load_json(path)
                    if "conditions" in data:
                        data["conditions"] = [
                            {
                                key: value
                                for key, value in row.items()
                                if key not in {"phase", "waveform", "median_waveform"}
                            }
                            for row in data["conditions"]
                        ]
                    tuning[identity] = dict(
                        source_file=name, source_sha256=file_sha256(path), summary=data
                    )
        write_json_atomic(run.export / "selection-history.json", tuning)
        for identity, record in records.items():
            source_run(
                REPO / ".pingstore",
                identity,
                reference={
                    "run_id": identity,
                    "payload_digest": record["payload_digest"],
                },
                phase=record.get("phase"),
            ).check_unchanged()
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = configuration()
    parser.add_argument("--duration", type=float, default=defaults["duration"])
    parser.add_argument("--max-lag-ms", type=float, default=defaults["max_lag_ms"])
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument(
        "--compile", dest="compile_mode", action="store_const", const="default"
    )
    execution.add_argument(
        "--eager", dest="compile_mode", action="store_const", const=None
    )
    parser.add_argument(
        "--native",
        action="store_true",
        help="Validate the fixed recipe without a search input",
    )
    parser.add_argument(
        "--consolidate-source",
        help="Import explicit compute evidence with archived lineage",
    )
    parser.add_argument(
        "--archive-source",
        help="Explicit report presentation whose lineage is embedded on import",
    )
    parser.add_argument("--cycle-source")
    parser.add_argument("--condition-index", type=int)
    parser.add_argument("--cycle-validation-source")
    parser.add_argument(
        "--adapt",
        action="store_true",
        help="Test native online adaptation with frozen and reset validation",
    )
    parser.add_argument("--target-frequency-hz", type=float)
    parser.add_argument("--target-excursion-deg", type=float)
    parser.add_argument("--adaptation-iterations", type=int)
    parser.add_argument("--run-id")
    arguments = vars(parser.parse_args())
    archive = arguments.pop("archive_source")
    native = arguments.pop("native")
    consolidate = arguments.pop("consolidate_source")
    source = arguments.pop("cycle_source")
    index = arguments.pop("condition_index")
    validation = arguments.pop("cycle_validation_source")
    adaptive = arguments.pop("adapt")
    adaptation_arguments = dict(
        target_frequency_hz=arguments.pop("target_frequency_hz"),
        target_excursion_deg=arguments.pop("target_excursion_deg"),
        iterations=arguments.pop("adaptation_iterations"),
    )
    if not adaptive and any(
        value is not None for value in adaptation_arguments.values()
    ):
        parser.error("Adaptation settings require --adapt")
    if archive is not None and consolidate is None:
        parser.error("--archive-source applies only to consolidation")
    if (source is None) != (index is None):
        parser.error("Use --cycle-source and --condition-index together")
    if adaptive:
        if (
            native
            or source is not None
            or consolidate is not None
            or validation is not None
        ):
            parser.error("--adapt is a separate native protocol")
        if (
            arguments["duration"] != defaults["duration"]
            or arguments["max_lag_ms"] != defaults["max_lag_ms"]
        ):
            parser.error(
                "--adapt uses fixed measurement durations and no ROS lag budget"
            )
        run_id = adaptation_check(
            run_id=arguments["run_id"],
            compile_mode=arguments["compile_mode"],
            **{
                key: value
                for key, value in adaptation_arguments.items()
                if value is not None
            },
        )
    elif consolidate is not None:
        if native or source is not None:
            parser.error("Consolidation cannot execute a native simulation")
        run_id = consolidate_compute(
            consolidate,
            cycle_validation_source=validation,
            archive_source=archive,
            run_id=arguments["run_id"],
        )
    elif native or source is not None:
        if validation is not None:
            parser.error("--cycle-validation-source applies only to ROS acquisition")
        run_id = cycle_check(source, condition_index=index, **arguments)
    else:
        run_id = compute(cycle_validation_source=validation, **arguments)
    print(run_id, flush=True)
