"""Measure behaviour, model progress, delivery ages and absolute deadline misses.

This stage consumes one explicit raw acquisition and preserves every original
trace field in traces.npz. A short callback can still miss an absolute deadline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]
from experiments.exp152.recipe import SIDES
from pingstore.library import load_json, source_run, stage_run, write_json_atomic

NODES = ("flexor", "extensor", "plant")
TOPICS = tuple(
    "/" + kind + "/" + side for kind in ("motor", "sensory") for side in SIDES
)


def stats(values):
    data = np.asarray(values, dtype=float)
    if not len(data):
        return None
    return dict(
        n=len(data),
        mean=float(np.mean(data)),
        sd=float(np.std(data, ddof=1)) if len(data) > 1 else None,
        median=float(np.median(data)),
        p95=float(np.percentile(data, 95)),
        maximum=float(np.max(data)),
    )


def run_summary(summary):
    """Summarise observed callbacks; SD is the sample SD, not the standard error."""
    lines = ["Callback duration: mean ± sample SD; missed deadlines / observed updates"]
    for name in NODES:
        callback = summary["full_callback_ms"][name]
        if callback is None:
            lines.append(f"{name.capitalize()}: no callbacks recorded")
            continue
        sd = f"{callback['sd']:.3f}" if callback["sd"] is not None else "undefined"
        misses = summary["step_deadline_misses"][name]
        lines.append(
            f"{name.capitalize()}: {callback['mean']:.3f} ± {sd} ms; "
            f"{misses:,} / {callback['n']:,} missed deadlines"
        )
    lines.append(
        f"Behaviour: {'PASS' if summary['passed'] else 'FAIL'}; "
        f"timing: {'PASS' if summary['timing_passed'] else 'FAIL'}; "
        f"stop reasons: {summary['stop_reasons']}"
    )
    return "\n".join(lines)


def collect(records, spikes, start_ns, args, ready, error):
    ticks = {
        name: [r for r in rows if r["kind"] == "tick"] for name, rows in records.items()
    }
    deliveries = {
        name: [r for r in rows if r["kind"] == "delivery"]
        for name, rows in records.items()
    }

    def relative(stamp):
        return (stamp - start_ns) / 1e9

    populations = args.controller_populations
    channels = len(populations)
    values = {}
    for name, rows in ticks.items():
        values[name + "_time_s"] = np.array(
            [r.get("schedule_time_s", relative(r["stamp_ns"])) for r in rows]
        )
        for field in ("clock_offset_change_ms", "clock_sample_span_ms"):
            values[name + "_" + field] = np.array(
                [r.get(field, float("nan")) for r in rows]
            )
        values[name + "_compute_ms"] = np.array([r["compute_ms"] for r in rows])
        values[name + "_callback_ms"] = np.array([r["callback_ms"] for r in rows])
        values[name + "_lateness_ms"] = np.array(
            [r["deadline_lateness_ms"] for r in rows]
        )
        values[name + "_model_time_s"] = np.array([r["model_time_s"] for r in rows])
        values[name + "_lag_ms"] = np.array([r["wall_lag_ms"] for r in rows])
        if name in SIDES:
            values[name + "_voltage"] = np.array([r["voltage"] for r in rows]).reshape(
                -1, channels
            )
            values[name + "_substep_voltage"] = np.array(
                [r["substep_voltage"] for r in rows]
            ).reshape(-1, args.substeps, channels)
            values[name + "_substep_spikes"] = np.array(
                [r["substep_spikes"] for r in rows]
            ).reshape(-1, args.substeps, channels)
            values[name + "_spikes"] = np.array([r["spikes"] for r in rows]).reshape(
                -1, channels
            )
        rows = deliveries[name]
        values[name + "_delivery_time_s"] = np.array(
            [relative(r["stamp_ns"]) for r in rows]
        )
        values[name + "_delivery_ms"] = np.array([r["age_ms"] for r in rows])
    plant = ticks["plant"]
    for key in ("angle", "velocity", "torque", "gravity_torque"):
        values[key] = np.array([r[key] for r in plant])
    values["activation"] = np.array([r["activation"] for r in plant]).reshape(-1, 2)
    for topic, rows in spikes.items():
        key = topic.strip("/").replace("/", "_")
        values[key + "_time_s"] = np.array([relative(r[1]) for r in rows])
        values[key + "_model_time_s"] = np.array([r[3] / 1e9 for r in rows])
        values[key + "_neuron"] = np.array([r[2] for r in rows], dtype=int)
    velocity = values["velocity"]
    moving = np.sign(
        velocity[np.abs(velocity) > args.analysis["movement_velocity_threshold_rad_s"]]
    )
    reversals = int(np.count_nonzero(np.diff(moving)))
    expected = round(args.duration * 1000 / args.timer_ms)
    progress = {
        name: rows[-1]["model_time_s"] if rows else 0.0 for name, rows in ticks.items()
    }
    missing = {
        name: sum(r["missing"] for r in rows) for name, rows in deliveries.items()
    }
    compute = {name: stats(values[name + "_compute_ms"]) for name in NODES}
    callback = {name: stats(values[name + "_callback_ms"]) for name in NODES}
    lateness = {name: stats(values[name + "_lateness_ms"]) for name in NODES}
    delivery = {name: stats(values[name + "_delivery_ms"]) for name in NODES}
    callback_overruns = {
        name: int((values[name + "_callback_ms"] > args.timer_ms).sum())
        for name in NODES
    }
    overruns = {
        name: int((values[name + "_compute_ms"] > args.timer_ms).sum())
        for name in NODES
    }
    age_failures = {
        name: int((values[name + "_delivery_ms"] > args.max_spike_age_ms).sum())
        for name in NODES
    }
    step_late = {name: int((values[name + "_lateness_ms"] > 0).sum()) for name in NODES}
    observed_affinity = {
        name: next((r["cpu_affinity"] for r in rows if "cpu_affinity" in r), [])
        for name, rows in ticks.items()
    }
    observed_affinity["recorder"] = args.observed_recorder_affinity
    execution = {
        name: next((r["model_info"] for r in ticks[name] if "model_info" in r), {})
        for name in SIDES
    }
    stop_reasons = {
        name: next(
            (r["stop_reason"] for r in rows if r["kind"] == "finished"), "missing"
        )
        for name, rows in records.items()
    }
    limit_hits = max([r["limit_hits"] for r in plant], default=0)
    checks = dict(
        nodes_ready=ready,
        one_cpu_per_node=all(
            observed_affinity[n] == [cpu] for n, cpu in args.cpu_allocation.items()
        ),
        configured_controller_execution=all(
            info.get("compile_mode", "missing") == args.compile_mode
            and info.get("execution_path") in {"compiled_blocks", "kernel_loop"}
            for info in execution.values()
        ),
        no_process_error=error is None,
        complete_diagnostic_sequence=all(
            len(values[name + "_model_time_s"]) > 0
            and np.allclose(
                values[name + "_model_time_s"],
                np.arange(1, len(values[name + "_model_time_s"]) + 1)
                * args.timer_ms
                / 1000,
                rtol=0,
                atol=1e-9,
            )
            for name in NODES
        ),
        diagnostics_exported=all(
            reason != "missing" for reason in stop_reasons.values()
        ),
        no_safety_stop=all(reason == "completed" for reason in stop_reasons.values()),
        all_nodes_recorded=all(ticks.values()),
        finite_neural_traces=all(
            np.isfinite(values[name + "_substep_voltage"]).all() for name in SIDES
        ),
        finite_joint_state=bool(
            np.isfinite(values["angle"]).all() and np.isfinite(velocity).all()
        ),
        both_motor_outputs=all(spikes["/motor/" + s] for s in SIDES),
        both_sensory_streams=all(spikes["/sensory/" + s] for s in SIDES),
        repeated_movement=reversals >= args.analysis["minimum_reversals"],
        no_mechanical_stops=limit_hits == 0,
        no_detected_spike_loss=not any(missing.values()),
    )
    if getattr(args, "joint", {}).get("angle_convention"):
        checks["both_late_sensory_streams"] = all(
            np.any(values["sensory_" + s + "_model_time_s"] >= args.duration / 2)
            for s in SIDES
        )
    timing_checks = dict(
        full_model_progress=all(len(rows) >= expected - 1 for rows in ticks.values()),
        callback_within_budget=not any(callback_overruns.values()),
        compute_within_budget=not any(overruns.values()),
        steps_completed_by_deadline=not any(step_late.values()),
        spike_delivery_within_budget=not any(age_failures.values()),
    )
    wall_seconds = max(
        (
            float(values[name + "_time_s"][-1])
            for name in NODES
            if len(values[name + "_time_s"])
        ),
        default=0,
    )
    return values, dict(
        mode="ros2",
        observed_wall_seconds=wall_seconds,
        timing_profile=dict(enabled=False),
        passed=all(checks.values()),
        checks=checks,
        error=error,
        timing_passed=all(timing_checks.values()),
        timing_checks=timing_checks,
        duration_seconds=args.duration,
        dt_ms=args.dt_ms,
        timer_ms=args.timer_ms,
        substeps_per_update=args.substeps,
        neurons_per_population=args.neurons,
        total_neurons=2 * channels * args.neurons,
        controller_populations=populations,
        snnlab_ref=args.snnlab_ref,
        cpu_allocation=args.cpu_allocation,
        observed_cpu_affinity=observed_affinity,
        controller_execution=execution,
        model_seconds_per_node=progress,
        simulated_seconds_per_wall_second=min(progress.values()) / wall_seconds
        if wall_seconds
        else 0,
        joint_direction_reversals=reversals,
        mechanical_stop_hits=limit_hits,
        angle_range_rad=[float(values["angle"].min()), float(values["angle"].max())]
        if len(plant)
        else None,
        gravity_torque_nm=stats(values.get("gravity_torque", [])),
        mean_activation={
            s: float(values["activation"][:, i].mean()) for i, s in enumerate(SIDES)
        }
        if plant
        else None,
        motor_spikes={s: len(spikes["/motor/" + s]) for s in SIDES},
        sensory_spikes={s: len(spikes["/sensory/" + s]) for s in SIDES},
        update_compute_ms=compute,
        full_callback_ms=callback,
        deadline_lateness_ms=lateness,
        clock_diagnostics={
            name: dict(
                mapping=next(
                    (r["clock_mapping"] for r in rows if "clock_mapping" in r), None
                ),
                offset_change_ms=dict(
                    stats(values[name + "_clock_offset_change_ms"]),
                    minimum=float(values[name + "_clock_offset_change_ms"].min()),
                )
                if rows
                else None,
                sample_span_ms=stats(values[name + "_clock_sample_span_ms"]),
            )
            for name, rows in ticks.items()
        },
        callback_overruns=callback_overruns,
        schedule="Absolute releases from shared epoch; catch up without dropping windows or resetting deadlines; stop if start lag exceeds max_lag_ms",
        max_lag_ms=args.max_lag_ms,
        stop_reasons=stop_reasons,
        spike_delivery_ms=delivery,
        compute_overruns=overruns,
        step_deadline_misses=step_late,
        late_spikes=age_failures,
        missing_spikes=missing,
        spike_age_budget_ms=args.max_spike_age_ms,
        timing_scope="Releases, start lag, callback duration, deadline lateness and tick plot times use perf_counter_ns against the mapped monotonic origin. The legacy wall_lag_ms tick field now contains monotonic release lag. Clock diagnostics report changes in ROS-minus-monotonic offset since startup, with bracketing sample spans. Full callbacks include model work, spike publication and diagnostic-buffer writes. JSON conversion and diagnostic publication occur after control. ROS input callbacks execute between releases. Spike timestamps, delivery ages and spike plot times still use ROS time and can be affected by clock adjustments; delivery age excludes pre-publication buffering. Inputs enter at the first substep and outputs publish after the window. Late spikes are still applied.",
    )


def cycle_metrics(
    angle, velocity, *, dt_s=0.001, reference_angle=0.0, settings=None, times=None
):
    """Assess final complete Poincare cycles, without selecting a favourable window.

    Upward crossings of the reference angle delimit cycles. Turning-point errors
    and angle RMS waveform errors are normalized by median peak-to-peak excursion;
    velocity RMS errors by the median waveform's peak-to-peak velocity. All errors
    compare each of the final ten cycles against their pointwise median.
    """
    settings = settings or dict(
        cycle_count=10,
        cycle_tolerance=0.01,
        cycle_burn_in_s=5.0,
        cycle_phase_samples=128,
    )
    angle, velocity = np.asarray(angle), np.asarray(velocity)
    time = np.arange(1, len(angle) + 1) * dt_s if times is None else np.asarray(times)
    count, tolerance = settings["cycle_count"], settings["cycle_tolerance"]
    failed = dict(
        passed=False,
        complete_cycles=0,
        period_relative_error=None,
        turning_angle_relative_error=None,
        angle_waveform_relative_error=None,
        velocity_waveform_relative_error=None,
        mean_period_s=None,
        upper_angle_deg=None,
        lower_angle_deg=None,
        cycle_start_s=None,
    )
    if (
        len(angle) < 3
        or not np.isfinite(angle).all()
        or not np.isfinite(velocity).all()
    ):
        return failed
    indices = np.flatnonzero(
        (angle[:-1] <= reference_angle) & (angle[1:] > reference_angle)
    )
    crossing = (
        time[indices]
        + (reference_angle - angle[indices])
        / (angle[indices + 1] - angle[indices])
        * np.diff(time)[indices]
    )
    crossing = crossing[crossing >= settings["cycle_burn_in_s"]]
    failed["complete_cycles"] = max(0, len(crossing) - 1)
    if len(crossing) < count + 1:
        return failed
    crossing = crossing[-count - 1 :]
    periods = np.diff(crossing)
    phases = np.linspace(0, 1, settings["cycle_phase_samples"], endpoint=False)
    query = crossing[:-1, None] + periods[:, None] * phases
    waves = np.stack(
        [np.interp(query, time, signal) for signal in (angle, velocity)], axis=-1
    )
    upper, lower = [], []
    for start, end in zip(crossing[:-1], crossing[1:], strict=True):
        samples = angle[(time >= start) & (time <= end)]
        upper.append(float(samples.max()))
        lower.append(float(samples.min()))
    upper, lower = np.asarray(upper), np.asarray(lower)
    excursion = float(np.median(upper - lower))
    reference = np.median(waves, axis=0)
    velocity_scale = float(np.ptp(reference[:, 1]))
    if excursion <= 0 or velocity_scale <= 0:
        return failed
    period_error = float(np.max(np.abs(periods / np.median(periods) - 1)))
    turning_error = float(
        max(
            np.max(np.abs(upper - np.median(upper))),
            np.max(np.abs(lower - np.median(lower))),
        )
        / excursion
    )
    rms = np.sqrt(np.mean((waves - reference) ** 2, axis=1))
    angle_error, velocity_error = np.max(rms, axis=0) / [excursion, velocity_scale]
    return dict(
        passed=bool(
            max(period_error, turning_error, angle_error, velocity_error) < tolerance
        ),
        complete_cycles=failed["complete_cycles"],
        assessed_cycles=count,
        cycle_start_s=crossing.tolist(),
        periods_s=periods.tolist(),
        mean_period_s=float(periods.mean()),
        period_relative_error=period_error,
        turning_angle_relative_error=turning_error,
        angle_waveform_relative_error=float(angle_error),
        velocity_waveform_relative_error=float(velocity_error),
        upper_angle_deg=np.rad2deg(upper).tolist(),
        lower_angle_deg=np.rad2deg(lower).tolist(),
        minimum_excursion_rad=float((upper - lower).min()),
        phase=phases.tolist(),
        median_waveform=reference.tolist(),
        waveform=waves.tolist(),
    )


def analyse_cycles(upstream, cfg, run_id):
    rows = []
    for index, condition in enumerate(cfg["conditions"]):
        with np.load(upstream.file(f"condition-{index}.npz")) as data:
            trace = data["trace"]
        metadata = load_json(upstream.file(f"condition-{index}.json"))
        metrics = cycle_metrics(
            trace[:, 0],
            trace[:, 1],
            reference_angle=metadata["joint"]["reference_angle"],
            settings=cfg["analysis"],
        )
        metrics.update(
            condition=condition,
            limit_hits=metadata["limit_hits"],
            late_sensory_counts=trace[len(trace) // 2 :, 6:8]
            .sum(0)
            .astype(int)
            .tolist(),
            motor_counts=trace[:, 4:6].sum(0).astype(int).tolist(),
        )
        centre, threshold = (
            metadata["joint"]["reference_angle"],
            metadata["joint"]["sensory_limit"],
        )
        metrics["passed"] = bool(
            metrics["passed"]
            and metadata["limit_hits"] == 0
            and all(metrics["late_sensory_counts"])
            and min(metrics["upper_angle_deg"]) > np.rad2deg(centre + threshold)
            and max(metrics["lower_angle_deg"]) < np.rad2deg(centre - threshold)
        )
        rows.append(metrics)
    with stage_run(
        REPO,
        "exp152",
        "analyse",
        inputs={"compute": upstream},
        run_id=run_id,
        configuration=cfg,
    ) as run:
        write_json_atomic(
            run.export / "cycles.json",
            dict(
                conditions=rows,
                definition="Final ten complete upward reference crossings after 5 s; maximum period and turning-point errors and normalized angle/velocity RMS waveform errors below 1%; both late sensory channels; no stops.",
            ),
        )
    keys = (
        "period_relative_error",
        "turning_angle_relative_error",
        "angle_waveform_relative_error",
        "velocity_waveform_relative_error",
    )
    for index, row in enumerate(rows):
        print(
            index,
            row["condition"],
            "PASS" if row["passed"] else "FAIL",
            {key: row[key] for key in keys},
            flush=True,
        )
    return run.run_id


def adaptation_measure(trace, cfg, *, validation=False, limit_hits=0):
    """Measure complete cycles and the feedback loss, including invalid probes."""
    settings = {
        **cfg["analysis"],
        "cycle_count": (
            cfg["analysis"]["cycle_count"]
            if validation
            else cfg["adaptation"]["measurement_cycles"]
        ),
        "cycle_burn_in_s": cfg["adaptation"]["settling_s"],
    }
    cycles = cycle_metrics(
        trace[:, 0],
        trace[:, 1],
        reference_angle=cfg["joint"]["reference_angle"],
        settings=settings,
    )
    result = dict(
        valid=False,
        eligible=False,
        loss=None,
        frequency_hz=None,
        excursion_rad=None,
        frequency_relative_error=None,
        excursion_relative_error=None,
        limit_hits=int(limit_hits),
        cycles=cycles,
    )
    if limit_hits or cycles["mean_period_s"] is None:
        return result
    frequency = 1.0 / cycles["mean_period_s"]
    excursion = float(
        np.mean(
            np.deg2rad(np.array(cycles["upper_angle_deg"]) - cycles["lower_angle_deg"])
        )
    )
    frequency_error = frequency / cfg["adaptation"]["target_frequency_hz"] - 1
    excursion_error = excursion / cfg["adaptation"]["target_excursion_rad"] - 1
    result.update(
        valid=True,
        eligible=bool(
            cycles["passed"]
            or not cfg["adaptation"].get("require_repeatable_cycles", False)
        ),
        loss=float(frequency_error**2 + excursion_error**2),
        frequency_hz=float(frequency),
        excursion_rad=excursion,
        frequency_relative_error=float(frequency_error),
        excursion_relative_error=float(excursion_error),
    )
    return result


def analyse_adaptation(upstream, cfg, run_id):
    """Independently remeasure all saved probes and fresh/frozen validations."""
    history = load_json(upstream.file("adaptation.json"))
    rows = []
    for row in history["segments"]:
        with np.load(upstream.file(row["trace_file"]), allow_pickle=False) as data:
            trace = data["trace"]
        measured = adaptation_measure(
            trace, cfg, validation=row["validation"], limit_hits=row["limit_hits"]
        )
        online = row["online_measurement"]
        if measured["valid"] != online["valid"]:
            raise ValueError("Saved trace does not reproduce online feedback")
        if measured["eligible"] != online.get("eligible", online["valid"]):
            raise ValueError("Saved trace does not reproduce online eligibility")
        if measured["valid"]:
            fields = (
                "loss",
                "frequency_hz",
                "excursion_rad",
                "frequency_relative_error",
                "excursion_relative_error",
            )
            np.testing.assert_allclose(
                [measured[key] for key in fields],
                [online[key] for key in fields],
                rtol=1e-12,
                atol=1e-12,
                err_msg="Saved trace does not reproduce online feedback",
            )
        rows.append({**row, "measurement": measured})
    if cfg["adaptation"].get("require_repeatable_cycles", False):
        for decision in history["decisions"]:
            evaluated = [
                row for row in rows if row.get("iteration") == decision["iteration"]
            ]
            if not evaluated or evaluated[0]["parameters"] != decision["before"]:
                raise ValueError("Adaptation decision has no matching baseline")
            if decision["accepted"]:
                selected = next(
                    row for row in evaluated if row["parameters"] == decision["after"]
                )
                if not (
                    evaluated[0]["measurement"]["eligible"]
                    and selected["measurement"]["eligible"]
                ):
                    raise ValueError("Unsettled feedback was used for an update")
                if (
                    evaluated[0]["measurement"]["loss"]
                    - selected["measurement"]["loss"]
                    <= cfg["adaptation"]["minimum_loss_improvement"]
                ):
                    raise ValueError("Accepted feedback does not reduce settled loss")
    baseline, final, restarted = (
        next(row for row in rows if row["kind"] == name)["measurement"]
        for name in ("initial_validation", "frozen_validation", "reset_validation")
    )
    tolerance = cfg["adaptation"]["target_relative_tolerance"]

    def reaches_target(measurement):
        return bool(
            measurement["eligible"]
            and max(
                abs(measurement["frequency_relative_error"]),
                abs(measurement["excursion_relative_error"]),
            )
            <= tolerance
        )

    summary = dict(
        configuration=cfg,
        baseline=baseline,
        final=final,
        restarted=restarted,
        final_parameters=history["final_parameters"],
        snapshot_paths=dict(
            initial="segment-000/flexor/start/snapshot.json",
            final=f"segment-{len(rows) - 1:03d}/flexor/start/snapshot.json",
        ),
        decisions=history["decisions"],
        accepted_updates=sum(decision["accepted"] for decision in history["decisions"]),
        segments=rows,
        loss_reduction_fraction=(
            1 - restarted["loss"] / baseline["loss"]
            if baseline["valid"] and restarted["valid"] and baseline["loss"] > 0
            else None
        ),
        reached_target=reaches_target(final) and reaches_target(restarted),
        total_limit_hits=sum(row["limit_hits"] for row in rows),
        invalid_segments=sum(not row["measurement"]["valid"] for row in rows),
        unsettled_segments=sum(
            not row["validation"]
            and row["measurement"]["valid"]
            and not row["measurement"]["eligible"]
            for row in rows
        ),
        scope="One deterministic native start; no ROS timing or cross-condition robustness claim",
    )
    with stage_run(
        REPO,
        "exp152",
        "analyse",
        inputs={"compute": upstream},
        run_id=run_id,
        configuration=cfg,
    ) as run:
        write_json_atomic(run.export / "adaptation.json", summary)
    print(
        {
            key: summary[key]
            for key in (
                "loss_reduction_fraction",
                "reached_target",
                "total_limit_hits",
                "final_parameters",
            )
        },
        flush=True,
    )
    return run.run_id


def analyse(source, *, run_id=None):
    upstream = source_run(
        REPO / ".pingstore", source, stage="compute", experiment="exp152"
    )
    cfg = upstream.record["execution"]["configuration"]
    if cfg.get("protocol") == "adaptation_check":
        return analyse_adaptation(upstream, cfg, run_id)
    if cfg.get("protocol") == "cycle_check":
        return analyse_cycles(upstream, cfg, run_id)
    raw = load_json(upstream.file("diagnostics.json"))
    args = argparse.Namespace(
        **cfg,
        cpu_allocation=raw["cpu_allocation"],
        observed_recorder_affinity=raw["observed_recorder_affinity"],
    )
    values, summary = collect(
        raw["records"], raw["spikes"], raw["start_ns"], args, raw["ready"], raw["error"]
    )
    summary["cycles"] = cycle_metrics(
        values["angle"],
        values["velocity"],
        times=values["plant_model_time_s"],
        reference_angle=cfg["joint"].get("reference_angle", 0.0),
        settings=cfg["analysis"] if "cycle_count" in cfg["analysis"] else None,
    )
    summary["wall_cycles"] = cycle_metrics(
        values["angle"],
        values["velocity"],
        times=values["plant_time_s"],
        reference_angle=cfg["joint"].get("reference_angle", 0.0),
        settings=cfg["analysis"] if "cycle_count" in cfg["analysis"] else None,
    )
    if "cycle_count" in cfg["analysis"]:
        summary["checks"]["repeatable_cycles"] = summary["cycles"]["passed"]
        summary["checks"]["repeatable_wall_cycles"] = summary["wall_cycles"]["passed"]
        peaks = summary["cycles"].get("upper_angle_deg")
        troughs = summary["cycles"].get("lower_angle_deg")
        centre, threshold = (
            cfg["joint"]["reference_angle"],
            cfg["joint"]["sensory_limit"],
        )
        summary["checks"]["cycles_reach_both_thresholds"] = bool(
            peaks
            and troughs
            and min(peaks) > np.rad2deg(centre + threshold)
            and max(troughs) < np.rad2deg(centre - threshold)
        )
        summary["passed"] = all(summary["checks"].values())
    inputs = {"compute": upstream}
    if "cycle_validation" in upstream.record["inputs"]:
        reference = upstream.record["inputs"]["cycle_validation"]
        validation = source_run(
            REPO / ".pingstore",
            reference["run_id"],
            stage="analyse",
            experiment="exp152",
            reference=reference,
        )
        summary["deterministic_cycles"] = load_json(validation.file("cycles.json"))[
            "conditions"
        ][0]
        summary["deterministic_validation_run"] = reference["run_id"]
        summary["deterministic_duration_seconds"] = validation.record["execution"][
            "configuration"
        ]["duration"]
        native_reference = validation.record["inputs"]["compute"]
        native = source_run(
            REPO / ".pingstore",
            native_reference["run_id"],
            stage="compute",
            experiment="exp152",
            reference=native_reference,
        )
        with np.load(native.file("condition-0.npz")) as data:
            expected_trace = data["trace"]
        compared = min(
            len(expected_trace),
            len(values["angle"]),
            *(len(values[name + "_spikes"]) for name in SIDES),
        )
        actual_joint = np.column_stack(
            (
                values["angle"][:compared],
                values["velocity"][:compared],
                values["activation"][:compared],
            )
        )
        actual_motor = np.column_stack(
            [values[name + "_spikes"][:compared, 0] for name in SIDES]
        )
        summary["native_trace_comparison"] = dict(
            steps=compared,
            source_run=native_reference["run_id"],
            exact_joint_state=bool(
                compared and np.array_equal(actual_joint, expected_trace[:compared, :4])
            ),
            exact_motor_spikes=bool(
                compared
                and np.array_equal(actual_motor, expected_trace[:compared, 4:6])
            ),
            maximum_joint_absolute_error=float(
                np.max(np.abs(actual_joint - expected_trace[:compared, :4]))
            )
            if compared
            else None,
        )
        inputs["deterministic_compute"] = native
        inputs["cycle_validation"] = validation
    with stage_run(
        REPO,
        "exp152",
        "analyse",
        inputs=inputs,
        run_id=run_id,
        configuration=cfg,
    ) as run:
        np.savez_compressed(run.export / "traces.npz", **values)
        write_json_atomic(
            run.export / "numbers.json", dict(configuration=cfg, **summary)
        )
    print(run_summary(summary))
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-id")
    analyse(**vars(parser.parse_args()))
