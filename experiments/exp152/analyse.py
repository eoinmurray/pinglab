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
from experiments.exp152.recipe import SIDES, SNNLAB_REF
from pingstore.contracts import load_json, write_json_atomic
from pingstore.stages import source_run, stage_run

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


def profile_summary(values, ticks, period_ms):
    """Keep CPU work and non-CPU elapsed time distinct; neither establishes causality."""
    nodes = {}

    def span_stats(samples):
        if not len(samples):
            return None
        return {
            key: dict(
                stats(data), mean=float(np.mean(data)), minimum=float(np.min(data))
            )
            for key, data in (
                ("wall_ms", samples[:, 0]),
                ("thread_cpu_ms", samples[:, 1]),
                ("wall_minus_thread_cpu_ms", samples[:, 0] - samples[:, 1]),
            )
        }

    for name, rows in ticks.items():
        names = next((r["profile_stages"] for r in rows if "profile_stages" in r), [])
        if not names:
            continue
        samples = np.array([r["stage_timing_ms"] for r in rows])
        values[name + "_stage_wall_ms"] = samples[:, :, 0]
        values[name + "_stage_cpu_ms"] = samples[:, :, 1]
        late = values[name + "_lateness_ms"] > 0
        callback = samples[:, -1]
        lag = values[name + "_lag_ms"]
        already_late = lag >= period_ms
        cpu_exceeds_remaining = lag + callback[:, 1] > period_ms
        stages = {}
        for index, stage in enumerate(names):
            stages[stage] = dict(
                all_updates=span_stats(samples[:, index]),
                missed_deadlines=span_stats(samples[late, index]),
                callback_cpu_share_percent=float(
                    100 * samples[:, index, 1].sum() / callback[:, 1].sum()
                )
                if callback[:, 1].sum()
                else None,
            )
        other = callback - samples[:, :-1].sum(axis=1)
        stages["callback_other"] = dict(
            all_updates=span_stats(other),
            missed_deadlines=span_stats(other[late]),
            callback_cpu_share_percent=float(
                100 * other[:, 1].sum() / callback[:, 1].sum()
            )
            if callback[:, 1].sum()
            else None,
        )
        nodes[name] = dict(
            stage_order=names,
            clock_probe=next(r["clock_probe"] for r in rows if "clock_probe" in r),
            stages=stages,
            missed_deadline_breakdown=dict(
                total=int(late.sum()),
                already_past_deadline_at_start=int((late & already_late).sum()),
                start_lag_plus_thread_cpu_exceeds_budget=int(
                    (late & ~already_late & cpu_exceeds_remaining).sum()
                ),
                elapsed_callback_gap_pushes_over_deadline=int(
                    (late & ~already_late & ~cpu_exceeds_remaining).sum()
                ),
            ),
        )
    return dict(
        enabled=bool(nodes),
        nodes=nodes,
        units="milliseconds",
        method="perf_counter_ns elapsed wall time; thread_time_ns CPU time of the calling thread",
        interpretation="Wall minus thread CPU is a signed estimate of blocking/descheduling plus clock-probe error. It excludes CPU consumed on other threads and cannot identify a causal library or OS layer. Executor spans include SNNLab, Python, PyTorch and compiled kernels. Start lag is measured separately, before callback entry. Deadline categories are arithmetic conditions at the measured start time, not causal attributions.",
        instrumentation="Optional --profile-timing; measurements include clock-read and storage overhead. Empty-span probes are reported, not subtracted. Profiling can change timing and spike arrival order. callback_other is the full span minus measured inner spans, including profiling overhead and diagnostic-buffer writes.",
    )


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
                -1, 2
            )
            values[name + "_substep_voltage"] = np.array(
                [r["substep_voltage"] for r in rows]
            ).reshape(-1, args.substeps, 2)
            values[name + "_substep_spikes"] = np.array(
                [r["substep_spikes"] for r in rows]
            ).reshape(-1, args.substeps, 2)
            values[name + "_spikes"] = np.array([r["spikes"] for r in rows]).reshape(
                -1, 2
            )
        rows = deliveries[name]
        values[name + "_delivery_time_s"] = np.array(
            [relative(r["stamp_ns"]) for r in rows]
        )
        values[name + "_delivery_ms"] = np.array([r["age_ms"] for r in rows])
    plant = ticks["plant"]
    for key in ("angle", "velocity", "torque"):
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
    timing_checks = dict(
        full_model_progress=all(len(rows) >= expected - 1 for rows in ticks.values()),
        callback_within_budget=not any(callback_overruns.values()),
        compute_within_budget=not any(overruns.values()),
        steps_completed_by_deadline=not any(step_late.values()),
        spike_delivery_within_budget=not any(age_failures.values()),
    )
    profiling = (
        profile_summary(values, ticks, args.timer_ms)
        if args.profile_timing
        else {"enabled": False}
    )
    return values, dict(
        mode="ros2",
        timing_profile=profiling,
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
        total_neurons=4 * args.neurons,
        snnlab_ref=SNNLAB_REF,
        cpu_allocation=args.cpu_allocation,
        observed_cpu_affinity=observed_affinity,
        controller_execution=execution,
        model_seconds_per_node=progress,
        simulated_seconds_per_wall_second=min(progress.values()) / args.duration,
        joint_direction_reversals=reversals,
        mechanical_stop_hits=limit_hits,
        angle_range_rad=[float(values["angle"].min()), float(values["angle"].max())]
        if len(plant)
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


def startup_metrics(angle, *, dt_s=0.001):
    """Compare extrema against late-run positive/negative amplitudes and periods.

    Settling is the earliest extremum after which all measured extrema and
    same-direction periods stay within 20% of their late-run medians. The first
    150 ms is excluded from extrema detection to avoid tiny onset ripples.
    """
    from scipy.signal import find_peaks

    angle = np.asarray(angle)
    extrema = np.sort(
        np.concatenate(
            [
                find_peaks(angle, prominence=0.08, distance=max(1, round(0.15 / dt_s)))[
                    0
                ],
                find_peaks(
                    -angle, prominence=0.08, distance=max(1, round(0.15 / dt_s))
                )[0],
            ]
        )
    )
    extrema = extrema[extrema * dt_s >= 0.15]
    times, peaks = extrema * dt_s, angle[extrema]
    tail = times >= len(angle) * dt_s / 2
    if len(peaks) < 4 or not all(
        np.count_nonzero(tail & (peaks * sign > 0)) >= 2 for sign in (1, -1)
    ):
        return dict(
            settling_s=None,
            early_mean_amplitude_error=None,
            first_extremum_s=None,
            first_extremum_deg=None,
            extrema_time_s=times.tolist(),
            extrema_deg=np.rad2deg(peaks).tolist(),
            late_amplitude_deg=None,
            amplitude_relative_error=[],
            periods=[],
            maximum_angle_deg=float(np.rad2deg(np.abs(angle).max()))
            if len(angle)
            else None,
        )
    targets = [float(np.median(peaks[tail & (peaks * sign > 0)])) for sign in (1, -1)]
    reference = np.where(peaks > 0, targets[0], targets[1])
    amplitude_error = np.abs(peaks - reference) / np.abs(reference)
    period_rows = []
    for sign in (1, -1):
        side_times = times[peaks * sign > 0]
        periods = np.diff(side_times)
        late = side_times[1:] >= len(angle) * dt_s / 2
        reference_period = float(np.median(periods[late]))
        period_rows.extend(
            (float(t), float(p), float(abs(p / reference_period - 1)))
            for t, p in zip(side_times[1:], periods)
        )
    bad_times = list(times[amplitude_error > 0.2])
    bad_times += [t for t, _, error in period_rows if error > 0.2]
    settled = times[(times > max(bad_times, default=-1)) & (times <= times[-3])]
    # Require two full-period observations after the proposed settling point.
    settled = [t for t in settled if sum(end >= t for end, _, _ in period_rows) >= 2]
    early = times < 3
    return dict(
        extrema_time_s=times.tolist(),
        extrema_deg=np.rad2deg(peaks).tolist(),
        late_amplitude_deg=np.rad2deg(targets).tolist(),
        amplitude_relative_error=amplitude_error.tolist(),
        early_mean_amplitude_error=float(np.mean(amplitude_error[early]))
        if np.any(early)
        else None,
        first_extremum_s=float(times[0]) if len(times) else None,
        first_extremum_deg=float(np.rad2deg(peaks[0])) if len(peaks) else None,
        settling_s=float(settled[0]) if len(settled) else None,
        periods=period_rows,
        maximum_angle_deg=float(np.rad2deg(np.abs(angle).max())),
    )


def analyse_startup(upstream, cfg, run_id):
    rows = []
    for index, condition in enumerate(cfg["conditions"]):
        with np.load(upstream.file(f"condition-{index}.npz")) as data:
            trace = data["trace"]
        metrics = startup_metrics(trace[:, 0])
        metrics.update(
            sensory_counts=trace[:, 6:8].sum(axis=0).astype(int).tolist(),
            first_250ms_maximum_angle_deg=float(
                np.rad2deg(np.abs(trace[:250, 0]).max())
            ),
            first_250ms_maximum_speed_deg_s=float(
                np.rad2deg(np.abs(trace[:250, 1]).max())
            ),
        )
        rows.append(dict(condition=condition, **metrics))
    inputs = {"compute": upstream}
    with stage_run(
        REPO,
        "exp152",
        "analyse",
        inputs=inputs,
        run_id=run_id,
        configuration=cfg,
    ) as run:
        write_json_atomic(
            run.export / "startup.json",
            dict(
                conditions=rows,
                definition="Extrema and full periods within 20% of late-run medians; deterministic 1 ms transport; centred stationary plant",
            ),
        )
    for index, row in enumerate(rows):
        print(
            index,
            row["condition"],
            "settle",
            row["settling_s"],
            "early amplitude error",
            round(row["early_mean_amplitude_error"], 3),
            "first",
            row["first_extremum_s"],
            round(row["first_extremum_deg"], 2),
        )
    print(run.run_id)
    return run.run_id


def analyse(source, *, run_id=None, baseline_source=None):
    upstream = source_run(
        REPO / ".pingstore", source, stage="compute", experiment="exp152"
    )
    cfg = upstream.record["execution"]["configuration"]
    if cfg.get("protocol") == "startup_sweep":
        return analyse_startup(upstream, cfg, run_id)
    raw = load_json(upstream.file("diagnostics.json"))
    args = argparse.Namespace(
        **cfg,
        cpu_allocation=raw["cpu_allocation"],
        observed_recorder_affinity=raw["observed_recorder_affinity"],
    )
    values, summary = collect(
        raw["records"], raw["spikes"], raw["start_ns"], args, raw["ready"], raw["error"]
    )
    summary["startup"] = startup_metrics(values["angle"], dt_s=cfg["timer_ms"] / 1000)
    inputs = {"compute": upstream}
    if baseline_source is not None:
        baseline = source_run(
            REPO / ".pingstore", baseline_source, stage="analyse", experiment="exp152"
        )
        baseline_cfg = baseline.record["execution"]["configuration"]
        with np.load(baseline.file("traces.npz")) as data:
            reference = startup_metrics(
                data["angle"], dt_s=baseline_cfg["timer_ms"] / 1000
            )
        summary["startup_comparison"] = dict(
            baseline_run=baseline_source,
            baseline=reference,
            candidate=summary["startup"],
        )
        inputs["baseline"] = baseline
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
    parser.add_argument("--baseline-source")
    parser.add_argument("--run-id")
    analyse(**vars(parser.parse_args()))
