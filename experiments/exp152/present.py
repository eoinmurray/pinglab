"""Export the callback replay and native controller circuit.

Consumes an explicit analysis run without executing the model. Full traces remain
in its ancestry; replay states appear only at recorded callback completion times.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]
from pingstore.library import load_json, source_run, stage_run, write_json_atomic

NODES = ("flexor", "extensor", "plant")


def recording(values, summary):
    """Keep every timing sample and expose controller states when callbacks finish."""
    cfg = summary["configuration"]
    populations = cfg["controller_populations"]
    if (
        cfg["neuron_model"] != "elevate"
        or populations != ["motor"]
        or cfg["neurons"] != 1
        or cfg["spike_transport"] != "logical"
        or cfg["joint"]["angle_convention"] != "zero_downward_positive_right"
    ):
        raise ValueError("Replay requires the single-cell Elevate logical joint model")
    nodes = {}
    for name in NODES:
        prefix = name + "_"
        times = np.asarray(values[prefix + "time_s"]) * 1000
        callback = np.asarray(values[prefix + "callback_ms"])
        row = dict(
            time_ms=times.tolist(),
            callback_ms=callback.tolist(),
            lag_ms=np.asarray(values[prefix + "lag_ms"]).tolist(),
            lateness_ms=np.asarray(values[prefix + "lateness_ms"]).tolist(),
            model_s=np.asarray(values[prefix + "model_time_s"]).tolist(),
            stop_reason=summary["stop_reasons"][name],
            statistics=summary["full_callback_ms"][name],
        )
        if len(times) != len(callback) or np.any(np.diff(times) < 0):
            raise ValueError(f"Invalid callback timeline for {name}")
        if name != "plant":
            row.update(
                voltage=np.asarray(values[prefix + "voltage"]).tolist(),
                spikes=np.asarray(values[prefix + "spikes"]).tolist(),
            )
        else:
            row.update(
                angle=np.asarray(values["angle"]).tolist(),
                velocity=np.asarray(values["velocity"]).tolist(),
                activation=np.asarray(values["activation"]).tolist(),
            )
            row["gravity_torque"] = np.asarray(values["gravity_torque"]).tolist()
        for field, samples in row.items():
            if not isinstance(samples, list):
                continue
            array = np.asarray(samples)
            if len(array) != len(times) or not np.isfinite(array).all():
                raise ValueError(f"Invalid callback field for {name}: {field}")
        if (
            name != "plant"
            and len(times)
            and (
                np.asarray(row["voltage"]).shape != (len(times), len(populations))
                or np.asarray(row["spikes"]).shape != (len(times), len(populations))
            )
        ):
            raise ValueError(f"Invalid population channels for {name}")
        if name != "plant" and (np.asarray(row["spikes"]) < 0).any():
            raise ValueError(f"Negative spike count for {name}")
        nodes[name] = row
    return dict(
        schema="pinglab.exp152.replay/v2",
        controller_populations=populations,
        mechanics=cfg["joint"],
        duration_ms=summary["duration_seconds"] * 1000,
        period_ms=summary["timer_ms"],
        neuron_model=cfg["neuron_model"],
        voltage_range_mv=[-80, 40],
        neurons_per_population=summary["neurons_per_population"],
        snnlab_ref=summary["snnlab_ref"],
        behaviour_passed=summary["passed"],
        timing_passed=summary["timing_passed"],
        clock="Monotonic callback completion time relative to each node's mapped shared start",
        sensory_time_ms={
            side: (np.asarray(values["sensory_" + side + "_time_s"]) * 1000).tolist()
            for side in ("flexor", "extensor")
        },
        nodes=nodes,
    )


def present(source, *, run_id=None):
    import shutil

    upstream = source_run(
        REPO / ".pingstore", source, stage="analyse", experiment="exp152"
    )
    numbers = load_json(upstream.file("numbers.json"))
    cfg = numbers["configuration"]
    with np.load(upstream.file("traces.npz"), allow_pickle=False) as data:
        values = dict(data)
    payload = recording(values, numbers)
    reference = upstream.record["inputs"]["compute"]
    compute = source_run(
        REPO / ".pingstore",
        reference["run_id"],
        stage="compute",
        experiment="exp152",
        reference=reference,
    )
    with stage_run(
        REPO,
        "exp152",
        "present",
        inputs={"analyse": upstream},
        run_id=run_id,
        configuration=cfg,
    ) as run:
        write_json_atomic(run.export / "circuit.json", payload)
        circuit = compute.file("flexor", "circuit.svg")
        shutil.copyfile(circuit, run.export / "circuit.svg")
        write_json_atomic(run.export / "numbers.json", numbers)
    return run.run_id


def adaptation_figures(summary, compute, output):
    """Show every evaluated segment, accepted updates and fresh-reset movement."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = summary["segments"]
    end_time = np.cumsum([row["duration_s"] for row in rows])
    colour = dict(
        initial_validation="#777777",
        baseline="#16617d",
        probe="#bc8740",
        frozen_validation="#347a48",
        reset_validation="#904e92",
    )
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for row, time in zip(rows, end_time, strict=True):
        measured = row["measurement"]
        if not measured["valid"]:
            continue
        for ax, value in zip(
            axes.flat,
            [
                measured["frequency_hz"],
                np.rad2deg(measured["excursion_rad"]),
                measured["loss"],
            ],
            strict=False,
        ):
            ax.scatter(
                time,
                value,
                color=colour[row["kind"]],
                s=28,
                marker="o" if measured.get("eligible", True) else "x",
            )
    settings = summary["configuration"]["adaptation"]
    for ax, target in zip(
        axes[0],
        [settings["target_frequency_hz"], np.rad2deg(settings["target_excursion_rad"])],
        strict=True,
    ):
        ax.axhline(target, color="#222222", linestyle="--", linewidth=1)
        tolerance = settings["target_relative_tolerance"]
        ax.axhspan(
            target * (1 - tolerance),
            target * (1 + tolerance),
            color="#347a48",
            alpha=0.08,
        )
    axes[0, 0].set_ylabel("Joint frequency (Hz)")
    axes[0, 1].set_ylabel("Peak-to-peak excursion (degrees)")
    axes[1, 0].set_ylabel("Squared relative error sum")
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.set_xlabel("Accumulated simulated time (s)")
        ax.grid(alpha=0.2)
    decisions = summary["decisions"]
    for name, label in (
        ("tonic_us", "Tonic excitation"),
        ("sensory_drive_us", "Sensory weight"),
    ):
        values = [decisions[0]["before"][name]] + [
            row["after"][name] for row in decisions
        ]
        axes[1, 1].step(range(len(values)), values, where="post", label=label)
    axes[1, 1].set(xlabel="Adaptation iteration", ylabel="Accepted conductance (µS)")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(alpha=0.2)
    from matplotlib.lines import Line2D

    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            color=value,
            label=key.replace("_", " "),
        )
        for key, value in colour.items()
    ]
    if any(not row["measurement"].get("eligible", True) for row in rows):
        handles.append(
            Line2D(
                [],
                [],
                marker="x",
                linestyle="none",
                color="#222222",
                label="unsettled; not used for updates",
            )
        )
    fig.legend(handles=handles, loc="outside lower center", ncol=5, fontsize=8)
    fig.suptitle("exp152: bounded adaptation; one deterministic start", fontsize=13)
    for suffix in ("png", "svg"):
        fig.savefig(output / f"adaptation.{suffix}", dpi=180)
    plt.close(fig)

    adaptation_trace_figure(summary, compute, output, colour)


def adaptation_comparison_figure(current, previous, output):
    """Compare independent reset assays for the initial and revised protocols."""
    import matplotlib.pyplot as plt

    labels = ["Initial", "First attempt", "Settled feedback"]
    measurements = [current["baseline"], previous["restarted"], current["restarted"]]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6), constrained_layout=True)
    settings = current["configuration"]["adaptation"]
    values = [
        [
            row["frequency_hz"] if row["frequency_hz"] is not None else np.nan
            for row in measurements
        ],
        [
            np.rad2deg(row["excursion_rad"])
            if row["excursion_rad"] is not None
            else np.nan
            for row in measurements
        ],
        [row["loss"] if row["loss"] is not None else np.nan for row in measurements],
    ]
    targets = [
        settings["target_frequency_hz"],
        np.rad2deg(settings["target_excursion_rad"]),
    ]
    for index, (ax, data, label) in enumerate(
        zip(
            axes,
            values,
            [
                "Frequency (Hz)",
                "Peak-to-peak excursion (degrees)",
                "Squared relative error sum",
            ],
            strict=True,
        )
    ):
        bars = ax.bar(labels, data, color=["#777777", "#bc8740", "#347a48"])
        ax.bar_label(
            bars,
            labels=[
                f"{value:.3g}" if np.isfinite(value) else "Unavailable"
                for value in data
            ],
            padding=3,
            fontsize=8,
        )
        ax.set_ylabel(label)
        ax.tick_params(axis="x", labelrotation=20)
        ax.grid(axis="y", alpha=0.2)
        if index < 2:
            ax.axhline(targets[index], color="#222222", linestyle="--", linewidth=1)
        ceiling = max(
            [value for value in data if np.isfinite(value)] + targets[index : index + 1]
        )
        ax.set_ylim(0, 1.15 * ceiling)
    for suffix in ("png", "svg"):
        fig.savefig(output / f"adaptation-comparison.{suffix}", dpi=180)
    plt.close(fig)


def adaptation_trace_figure(summary, compute, output, colour):
    """Display the last ten seconds of the initial and learned reset assays."""
    import matplotlib.pyplot as plt

    rows = summary["segments"]
    fig, axes = plt.subplots(
        2, 1, figsize=(10, 5), sharex=True, constrained_layout=True
    )
    for ax, kind, title in zip(
        axes,
        ("initial_validation", "reset_validation"),
        ("Initial parameters, frozen", "Learned parameters, fresh reset"),
        strict=True,
    ):
        row = next(row for row in rows if row["kind"] == kind)
        with np.load(compute.file(row["trace_file"]), allow_pickle=False) as data:
            angle = np.rad2deg(
                data["trace"][-10000:, 0]
                - summary["configuration"]["joint"]["reference_angle"]
            )
        ax.plot(np.arange(len(angle)) / 1000, angle, color=colour[kind], linewidth=1)
        ax.set(ylabel="Angle about centre (degrees)", title=title)
        ax.grid(alpha=0.2)
    duration = summary["configuration"]["adaptation"]["validation_duration_s"]
    axes[-1].set_xlabel(f"Time within final 10 s of each {duration:g} s validation (s)")
    for suffix in ("png", "svg"):
        fig.savefig(output / f"adaptation-traces.{suffix}", dpi=180)
    plt.close(fig)


def present_adaptation(source, *, reference_present, run_id=None):
    """Add native adaptation evidence while preserving an explicit ROS replay."""
    import shutil

    upstream = source_run(
        REPO / ".pingstore", source, stage="analyse", experiment="exp152"
    )
    summary = load_json(upstream.file("adaptation.json"))
    reference = source_run(
        REPO / ".pingstore", reference_present, stage="present", experiment="exp152"
    )
    analysis_reference = reference.record["inputs"]["analyse"]
    ros_analysis = source_run(
        REPO / ".pingstore",
        analysis_reference["run_id"],
        stage="analyse",
        experiment="exp152",
        reference=analysis_reference,
    )
    compute_reference = upstream.record["inputs"]["compute"]
    compute = source_run(
        REPO / ".pingstore",
        compute_reference["run_id"],
        stage="compute",
        experiment="exp152",
        reference=compute_reference,
    )
    numbers = load_json(reference.file("numbers.json"))
    previous = numbers.get("adaptation_test")
    if previous is not None:
        numbers["adaptation_previous_test"] = previous
    numbers["adaptation_test"] = {
        key: value
        for key, value in summary.items()
        if key not in ("segments", "decisions")
    }
    with stage_run(
        REPO,
        "exp152",
        "present",
        run_id=run_id,
        configuration=numbers["configuration"],
        inputs={
            "analyse": ros_analysis,
            "adaptation": upstream,
            "reference_presentation": reference,
        },
    ) as run:
        for name in ("circuit.json", "circuit.svg"):
            shutil.copyfile(reference.file(name), run.export / name)
        adaptation_figures(summary, compute, run.export)
        if previous is not None:
            adaptation_comparison_figure(summary, previous, run.export)
        write_json_atomic(run.export / "adaptation.json", summary)
        write_json_atomic(run.export / "numbers.json", numbers)
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-id")
    parser.add_argument(
        "--reference-present",
        help="Explicit ROS presentation to retain alongside native adaptation evidence",
    )
    arguments = vars(parser.parse_args())
    reference_present = arguments.pop("reference_present")
    if reference_present is None:
        print(present(**arguments), flush=True)
    else:
        print(
            present_adaptation(**arguments, reference_present=reference_present),
            flush=True,
        )
