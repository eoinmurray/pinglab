"""Present one HCO as a four-panel dynamics figure from explicit run evidence.

Run present.py --source ANALYSE_ID. Both voltages, directional conductances,
the voltage phase portrait and injected currents come from the same 200 ms
trajectory. The native compiled circuit is retained for the appendix.
"""

import argparse
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from experiments import theme
from experiments.exp147.recipe import validate_configuration
from pingstore.contracts import load_json
from pingstore.stages import source_run, stage_run


def figure(data, cfg):
    theme.apply()
    fig = plt.figure(figsize=(12, 9), layout="constrained")
    grid = fig.add_gridspec(3, 2, width_ratios=(2.2, 1), height_ratios=(1.5, 1, 0.8))
    voltage_ax = fig.add_subplot(grid[0, 0])
    conductance_ax = fig.add_subplot(grid[1, 0], sharex=voltage_ax)
    phase_ax = fig.add_subplot(grid[:, 1])
    current_ax = fig.add_subplot(grid[2, 0], sharex=voltage_ax)
    initial = cfg["neuron"]["initial_voltage_mv"]
    time = np.r_[0.0, data["time_ms"]]
    for name, target, color in (
        ("n1", "n2", theme.INK_BLACK),
        ("n2", "n1", theme.DEEP_RED),
    ):
        voltage_ax.plot(
            time, np.r_[initial, data[f"{name}_voltage_mv"]], color=color, label=name
        )
        spikes = data[f"{name}_spikes"].astype(bool)
        voltage_ax.scatter(
            data["time_ms"][spikes],
            data[f"{name}_voltage_mv"][spikes],
            color=color,
            s=8,
            zorder=3,
        )
        synapse = cfg["synapse"]
        initial_g = synapse["g_max_us"] / (
            1
            + np.exp(
                -synapse["gamma_per_mv"]
                * (synapse["initial_gate_mv"] - synapse["bias_mv"])
            )
        )
        conductance_ax.plot(
            time,
            np.r_[initial_g, data[f"{name}_to_{target}_conductance_us"]],
            color=color,
            label=f"{name} → {target}",
        )
        current_ax.stairs(
            data[f"{name}_current_na"],
            np.r_[data["input_time_ms"], cfg["duration_ms"]],
            color=color,
            label=name,
            linestyle="-" if name == "n1" else "--",
        )
    x = np.r_[initial, data["n1_voltage_mv"]]
    y = np.r_[initial, data["n2_voltage_mv"]]
    phase_ax.plot(x, y, color=theme.GREY_MID, linewidth=0.8)
    phase_ax.scatter(x[0], y[0], color=theme.INK_BLACK, s=35, label="Start", zorder=4)
    phase_ax.scatter(
        x[-1], y[-1], color=theme.DEEP_RED, marker="s", s=30, label="End", zorder=4
    )
    # Arrows follow actual movement where the trajectory has resolvable extent.
    candidates = np.arange(0, len(x) - 100, 100)
    movement = np.hypot(
        x[candidates + 100] - x[candidates], y[candidates + 100] - y[candidates]
    )
    active = candidates[movement > 10]
    if len(active):
        for index in active[np.linspace(0, len(active) - 1, 4, dtype=int)]:
            phase_ax.annotate(
                "",
                xy=(x[index + 100], y[index + 100]),
                xytext=(x[index], y[index]),
                arrowprops=dict(arrowstyle="->", color=theme.GREY_DARK, lw=1.1),
            )
    voltage_ax.set(ylabel="Membrane voltage (mV)", xlim=(0, cfg["duration_ms"]))
    conductance_ax.set(ylabel="Inhibitory g (µS)")
    current_ax.set(ylabel="Current (nA)", xlabel="Time (ms)", ylim=(-0.08, 1.15))
    phase_ax.set(xlabel="n1 voltage (mV)", ylabel="n2 voltage (mV)", aspect="equal")
    for letter, ax in zip("ABCD", (voltage_ax, conductance_ax, phase_ax, current_ax)):
        ax.text(
            -0.1 if ax != phase_ax else -0.18,
            1.04,
            letter,
            transform=ax.transAxes,
            fontsize=theme.SIZE_PANEL,
            weight="bold",
            va="bottom",
        )
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="upper right", fontsize=8)
    for ax in (voltage_ax, conductance_ax, current_ax):
        for onset in (5, 10):
            ax.axvline(onset, color=theme.GREY_LIGHT, linestyle=":", linewidth=0.7)
    voltage_ax.tick_params(labelbottom=False)
    conductance_ax.tick_params(labelbottom=False)
    return fig


def present(*, source, run_id=None):
    analysed = source_run(
        REPO / ".pingstore", source, experiment="exp147", stage="analyse"
    )
    ref = analysed.record["inputs"]["compute"]
    computed = source_run(
        REPO / ".pingstore",
        ref["run_id"],
        experiment="exp147",
        stage="compute",
        reference=ref,
    )
    cfg = analysed.record["execution"]["configuration"]
    validate_configuration(cfg)
    if (
        cfg != computed.record["execution"]["configuration"]
        or not load_json(analysed.file("results.json"))["independent_euler_passed"]
    ):
        raise ValueError("presentation requires matching validated evidence")
    with stage_run(
        REPO,
        "exp147",
        "present",
        inputs={"analyse": analysed, "compute": computed},
        configuration=cfg,
        run_id=run_id,
    ) as run:
        with np.load(analysed.file("traces.npz")) as data:
            fig = figure(data, cfg)
        for suffix in ("svg", "png"):
            fig.savefig(run.export / f"hco.{suffix}", dpi=180)
        plt.close(fig)
        shutil.copyfile(computed.file("circuit.svg"), run.export / "circuit.svg")
        shutil.copyfile(analysed.file("results.json"), run.export / "numbers.json")
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-id")
    present(**vars(parser.parse_args()))
