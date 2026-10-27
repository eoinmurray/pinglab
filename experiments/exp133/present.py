"""Render the map and coupling-onset figures used by the article.

Require explicit map and onset analysis IDs. Save SVG figures, retained circuit
diagrams and measured numerical summaries in one Pingstore presentation run."""

import argparse
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]
import matplotlib.pyplot as plt
import numpy as np
from experiments import theme
from experiments.exp133 import recipe
from experiments.exp133.custom import evidence, inputs, sender_receiver
from pingstore.library import PingstoreError, load_json, write_json_atomic


def edges(centres):
    centres = np.asarray(centres)
    if (
        len(centres) < 2
        or not np.isfinite(centres).all()
        or np.any(np.diff(centres) <= 0)
    ):
        raise ValueError("map needs at least two distinct finite ordered detunings")
    middle = (centres[:-1] + centres[1:]) / 2
    return np.r_[
        centres[0] - (middle[0] - centres[0]),
        middle,
        centres[-1] + (centres[-1] - middle[-1]),
    ]


def plot_map(data, output):
    """Render the original Fig. 1A concentration map in the shared house style."""
    theme.apply()
    order = np.argsort(data["detuning_hz"])
    x = edges(data["detuning_hz"][order])
    y = edges(data["coupling_us"])
    fig, ax = plt.subplots(figsize=(7.1, 5.3), layout="constrained")
    values = np.ma.masked_where(
        ~data["valid"][:, order], data["concentration"][:, order]
    )
    artist = ax.pcolormesh(
        x,
        y,
        values,
        cmap="pinglab_brand",
        vmin=0,
        vmax=1,
        shading="flat",
        rasterized=True,
    )
    ax.set_facecolor(theme.GREY_LIGHT)
    ax.set(
        xlabel="uncoupled detuning fA − fB (Hz)",
        ylabel="equal reciprocal coupling (μS)",
        ylim=(0, data["coupling_us"].max()),
    )
    drive_axis = ax.secondary_xaxis("top")
    drive_axis.set_xticks(
        data["detuning_hz"][order],
        labels=[f"{value:g}" for value in data["input_difference_hz"][order]],
    )
    drive_axis.set_xlabel("input drive difference A − B (Hz)")
    fig.colorbar(artist, ax=ax, label="PLV (dimensionless)")
    fig.savefig(output)
    plt.close(fig)


def plot_sender_receiver(spikes, phase, frequencies, output):
    theme.apply()
    fig, axes = plt.subplots(
        3, 1, figsize=(7.1, 6.5), sharex=True, layout="constrained"
    )
    for ax, name in zip(axes[:2], ("S", "R"), strict=True):
        for pop, offset, color in (
            ("E", 0, theme.INK_BLACK),
            ("I", recipe.N_E, theme.DEEP_RED),
        ):
            step, neuron = np.nonzero(spikes[f"{name}_{pop}"][:, 0])
            ax.scatter(
                step * recipe.DT_MS,
                neuron + offset + 1,
                marker="|",
                s=4,
                linewidths=0.45,
                color=color,
                rasterized=True,
            )
        ax.set(
            ylim=(0, recipe.N_E + recipe.N_I + 1),
            yticks=[1, recipe.N_E, recipe.N_E + recipe.N_I],
            ylabel=f"{name} neuron",
        )
        ax.axhline(recipe.N_E + 0.5, color=theme.GREY_LIGHT, lw=0.5)
        for label, x, align in (("first", 0.02, "left"), ("last", 0.98, "right")):
            frequency = frequencies[name][label]["frequency_hz"]
            text = "—" if frequency is None else f"{frequency:.1f} Hz"
            ax.text(
                x,
                0.97,
                text,
                transform=ax.transAxes,
                ha=align,
                va="top",
                fontsize=theme.SIZE_ANNOTATION,
                bbox=dict(facecolor=theme.PAPER, edgecolor="none", alpha=0.92, pad=2),
            )
    axes[2].plot(phase["time_ms"], phase["phase_cycles"], color=theme.INK_BLACK, lw=1)
    axes[2].set(ylabel="relative phase\nS − R (cycles)", xlabel="time (ms)")
    for ax in axes:
        ax.axvline(sender_receiver.ONSET_MS, color=theme.DEEP_RED, ls="--", lw=1.2)
        ax.set_xlim(0, sender_receiver.DURATION_MS)
        ax.set_xticks(np.arange(0, sender_receiver.DURATION_MS + 1, 500))
    axes[0].set_title(
        f"Sender S: {sender_receiver.DRIVE_HZ['S']:g} Hz input per channel"
    )
    axes[1].set_title(
        f"Receiver R: {sender_receiver.DRIVE_HZ['R']:g} Hz input per channel"
    )
    axes[0].annotate(
        "coupling on",
        xy=(sender_receiver.ONSET_MS, 1),
        xycoords=("data", "axes fraction"),
        xytext=(5, 4),
        textcoords="offset points",
        fontsize=theme.SIZE_ANNOTATION,
        color=theme.DEEP_RED,
    )
    theme.label_panels(axes)
    fig.savefig(output)
    plt.close(fig)


def sender_receiver_sources(identity):
    analysis = inputs.source(REPO, identity, "analyse")
    cfg = sender_receiver.configuration()
    if analysis.record["execution"].get("configuration") != cfg or set(
        analysis.record["inputs"]
    ) != {"compute"}:
        raise PingstoreError("expected sender–receiver analysis with one compute input")
    pin = analysis.record["inputs"]["compute"]
    compute = inputs.source(REPO, pin["run_id"], "compute", reference=pin)
    sender_receiver.compute_contract(compute)
    result = load_json(analysis.export / "results.json")
    if (
        result.get("schema") != "exp133.sender-receiver-analysis/v1"
        or result.get("configuration") != cfg
    ):
        raise PingstoreError("sender–receiver analysis payload mismatch")
    return analysis, compute, result


def present(identity, *, sender_receiver_analysis, run_id=None):
    analysis = inputs.source(REPO, identity, "analyse")
    evidence.configuration(analysis)
    if set(analysis.record["inputs"]) != {"compute"}:
        raise PingstoreError("analysis must pin exactly one grid compute run")
    pin = analysis.record["inputs"]["compute"]
    compute = inputs.source(REPO, pin["run_id"], "compute", reference=pin)
    cfg = evidence.compute_contract(compute)
    result = load_json(analysis.export / "results.json")
    if (
        result.get("schema") != "exp133.analysis/v1"
        or result.get("configuration") != cfg
    ):
        raise PingstoreError("analysis payload and compute recipe disagree")
    with np.load(analysis.export / "metrics.npz", allow_pickle=False) as raw:
        data = dict(raw)
    sources = {"analysis": analysis, "compute": compute}
    sr_analysis, sr_compute, sr_result = sender_receiver_sources(
        sender_receiver_analysis
    )
    sources.update(
        sender_receiver_analysis=sr_analysis, sender_receiver_compute=sr_compute
    )
    result["sender_receiver"] = sr_result
    with np.load(sr_analysis.export / "relative-phase.npz", allow_pickle=False) as raw:
        phase = dict(raw)
    sr = (
        sender_receiver.recording(sr_compute),
        phase,
        sr_result["raster_frequencies"],
    )
    with inputs.execution(
        REPO,
        "present",
        sources=sources,
        run_id=run_id,
    ) as run:
        plot_map(data, run.export / "arnold-tongue.svg")
        mapping = compute.record["scientific_artifacts"]["native_paths"]
        native = "paired-ping.bundle/reports/circuit.svg"
        shutil.copyfile(
            compute.export / mapping[native], run.export / "network-circuit.svg"
        )
        plot_sender_receiver(*sr, run.export / "sender-receiver.svg")
        shutil.copyfile(
            sr_compute.file("sender-receiver.bundle", "reports", "circuit.svg"),
            run.export / "sender-receiver-circuit.svg",
        )
        write_json_atomic(run.export / "numbers.json", result)
    return run.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-id")
    parser.add_argument(
        "--sender-receiver", required=True, help="Explicit sender–receiver analyse run"
    )
    args = parser.parse_args()
    present(
        args.source, run_id=args.run_id, sender_receiver_analysis=args.sender_receiver
    )


if __name__ == "__main__":
    main()
