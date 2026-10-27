"""Present full-window and 200 ms rasters and four AI diagnostics independently.

Run present.py --source ANALYSE_ID. The analysis run identifies its validated
compute input, whose retained spikes supply the raster after burn-in removal.
Four diagnostic panels summarize the same measurement window.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

import argparse
import json
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from experiments import theme
from experiments.exp145.custom.recording import read_recording
from pingstore.library import source_run, stage_run


def style_circuit(source, target):
    """Restyle the executed bundle's SVG without changing its circuit or layout."""
    svg = source.read_text().replace(
        'font-family="Helvetica Neue"',
        'font-family="JetBrains Mono, IBM Plex Mono, Menlo, Consolas, DejaVu Sans Mono, monospace"',
    )
    # Native boxes were sized for a narrower font; keep monospace labels inside them.
    for original, size in (
        ("11.00", theme.SIZE_LABEL),
        ("14.00", theme.SIZE_PANEL),
        ("18.00", theme.SIZE_TITLE),
    ):
        svg = svg.replace(f'font-size="{original}"', f'font-size="{size}"')
    for original, color in {
        "#24292f": theme.INK_BLACK,
        "#aa5b63": theme.DEEP_RED,
        "#536d82": theme.ELECTRIC_CYAN,
        "#59636e": theme.MUTED,
        "#756b5b": theme.GREY_MID,
        "#f3f6f8": theme.PAPER,
        "#f5f4f1": theme.PAPER,
    }.items():
        svg = svg.replace(original, color)
    target.write_text(svg)


def present(args):
    theme.apply()
    plt.rcParams["svg.fonttype"] = "none"
    source = source_run(
        REPO / ".pingstore", args.source, experiment="exp145", stage="analyse"
    )
    # Resolve the recorded ancestry explicitly; no latest-run selection.
    inputs = source.record["inputs"]
    compute_id = inputs["compute"]["run_id"]
    compute = source_run(
        REPO / ".pingstore", compute_id, experiment="exp145", stage="compute"
    )
    cfg = source.record["execution"]["configuration"]
    summary = json.loads(source.file("results.json").read_text())
    start = round(cfg["burn_in_ms"] / cfg["dt_ms"])
    window_ms = cfg["duration_ms"] - cfg["burn_in_ms"]
    with (
        np.load(source.export / "measurements.npz") as data,
        stage_run(
            REPO,
            "exp145",
            "present",
            inputs={"analyse": source, "compute": compute},
            configuration=cfg,
            run_id=args.run_id,
        ) as run,
    ):
        recording = read_recording(compute.unit("recording"), run.scratch / "reader")
        style_circuit(compute.file("circuit.svg"), run.export / "circuit.svg")
        fig, axes = plt.subplots(2, 1, figsize=(11, 7), layout="constrained")
        for population, color, offset in (
            ("E", theme.INK_BLACK, 0),
            ("I", theme.DEEP_RED, cfg["n_e"]),
        ):
            times, cells = np.nonzero(recording.signals[f"{population}.spikes"][start:])
            times_ms = times * cfg["dt_ms"]
            for axis, selected, marker_size in (
                (axes[0], slice(None), 1),
                (axes[1], times_ms < min(200.0, window_ms), 3),
            ):
                axis.scatter(
                    times_ms[selected],
                    cells[selected] + offset,
                    s=marker_size,
                    c=color,
                    linewidths=0,
                    rasterized=True,
                )
        axes[0].set(
            xlim=(0, window_ms),
            xlabel="Time after burn-in (ms)",
            ylabel="Neuron index",
            title="Shared-input conductance network",
        )
        axes[1].set(
            xlim=(0, min(200.0, window_ms)),
            xlabel="Time after burn-in (ms)",
            ylabel="Neuron index",
            title="First 200 ms after burn-in",
        )
        for axis in axes:
            axis.set_ylim(0, cfg["n_e"] + cfg["n_i"])
        theme.label_panels(axes)
        fig.savefig(run.export / "onset.png")
        fig.savefig(run.export / "onset.svg")
        plt.close(fig)
        diagnostics = diagnostic_figure(data, summary, cfg)
        diagnostics.savefig(run.export / "ai-diagnostics.png")
        diagnostics.savefig(run.export / "ai-diagnostics.svg")
        plt.close(diagnostics)
        shutil.copyfile(source.file("results.json"), run.export / "numbers.json")
    return run.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-id")
    print(present(parser.parse_args()))


COLORS = {"E": theme.INK_BLACK, "I": theme.DEEP_RED}


def diagnostic_figure(data, summary, cfg):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    cvax, corrax, scaleax, currentax = axes.ravel()
    finite_cv = np.concatenate(
        [data[f"{p}_cv"][np.isfinite(data[f"{p}_cv"])] for p in ("E", "I")]
    )
    bins = np.linspace(
        0, max(2.5, float(finite_cv.max()) * 1.05 if finite_cv.size else 2.5), 26
    )
    for index, p in enumerate(("E", "I")):
        values = data[f"{p}_cv"]
        values = values[np.isfinite(values)]
        info = summary["populations"][p]
        if values.size:
            cvax.hist(
                values,
                bins=bins,
                density=True,
                histtype="step",
                linewidth=2,
                color=COLORS[p],
                label=p,
            )
        median = (
            f"{info['median_cv']:.2f}"
            if info["median_cv"] is not None
            else "unresolved"
        )
        cvax.text(
            0.98,
            0.95 - index * 0.12,
            f"{p}: median {median}; {info['cv_eligible']}/{info['cells']} eligible",
            transform=cvax.transAxes,
            ha="right",
            va="top",
            color=COLORS[p],
            fontsize=theme.SIZE_ANNOTATION,
        )
    cvax.axvline(1, color=theme.GREY_LIGHT, ls="--", lw=1)
    cvax.set(
        title="Spike irregularity",
        xlabel="Interspike-interval CV",
        ylabel="Density",
        xlim=(bins[0], bins[-1]),
    )
    if not finite_cv.size:
        cvax.text(
            0.5,
            0.45,
            "Insufficient spikes for ISI CV",
            ha="center",
            transform=cvax.transAxes,
        )

    widths = np.asarray(cfg["analysis"]["correlation_bins_ms"])
    for pair, color in (
        ("EE", COLORS["E"]),
        ("II", COLORS["I"]),
        ("EI", theme.ELECTRIC_CYAN),
    ):
        rows = [summary["correlations"][str(float(width))][pair] for width in widths]
        means = np.asarray(
            [np.nan if row["mean"] is None else row["mean"] for row in rows]
        )
        stds = np.asarray(
            [np.nan if row["std"] is None else row["std"] for row in rows]
        )
        corrax.plot(widths, means, "o-", color=color, label=pair[0] + "–" + pair[1])
        corrax.fill_between(widths, means - stds, means + stds, color=color, alpha=0.12)
    corrax.axhline(0, color=theme.GREY_LIGHT, lw=1, ls="--")
    corrax.set(
        title="Pairwise spike-count correlation",
        xlabel="Sliding counting window (ms)",
        ylabel="Mean Pearson r ± pairwise SD",
    )
    corrax.legend()

    for p in ("E", "I"):
        sizes = data[f"{p}_group_sizes"]
        med = data[f"{p}_q_median"]
        slope = summary["populations"][p]["scaling"]["loglog_slope"]
        label = f"{p}: slope {slope:.2f}" if slope is not None else f"{p}: unresolved"
        scaleax.loglog(sizes, med, "o-", color=COLORS[p], label=label, markersize=4)
        scaleax.fill_between(
            sizes, data[f"{p}_q_low"], data[f"{p}_q_high"], color=COLORS[p], alpha=0.15
        )
    reference = np.asarray(
        [min(cfg["analysis"]["group_sizes"]), max(cfg["n_e"], cfg["n_i"])]
    )
    scaleax.loglog(
        reference,
        1 / reference,
        "--",
        color=theme.GREY_MID,
        label="Independent reference: 1/K",
    )
    scaleax.set(
        title="Population-fluctuation scaling",
        xlabel="Neurons averaged, K",
        ylabel="Normalised variance Q(K)",
    )
    scaleax.legend()
    scaleax.text(
        0.03,
        0.04,
        "Shading: 5–95% across random groups",
        transform=scaleax.transAxes,
        fontsize=theme.SIZE_ANNOTATION,
    )

    lag = data["current_lags_ms"]
    for name, color, style in (
        ("EE", theme.INK_BLACK, "--"),
        ("II", theme.DEEP_RED, "-"),
        ("EI", theme.ELECTRIC_CYAN, "-"),
        ("IE", theme.AMBER, "--"),
    ):
        currentax.plot(
            lag, data[f"current_{name}"], color=color, ls=style, lw=1.3, label=name
        )
    currentax.plot(
        lag, data["current_total"], color=theme.INK_BLACK, lw=2, label="Total"
    )
    currentax.plot(
        lag,
        data["current_filtered_total"],
        color=theme.GREY_MID,
        ls=":",
        lw=2,
        label=f"Total, {cfg['analysis']['current_filter_ms']:g} ms filter",
    )
    currentax.axhline(0, color=theme.GREY_LIGHT, lw=0.7)
    currentax.set(
        title="Current covariance cancellation",
        xlabel="Lag τ (ms)",
        ylabel="Mean cross-covariance (nA²)",
    )
    currentax.legend(ncol=3, fontsize=theme.SIZE_ANNOTATION)
    # Leave headroom so the framed legend does not hide the covariance peaks.
    low, high = currentax.get_ylim()
    currentax.set_ylim(low, high + 0.25 * (high - low))
    ratio = summary["current_cancellation"]["total_to_same_type_zero_lag"]
    ratio_text = f"{ratio:.3f}" if ratio is not None else "unresolved"
    currentax.text(
        0.02,
        0.04,
        f"Zero-lag total / (EE + II): {ratio_text}",
        transform=currentax.transAxes,
        fontsize=theme.SIZE_ANNOTATION,
    )
    theme.label_panels(axes.ravel())
    fig.suptitle(
        "Shared-input conductance network · AI diagnostics",
        fontsize=theme.SIZE_TITLE,
        fontweight="bold",
    )
    fig.supxlabel(
        f"{cfg['burn_in_ms']:g} ms burn-in · {summary['window_ms'] / 1000:g} s measured · seed {cfg['seed']} · CV requires ≥{cfg['analysis']['min_cv_spikes']} spikes",
        fontsize=theme.SIZE_CAPTION,
    )
    return fig


if __name__ == "__main__":
    main()
