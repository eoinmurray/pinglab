"""Render saved true-class score trajectories without upstream execution.

The CLI requires explicit completed analysis evidence. Figures and browser
replays are written to a new presentation run; upstream execution is never
launched automatically.
"""

import sys
from pathlib import Path

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2]),
    str(Path(__file__).resolve().parents[2] / "tools"),
]

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as effects
import matplotlib.pyplot as plt
import numpy as np
from experiments import theme
from experiments.exp123 import recipe
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
from matplotlib.ticker import MaxNLocator
from pingstore.stages import source_run, stage_run


def save(fig, destination):
    with plt.rc_context({"savefig.bbox": None}):
        for ext in ("svg", "pdf"):
            fig.savefig(destination.with_suffix("." + ext), dpi=300)
    plt.close(fig)


def present(identity):
    source = source_run(
        recipe.REPO / ".pingstore", identity, stage="analyse", experiment=recipe.SLUG
    )
    if (
        source.record["execution"]["configuration"].get("schema")
        == "exp123.shared.measurement/v1"
    ):
        theme.set_paper_mode(True)
        theme.apply()
        plt.rcParams["svg.fonttype"] = "none"
        with stage_run(
            recipe.REPO,
            recipe.SLUG,
            "present",
            inputs={"analyse": source},
            configuration=dict(
                schema="exp123.shared.presentation/v1",
                paper_mode=True,
                figures="six figures from one uniform sweep; evidence traces coloured by mean I-volley interval frequency, higher darker, shared scale across nine panels and both coordinates; pooled sustained decisions in 5 ms / 1 volley by 5 Hz rectangular bins with shared log counts and frequency-bin medians; coverage-limited volleys; 80% population-accuracy first-crossing cost/latency frontier in house colours with matched mean I-frequency and nonlinear GABA-decay colour scales",
                show_uncertainty_estimates=False,
            ),
        ) as run:
            present_shared(source, run, save)
            result = run.run_id
        return result
    raise ValueError("expected shared-sweep analysis")


def plot_inhibitory_overview(
    raw, raster_compute, destination, save, raster_analysis=None
):
    sweep = "tau"
    cfg = raw["design"]
    conditions = [c for c in cfg["conditions"] if c["sweep"] == sweep]
    values = [c["axis_value"] for c in conditions]
    summaries = [r for r in raw["summaries"] if r["sweep"] == sweep]
    fig = plt.figure(figsize=(180 / 25.4, 105 / 25.4), layout="constrained")
    groups = fig.add_gridspec(1, 2)
    metric_grid = groups[0].subgridspec(2, 2)
    raster_grid = groups[1].subgridspec(3, 1)
    axes = np.array(
        [fig.add_subplot(metric_grid[i, j]) for i in range(2) for j in range(2)]
    )
    fig.get_layout_engine().set(w_pad=0.08, h_pad=0.06)
    specs = [
        ("psd_frequency_hz", "PSD frequency", "Spectral peak (Hz)"),
        ("participation_e_pct", "E participation", "Participation (%)"),
        ("e_spikes_per_second", "E spike rate", "E spikes/s"),
        ("rhythmicity_contrast", "Rhythmicity", "Lobe–trough contrast"),
    ]
    for ax, (key, title, ylabel) in zip(axes.flat, specs):
        ax.plot(
            values,
            [r[key] for r in summaries],
            "o-",
            color=theme.INK_BLACK,
            linewidth=1.3,
            markersize=3,
            label="Peak of mean PSD" if key == "psd_frequency_hz" else "Encoding mean",
        )
        ax.axvline(6.0 if sweep == "tau" else 1.0, color=theme.FAINT, ls="--", lw=0.7)
        ax.set(
            xscale="log",
            xlabel={
                "tau": "GABA decay (ms)",
                "leak": "Leak multiplier",
                "cap": "Capacitance multiplier",
            }[sweep],
            ylabel=ylabel,
            title=title,
        )
        ax.set_xticks(
            [3, 6, 10, 20, 30] if sweep == "tau" else [0.5, 1, 2],
            labels=["3", "6", "10", "20", "30"]
            if sweep == "tau"
            else ["0.5", "1", "2"],
        )
        ax.minorticks_off()
        if key == "participation_e_pct":
            ax.set_ylim(0, 30)
        if key == "rhythmicity_contrast":
            ax.set_ylim(0, 1.05)
        if key == "e_spikes_per_second":
            ax.ticklabel_format(axis="y", style="sci", scilimits=(3, 3))
    theme.label_panels(axes.flat)
    axes = [fig.add_subplot(raster_grid[i, 0]) for i in range(3)]
    if raster_analysis is not None:
        with np.load(
            raster_analysis.file("overview_rasters.npz"), allow_pickle=False
        ) as data:
            events = {key: data[key].copy() for key in data.files}
        dt_ms = float(events["dt_ms"])
    else:
        definition = raster_compute.record["execution"]["configuration"]
        shared = definition.get("schema") == "exp123.compute/v3"
        raster_cfg = (
            definition["sweep"] if shared else definition["protocols"]["trajectories"]
        )
        image_index = definition["illustrative_image_indices"][0] if shared else 0
        raster_unit = raster_compute.unit(
            recipe.recording_unit(
                image_index, recipe.RATE_HZ, "benchmark" if shared else "trajectories"
            )
        )
        dt_ms = raster_cfg["dt_ms"]
    for j, (ax, tau) in enumerate(zip(axes, recipe.TAUS_MS)):
        if raster_analysis is None:
            with np.load(
                raster_unit / f"tau{tau:g}--spikes.npz", allow_pickle=False
            ) as data:
                coordinates = {
                    key: np.nonzero(data[key][:, 0]) for key in ("spk_e", "spk_i")
                }
        else:
            coordinates = {
                key: (events[f"tau{j}--{key}--step"], events[f"tau{j}--{key}--neuron"])
                for key in ("spk_e", "spk_i")
            }
        for key, colour, offset in [
            ("spk_e", theme.INK_BLACK, 0),
            ("spk_i", theme.DEEP_RED, 1024),
        ]:
            k, neuron = coordinates[key]
            ax.scatter(
                k * dt_ms,
                neuron + offset,
                s=0.22,
                color=colour,
                linewidths=0,
                rasterized=True,
            )
        ax.axhline(1023.5, color=theme.FAINT, lw=0.5)
        ax.set(
            xlim=(0, 200),
            ylim=(-1, 1280),
            xlabel="Time (ms)",
            ylabel="Neuron",
            title=f"τ = {tau:g} ms",
        )
        ax.set_yticks([0, 1024, 1280])
        ax.set_xticks([0, 100, 200])
    theme.label_panels(axes, labels=["E", "F", "G"])
    save(fig, destination)


def mean_available(values, axis):
    valid = np.isfinite(values)
    count = valid.sum(axis=axis)
    return np.divide(
        np.nansum(values, axis=axis),
        count,
        out=np.full(count.shape, np.nan),
        where=count > 0,
    )


def tau_axis(ax):
    ax.set_yscale("log")
    ax.set_yticks([3, 6, 12, 30], labels=["3", "6", "12", "30"])
    ax.set_ylim(3, 30)
    ax.minorticks_off()


def evidence_frequencies(report, source):
    """Annotate each existing decay group with its mean I-interval frequency."""
    frequencies = {}
    for image in report["image_reports"]:
        index = image["image_index"]
        accepted = {
            (r["tau_gaba_ms"], r["seed"]): r["valid_cycle_coordinate"]
            for r in report["rows"]
            if r["image_index"] == index
        }
        values = []
        with np.load(
            source.unit(f"image{index:05d}") / "volleys.npz", allow_pickle=False
        ) as data:
            for ti, tau in enumerate(report["design"]["tau_gaba_ms"]):
                trials = []
                for seed in report["design"]["seeds"]:
                    peaks = data[f"tau{ti}-seed{seed}--volley_times_ms"]
                    if accepted[(tau, seed)] and len(peaks) > 1:
                        trials.append(1000 / np.diff(peaks).mean())
                if not trials:
                    raise ValueError(
                        f"No accepted I-volley frequency for image {index}, decay {tau:g}"
                    )
                values.append(float(np.mean(trials)))
        frequencies[index] = values
    return frequencies


def plot_evidence(report, source, destination, coordinate, save, frequencies=None):
    if frequencies is None:
        frequencies = evidence_frequencies(report, source)
    all_frequencies = np.asarray(list(frequencies.values()))
    low, high = float(all_frequencies.min()), float(all_frequencies.max())
    if np.isclose(low, high):
        low, high = low * 0.95, high * 1.05
    norm = Normalize(vmin=low, vmax=high)
    cmap = plt.colormaps["viridis_r"]
    fig, axes = plt.subplots(
        3, 3, figsize=(180 / 25.4, 140 / 25.4), layout="constrained"
    )
    for j, (ax, image) in enumerate(zip(axes.flat, report["image_reports"])):
        with np.load(
            source.unit(f"image{image['image_index']:05d}") / "trajectories.npz"
        ) as data:
            if coordinate == "time":
                x = data["time_ms"][::10]
                values = data["true_score"][:, :, ::10].mean(axis=1)
            else:
                x = np.concatenate(([0], data["completed_volley_count"]))
                values = mean_available(data["true_score_by_completed_volley"], axis=1)
                values[data["completed_volley_coverage"] < 0.8] = np.nan
                values = np.concatenate(
                    (data["true_score"][:, :, :1].mean(axis=1), values), axis=1
                )
            colours = np.asarray(frequencies[image["image_index"]])
            for ti in np.argsort(colours):
                ax.plot(
                    x,
                    values[ti],
                    color=cmap(norm(colours[ti])),
                    lw=1,
                    marker="o" if coordinate != "time" else None,
                    markersize=1.8,
                )
            if coordinate == "time":
                ax.set_xlim(0, 200)
                ax.set_xticks([0, 100, 200])
            else:
                finite = np.flatnonzero(np.isfinite(values).any(axis=0))
                ax.set_xlim(0, max(1, x[finite[-1]]) if len(finite) else 1)
        ax.set_ylim(0, 1.02)
        ax.set_yticks([0, 0.5, 1])
        if j % 3 != 0:
            ax.tick_params(labelleft=False)
        ax.set_title(f"{image['image_index']:02d} · digit {image['label']}")
        if j % 3 == 0:
            ax.set_ylabel("Mean true-class\nsoftmax score")
        if j >= 6:
            ax.set_xlabel(
                "Elapsed time (ms)"
                if coordinate == "time"
                else "Completed volley count"
            )
    theme.label_panels(axes.flat)
    colourbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=axes,
        label="Mean I-volley frequency (Hz)",
        shrink=0.8,
    )
    colourbar.locator = MaxNLocator(nbins=5)
    colourbar.update_ticks()
    save(fig, destination)


def plot_matched_accuracy(report, destination, save):
    fig, axes = plt.subplots(
        2, 2, figsize=(180 / 25.4, 140 / 25.4), layout="constrained"
    )
    specifications = [
        ("time", "Elapsed time (ms)"),
        ("cycles", "Cycle position"),
        ("input", "Cumulative input spikes"),
        ("network", "Network-spike budget"),
    ]
    for ax, (key, xlabel) in zip(axes.flat, specifications):
        curve = report["curves"][key]
        grid = np.asarray(curve["grid"])
        values = np.asarray(curve["accuracy"], dtype=float) * 100
        coverage = np.asarray(curve["trial_coverage"])
        values[:, coverage < 0.8] = np.nan
        image_plot = ax.pcolormesh(
            grid,
            report["design"]["tau_gaba_ms"],
            values,
            shading="nearest",
            cmap="viridis",
            vmin=0,
            vmax=100,
        )
        retained = np.flatnonzero(coverage >= 0.8)
        end = grid[retained[-1]] if len(retained) else 0
        ax.set_xlim(0, max(grid[1], end))
        ax.set_xlabel(xlabel)
        ax.set_ylabel("GABA decay (ms)")
        tau_axis(ax)
    theme.label_panels(axes.flat)
    fig.colorbar(image_plot, ax=axes, label="Accuracy (%)", shrink=0.8)
    save(fig, destination)


def plot_decisions(report, destination, save):
    fig, axes = plt.subplots(
        1, 2, figsize=(180 / 25.4, 100 / 25.4), sharey=True, layout="constrained"
    )
    ax, bx = axes
    time_edges = np.asarray(report["time_bin_edges_ms"])
    frequency_edges = np.asarray(report["frequency_bin_edges_hz"])
    volley_edges = np.asarray(report["volley_bin_edges"])
    time_histogram = np.asarray(report["time_counts"])
    volley_histogram = np.asarray(report["volley_counts"])
    centres = np.asarray(report["frequency_bin_centres_hz"])
    median_time = np.asarray(report["median_elapsed_time_ms"], dtype=float)
    median_volleys = np.asarray(report["median_completed_volleys"], dtype=float)
    max_volley = int(volley_edges[-1] - 0.5)
    fmax = frequency_edges[-1]
    norm = LogNorm(vmin=1, vmax=max(2, time_histogram.max(), volley_histogram.max()))
    # Keep low positive counts visible against empty white bins.
    brand = plt.colormaps["pinglab_brand"]
    cmap = LinearSegmentedColormap.from_list(
        "positive_brand", brand(np.linspace(0.12, 1, 256))
    )
    for axis, edges, histogram in [
        (ax, time_edges, time_histogram),
        (bx, volley_edges, volley_histogram),
    ]:
        axis.pcolormesh(
            edges,
            frequency_edges,
            np.ma.masked_equal(histogram.T, 0),
            cmap=cmap,
            norm=norm,
            shading="flat",
        )
    for axis, median in [(ax, median_time), (bx, median_volleys)]:
        axis.plot(
            median,
            centres,
            color=theme.INK_BLACK,
            lw=1.3,
            path_effects=[
                effects.Stroke(linewidth=3, foreground="white"),
                effects.Normal(),
            ],
        )
    ax.set(
        xlabel="Elapsed time to sustained\ncorrect decision (ms)",
        ylabel="I-volley frequency (Hz)",
        xlim=(0, 200),
        ylim=(0, fmax),
        title="Classification latency",
    )
    ax.set_xticks([0, 50, 100, 150, 200])
    bx.set(
        xlabel="Completed volleys at\nsustained correct decision",
        xlim=(0.5, max_volley + 0.5),
        title="Volley requirement",
    )
    bx.set_xticks([1] + list(range(5, max_volley + 1, 5)))
    theme.label_panels(axes)
    colourbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=axes,
        location="bottom",
        shrink=0.7,
        pad=0.04,
        aspect=45,
    )
    colourbar.set_label("Trials per bin (log scale)")
    save(fig, destination)


def plot_latency(report, destination, save):
    fig = plt.figure(figsize=(180 / 25.4, 120 / 25.4), layout="constrained")
    grid = fig.add_gridspec(1, 3, width_ratios=[1, 0.028, 0.028], wspace=0.12)
    ax = fig.add_subplot(grid[0, 0])
    frequency_axis = fig.add_subplot(grid[0, 1])
    tau_axis = fig.add_subplot(grid[0, 2])
    settings = report["frequency_by_setting"]
    frequencies = {s["tau_gaba_ms"]: s["mean_frequency_hz"] for s in settings}
    values = np.array([s["mean_frequency_hz"] for s in settings])
    low, high = float(values.min()), float(values.max())
    flat = np.isclose(low, high)
    if flat:
        low, high = low * 0.95, high * 1.05
    elif not np.all(np.diff(values) < 0):
        raise ValueError(
            "Dual colour scales require decreasing mean frequency across decay settings"
        )
    norm = Normalize(low, high)
    brand = plt.colormaps["pinglab_brand"]
    cmap = LinearSegmentedColormap.from_list(
        "visible_brand", brand(np.linspace(0.12, 1, 256))
    )
    points = report["frontier"]
    if not points:
        raise ValueError("No setting reaches the target accuracy")
    ax.plot(
        [p["mean_network_spikes"] / 1000 for p in points],
        [p["deadline_ms"] for p in points],
        color=theme.INK_BLACK,
        lw=1.2,
        zorder=2,
    )
    ax.scatter(
        [p["mean_network_spikes"] / 1000 for p in points],
        [p["deadline_ms"] for p in points],
        c=[p["mean_frequency_hz"] for p in points],
        cmap=cmap,
        norm=norm,
        s=56,
        marker="o",
        linewidths=0.8,
        edgecolors=theme.INK_BLACK,
        zorder=3,
    )
    ax.set(
        title=f"Reach {report['target_accuracy_pct']:g}% accuracy",
        xlabel="Mean network spikes needed (×10³)",
        ylabel="Time to target accuracy (ms)",
    )
    ax.margins(x=0.08, y=0.12)
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    frequency_bar = fig.colorbar(mappable, cax=frequency_axis)
    frequency_bar.set_label("Mean I-volley frequency (Hz)")
    if low <= 20 and high >= 100:
        frequency_bar.set_ticks([20, 40, 60, 80, 100])
    if flat:
        # A constant frequency cannot identify a decay value on a second scale.
        tau_axis.set_axis_off()
    else:
        tau_bar = fig.colorbar(mappable, cax=tau_axis)
        anchor_taus = [tau for tau in (30.0, 12.0, 6.0, 3.0) if tau in frequencies]
        tau_bar.set_ticks(
            [frequencies[tau] for tau in anchor_taus],
            labels=[f"{tau:g}" for tau in anchor_taus],
        )
        tau_bar.set_label(r"GABA decay, $\tau_{\mathrm{GABA}}$ (ms)")
    save(fig, destination)


def present_shared(source, run, save):

    def read(name):
        return json.loads(source.file(name).read_text())

    trajectories, accuracy, dynamics, tradeoffs = [
        read(name)
        for name in ("results.json", "accuracy.json", "metrics.json", "tradeoffs.json")
    ]
    plot_inhibitory_overview(
        dynamics, None, run.export / "inhibitory_overview", save, raster_analysis=source
    )
    frequencies = evidence_frequencies(trajectories, source)
    plot_evidence(
        trajectories, source, run.export / "confidence_time", "time", save, frequencies
    )
    plot_evidence(
        trajectories,
        source,
        run.export / "confidence_cycles",
        "cycles",
        save,
        frequencies,
    )
    (run.export / "frequency_colours.json").write_text(
        json.dumps(
            dict(
                definition="within each existing image/decay group: mean across accepted draws of 1000 / mean successive I-volley intervals in ms; Hz; no frequency binning",
                mean_frequency_hz_by_image=frequencies,
                tau_gaba_ms=trajectories["design"]["tau_gaba_ms"],
                colour_scale="shared linear frequency scale across both figures and all nine panels; viridis reversed, higher frequency darker",
            ),
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    pooled = read("pooled_decisions.json")
    plot_decisions(pooled, run.export / "sustained_decisions", save)
    (run.export / "pooled_decision_numbers.json").write_text(
        json.dumps(pooled, indent=2, allow_nan=False) + "\n"
    )
    plot_matched_accuracy(accuracy, run.export / "accuracy_comparison", save)
    target = read("target_accuracy.json")
    plot_latency(target, run.export / "deadline_tradeoffs", save)
    (run.export / "target_accuracy_numbers.json").write_text(
        json.dumps(target, indent=2, allow_nan=False) + "\n"
    )
    for name, report in [
        ("numbers", trajectories),
        ("accuracy_numbers", accuracy),
        ("overview_numbers", dynamics),
        ("tradeoff_numbers", tradeoffs),
    ]:
        (run.export / f"{name}.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    print(present(parser.parse_args().source))
