"""Physical-time comparison figures for exp142.

Render the experiment's supplied measurements and input events. The presentation
stage owns source validation and provenance; plotting never executes a network."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from experiments import theme
from matplotlib.lines import Line2D

NS_PER_MS = 1_000_000
WIDTH_IN = 180 / 25.4
EXTENSIONS = ("png", "svg", "pdf")


def _style():
    theme.set_paper_mode(True)
    theme.apply()
    plt.rcParams["svg.fonttype"] = "none"


def _colors():
    return [
        theme.INK_BLACK,
        theme.GREY_LIGHT,
        theme.ELECTRIC_CYAN,
        theme.AMBER,
        theme.DEEP_RED,
    ]


def _save(figure, destination, name):
    paths = []
    try:
        # Preserve the authored publication geometry, not a tight-cropped size.
        with plt.rc_context({"savefig.bbox": None}):
            for extension in EXTENSIONS:
                path = Path(destination) / f"{name}.{extension}"
                figure.savefig(path, dpi=300)
                paths.append(path)
    finally:
        plt.close(figure)
    return paths


def _labels(cases):
    return [
        f"{case['dt_ns'] / NS_PER_MS:g}" if case["dt_ns"] is not None else "CT"
        for case in cases
    ]


def _input_figure(summary, image, events, *, label, image_index):
    image = np.asarray(image)
    image_spec = summary.get("input_image", {})
    image_shape = tuple(image_spec.get("shape", (28, 28)))
    image_name = image_spec.get("name", "MNIST test image")
    if (
        image.shape != image_shape
        or not np.isfinite(image).all()
        or np.any((image < 0) | (image > 1))
    ):
        raise ValueError(
            f"image must be a finite normalized {image_shape[0]}x{image_shape[1]} image"
        )
    times = np.asarray(events["time_ns"])
    channels = np.asarray(events["channels"])
    batches = np.asarray(events["batches"])
    duration = summary["configuration"]["duration_ns"]
    if (
        times.ndim != 1
        or times.shape != channels.shape
        or batches.shape != times.shape
        or not np.issubdtype(times.dtype, np.integer)
        or not np.issubdtype(channels.dtype, np.integer)
        or not np.issubdtype(batches.dtype, np.integer)
        or np.any(times < 0)
        or np.any(times >= duration)
        or np.any(channels < 0)
        or np.any(channels >= image.size)
        or np.any(batches != 0)
    ):
        raise ValueError("invalid canonical single-image event coordinates")
    figure, axes = plt.subplots(
        1, 2, figsize=(WIDTH_IN, 70 / 25.4), width_ratios=(1, 2.6), layout="constrained"
    )
    axes[0].imshow(image, cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
    axes[0].set(
        title=f"{image_name} {image_index}\nTrue label: {label}", xticks=[], yticks=[]
    )
    axes[1].scatter(
        times / NS_PER_MS,
        channels,
        s=2,
        marker="|",
        color=theme.INK_BLACK,
        linewidths=0.35,
        rasterized=True,
    )
    axes[1].set(
        title="One encoded input, shared by both models",
        xlabel="Time (ms)",
        ylabel="Pixel channel",
        xlim=(0, duration / NS_PER_MS),
        ylim=(-1, image.size),
    )
    audit = summary.get("input")
    # Do not infer all-event equivalence from a canonical raster. An audit is
    # displayed only when the caller supplies the actual measured evidence.
    if isinstance(audit, dict):
        text = audit.get("plot_annotation")
        if isinstance(text, str) and text:
            figure.supxlabel(text, fontsize=theme.SIZE_CAPTION)
    theme.label_panels(axes)
    return figure


def _scatter_population(ax, x, values, n_e, color, *, hollow=False, offset=0.0):
    for start, stop, marker in ((0, n_e, "o"), (n_e, len(values), "^")):
        group = np.asarray(values)[start:stop]
        finite = np.isfinite(group)
        # Fixed spread is purely visual and uses neuron identity, not a fit.
        positions = x + offset + np.linspace(-0.07, 0.07, len(group))
        ax.scatter(
            positions[finite],
            group[finite],
            s=7,
            marker=marker,
            facecolors="none" if hollow else color,
            edgecolors=color,
            linewidths=0.45,
            alpha=0.6,
            rasterized=True,
        )
        if finite.any():
            center = x + offset + (-0.025 if marker == "o" else 0.025)
            ax.plot(
                [center - 0.035, center + 0.035],
                [np.median(group[finite])] * 2,
                color=theme.INK_BLACK,
                lw=0.7,
            )


def _comparison_figure(summary, arrays, model):
    cases = summary["models"][model]["cases"]
    colors = _colors()
    n_e = summary["configuration"]["n_e"]
    n_i = summary["models"][model].get("n_i", summary["configuration"]["n_i"])
    neuron = summary["target_e"]
    duration = summary["configuration"]["duration_ns"] / NS_PER_MS
    figure, axes = plt.subplots(
        2, 3, figsize=(WIDTH_IN, 132 / 25.4), layout="constrained"
    )
    for index, (case, color) in enumerate(zip(cases, colors)):
        stem = f"{model}__{case['id']}__"
        g_clock, v_clock = (
            arrays[stem + key] / NS_PER_MS for key in ("g_time_ns", "v_time_ns")
        )
        for key, style in (("g_exc_ns", "-"), ("g_inh_ns", "--")):
            axes[0, 0].plot(
                g_clock,
                arrays[stem + key][:, neuron],
                color=color,
                ls=style,
                lw=0.9 if case["dt_ns"] is None else 0.6,
            )
        axes[0, 1].plot(
            v_clock, arrays[stem + "voltage_mv"][:, neuron], color=color, lw=0.7
        )
        selected = arrays[stem + "spike_neurons"] == neuron
        times = arrays[stem + "spike_time_ns"][selected] / NS_PER_MS
        axes[0, 2].scatter(
            times,
            np.full(len(times), index),
            marker="|",
            s=24,
            color=color,
            linewidths=0.75,
        )
        for key, hollow, offset in (
            ("g_exc_guarded_mae_ns", False, -0.13),
            ("g_inh_guarded_mae_ns", True, 0.13),
        ):
            _scatter_population(
                axes[1, 0],
                index,
                arrays[stem + key],
                n_e,
                color,
                hollow=hollow,
                offset=offset,
            )
        _scatter_population(
            axes[1, 1], index, arrays[stem + "v_reset_aware_mae_mv"], n_e, color
        )
        _scatter_population(
            axes[1, 2], index, arrays[stem + "spike_median_shift_ms"], n_e, color
        )
        e, i = (
            case["populations"][population]["timing_eligible_neurons"]
            for population in ("E", "I")
        )
        axes[1, 2].text(
            index,
            0.98,
            f"{e}\n{i}",
            transform=axes[1, 2].get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=theme.SIZE_ANNOTATION,
        )
    axes[0, 0].set(title=f"E{neuron}: conductance", ylabel="Conductance (nS)")
    axes[0, 1].set(title=f"E{neuron}: post-reset voltage", ylabel="Voltage (mV)")
    axes[0, 2].set(
        title=f"E{neuron}: spikes",
        ylabel="Timestep (ms)",
        yticks=range(len(cases)),
        yticklabels=_labels(cases),
        ylim=(len(cases) - 0.5, -0.5),
    )
    for ax in axes[0]:
        ax.set(xlabel="Time (ms)", xlim=(0, duration))
    for ax, title, ylabel in zip(
        axes[1],
        ("Receptor error", "Reset-aware voltage error", "Ordinal spike shift"),
        ("Mean |Δg| (nS)", "Mean |ΔV| (mV)", "Signed shift (ms)"),
    ):
        ax.set(
            title=title,
            ylabel=ylabel,
            xlabel="Timestep (ms)",
            xlim=(-0.5, len(cases) - 0.5),
        )
        ax.set_xticks(range(len(cases)), _labels(cases), rotation=45, ha="right")
        ax.axhline(0, color=theme.FAINT, lw=0.6, zorder=0)
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    handles = [
        Line2D(
            [],
            [],
            marker=marker,
            color=theme.INK_BLACK,
            ls="none",
            markersize=3,
            markerfacecolor="none" if hollow else theme.INK_BLACK,
            label=label,
        )
        for marker, hollow, label in (
            ("o", False, "E AMPA"),
            ("^", False, "I AMPA"),
            ("o", True, "E GABA"),
            ("^", True, "I GABA"),
        )
    ]
    axes[1, 0].legend(
        handles=handles,
        ncol=2,
        loc="upper right",
        fontsize=theme.SIZE_ANNOTATION,
        columnspacing=0.6,
        handletextpad=0.2,
    )
    for column, keys, floor in (
        (0, ("g_exc_guarded_mae_ns", "g_inh_guarded_mae_ns"), 0.001),
        (1, ("v_reset_aware_mae_mv",), 1.0),
    ):
        values = np.concatenate(
            [arrays[f"{model}__{case['id']}__{key}"] for case in cases for key in keys]
        )
        maximum = float(np.max(values[np.isfinite(values)], initial=0))
        axes[1, column].set_ylim(-0.04 * max(floor, maximum), max(floor, 1.5 * maximum))
    shifts = np.concatenate(
        [arrays[f"{model}__{case['id']}__spike_median_shift_ms"] for case in cases]
    )
    finite = shifts[np.isfinite(shifts)]
    extent = max(0.1, float(np.max(np.abs(finite), initial=0)))
    axes[1, 2].set_ylim(-1.15 * extent, 1.6 * extent)
    axes[1, 2].text(
        0.02,
        0.03,
        "− earlier / + later",
        transform=axes[1, 2].transAxes,
        fontsize=theme.SIZE_ANNOTATION,
    )
    coverage = []
    for signal in ("g", "v"):
        values = np.concatenate(
            [
                arrays[f"{model}__{case['id']}__{signal}_retained_sample_fraction"]
                for case in cases
            ]
        )
        coverage.append(
            f"{signal}: {100 * values.min():.0f}–{100 * values.max():.0f}% kept"
        )
    figure.supxlabel(
        f"E circles (n={n_e}) / I triangles (n={n_i}); black bars: population medians.\n"
        "A: AMPA solid / GABA dashed. "
        f"D/E: outside ±{summary['reset_guard_ms']:g} ms of either train's spikes; {', '.join(coverage)}.\n"
        "F: equal nonzero counts only; eligible counts at top: E above I.",
        fontsize=theme.SIZE_CAPTION,
    )
    figure.legend(
        [Line2D([], [], color=color, lw=1) for color in colors],
        [case["label"] for case in cases],
        ncol=5,
        loc="outside upper center",
        title=f"{model.upper()} · continuous reference (red)",
    )
    theme.label_panels(axes.flat)
    return figure


def _score_classification_figure(
    summary,
    arrays,
    *,
    label,
    decision_key,
    array_prefix,
    head_title,
    score_label,
    caption,
    drawstyle="default",
    clock_label="Neural timestep (ms)",
):
    """Plot supplied trajectories, final class scores and ranks in one layout."""
    n_output = summary["configuration"]["n_output"]
    if not 0 <= label < n_output:
        raise ValueError("true label is outside the recorded output classes")
    figure, axes = plt.subplots(
        3, 2, figsize=(WIDTH_IN, 170 / 25.4), layout="constrained"
    )
    for column, model in enumerate(("coba", "ping")):
        record = summary["models"][model]
        cases = record["cases"]
        anchor = record.get("native_anchor")
        display_cases = ([anchor] if anchor is not None else []) + cases
        display_labels = (
            [summary.get("native_anchor_label", "Train")] if anchor is not None else []
        ) + _labels(cases)
        styles = ([(theme.INK_BLACK, "-.")] if anchor is not None else []) + [
            (color, "-") for color in _colors()
        ]
        winner = cases[-1][decision_key]["prediction"]
        for case, (color, style) in zip(display_cases, styles):
            stem = f"{model}__{case['id']}__"
            time = arrays[stem + array_prefix + "v_time_ns"] / NS_PER_MS
            scores = arrays[stem + array_prefix + "readout_scores"]
            axes[0, column].plot(
                time,
                scores[:, label],
                color=color,
                ls=style,
                lw=0.9,
                drawstyle=drawstyle,
            )
            if winner != label:
                axes[0, column].plot(
                    time,
                    scores[:, winner],
                    color=color,
                    ls="--",
                    lw=0.65,
                    drawstyle=drawstyle,
                )
            axes[1, column].plot(
                range(n_output),
                arrays[stem + array_prefix + "class_scores"],
                color=color,
                ls=style,
                marker="o",
                ms=2.5,
                lw=0.7,
            )
        axes[0, column].set(
            title=f"{model.upper()} · {head_title}\nTrue {label}; CT prediction {winner}",
            xlabel="Time (ms)",
            ylabel=score_label,
        )
        axes[1, column].set(
            title="All final class scores",
            xlabel="Digit class",
            ylabel=score_label,
            xticks=range(n_output),
        )
        for ax in axes[:2, column]:
            ax.set_yscale("linear")
            ax.ticklabel_format(axis="y", style="plain", useOffset=False)
            ax.axhline(0, color=theme.FAINT, lw=0.5, zorder=0)
        values = np.stack(
            [
                arrays[f"{model}__{case['id']}__{array_prefix}class_scores"]
                for case in display_cases
            ]
        )
        ranks = 1 + (values[:, None, :] > values[:, :, None]).sum(axis=2)
        ax = axes[2, column]
        heat = ax.imshow(
            ranks,
            aspect="auto",
            interpolation="nearest",
            cmap=plt.get_cmap("pinglab_brand").reversed(),
            vmin=1,
            vmax=n_output,
        )
        ax.set(
            title="Final score ranks (1 = best)",
            xlabel="Digit class",
            ylabel=clock_label,
            xticks=range(n_output),
            yticks=range(len(display_cases)),
            yticklabels=display_labels,
        )
        lines = []
        for index, (case, dt) in enumerate(zip(display_cases, display_labels)):
            decision = case[decision_key]
            predicted = decision["prediction"]
            ax.plot(
                predicted,
                index,
                marker="s",
                ms=8,
                mfc="none",
                mec=theme.ELECTRIC_CYAN,
                mew=1,
            )
            margin = decision["margin"]
            margin_text = f"{margin:.3g}" if margin is not None else "n/a"
            lines.append(
                f"{dt}: {predicted}{'*' if decision['tie'] else ''}; Δ={margin_text}"
            )
            for digit, rank in enumerate(ranks[index]):
                ax.text(
                    digit,
                    index,
                    str(rank),
                    ha="center",
                    va="center",
                    fontsize=theme.SIZE_ANNOTATION,
                    color=theme.INK_BLACK if rank > (n_output + 1) / 2 else theme.PAPER,
                )
        ax.text(
            0,
            -0.36,
            "case: prediction; winner margin Δ\n" + "\n".join(lines),
            transform=ax.transAxes,
            va="top",
            fontsize=theme.SIZE_ANNOTATION,
        )
        colorbar = figure.colorbar(heat, ax=ax, fraction=0.035, pad=0.02)
        colorbar.set_ticks([1, n_output])
    handles = [Line2D([], [], color=color) for color in _colors()]
    labels = [case["label"] for case in summary["cases"]]
    if any(
        summary["models"][model].get("native_anchor") is not None
        for model in ("coba", "ping")
    ):
        handles.append(Line2D([], [], color=theme.INK_BLACK, ls="-."))
        labels.append(summary.get("native_anchor_legend", "Trained anchor"))
    figure.legend(handles, labels, ncol=3, loc="outside upper center")
    figure.supxlabel(caption, fontsize=theme.SIZE_CAPTION)
    for artist in theme.label_panels(axes.flat):
        artist.set_x(-0.02)
    return figure


def _fixed_classification_figure(summary, arrays, *, label):
    head_dt = summary["fixed_readout"]["dt_ns"] / NS_PER_MS
    return _score_classification_figure(
        summary,
        arrays,
        label=label,
        decision_key="fixed_readout_classification",
        array_prefix="fixed_",
        head_title=f"fixed {head_dt:g} ms head",
        score_label="Mean pre-reset score",
        caption=(
            f"Fixed {head_dt:g} ms SNNLab float32 output LIF: τ=2 ms, threshold=1; one soft reset per bin.\n"
            "Each model keeps its original trained weights; right-closed bins (kh,(k+1)h], t=0 in first bin.\n"
            "A/B: true-class mean pre-reset scores; C/D: all class scores; E/F: ranks; cyan: prediction.\n"
            "* Exact tie. Scores/margins are not calibrated probabilities."
        ),
    )


def _spiking_classification_figure(summary, arrays, *, label):
    return _score_classification_figure(
        summary,
        arrays,
        label=label,
        decision_key="classification",
        array_prefix="",
        head_title="COBA-LIF output",
        score_label="Output spike count",
        drawstyle="steps-post",
        clock_label="Network / output timestep (ms)",
        caption=(
            "Discrete: COBA-LIF output uses the same timestep as its E/I network. CT: continuous COBA-LIF output.\n"
            "Each model keeps its original trained weights; total output spikes are the trained class scores.\n"
            "A/B: true-class cumulative spikes; C/D: all final class counts; E/F: ranks; cyan: prediction.\n"
            "* Exact tie. Winner margin Δ is a count difference, not a calibrated probability."
        ),
    )


def _classification_figure(summary, arrays, *, label):
    if summary.get("readout_policy") == "matched-spiking":
        return _spiking_classification_figure(summary, arrays, label=label)
    if isinstance(summary.get("fixed_readout"), dict):
        return _fixed_classification_figure(summary, arrays, label=label)
    figure, axes = plt.subplots(
        2, 2, figsize=(WIDTH_IN, 160 / 25.4), layout="constrained"
    )
    colors = _colors()
    for column, model in enumerate(("coba", "ping")):
        cases = summary["models"][model]["cases"]
        n_output = summary["configuration"]["n_output"]
        if not 0 <= label < n_output:
            raise ValueError("true label is outside the recorded output classes")
        winner = cases[-1]["classification"]["prediction"]
        for case, color in zip(cases, colors):
            stem = f"{model}__{case['id']}__"
            time = arrays[stem + "v_time_ns"] / NS_PER_MS
            axes[0, column].plot(
                time, arrays[stem + "readout_scores"][:, label], color=color, lw=0.9
            )
            if winner != label:
                axes[0, column].plot(
                    time,
                    arrays[stem + "readout_scores"][:, winner],
                    color=color,
                    ls="--",
                    lw=0.7,
                )
            axes[0, column].plot(
                time,
                arrays[stem + "readout_common_scores"][:, label],
                color=color,
                ls=":",
                lw=0.7,
            )
        anchor = summary["models"][model].get("native_anchor")
        display_cases = ([anchor] if anchor is not None else []) + cases
        display_labels = (
            [summary.get("native_anchor_label", "Train")] if anchor is not None else []
        ) + _labels(cases)
        if anchor is not None:
            axes[0, column].plot(
                arrays[f"{model}__native__v_time_ns"] / NS_PER_MS,
                arrays[f"{model}__native__readout_scores"][:, label],
                color=theme.INK_BLACK,
                ls="-.",
                lw=0.8,
            )
        axes[0, column].set(
            title=f"{model.upper()} · true {label}, CT mean winner {winner}",
            xlabel="Time (ms)",
            ylabel="Mean score (symlog)",
        )
        axes[0, column].set_yscale("symlog", linthresh=0.1)
        values = np.stack(
            [arrays[f"{model}__{case['id']}__class_scores"] for case in display_cases]
        )
        ranks = 1 + (values[:, None, :] > values[:, :, None]).sum(axis=2)
        heat = axes[1, column].imshow(
            ranks,
            aspect="auto",
            interpolation="nearest",
            cmap=plt.get_cmap("pinglab_brand").reversed(),
            vmin=1,
            vmax=n_output,
        )
        axes[1, column].set(
            title="Final mean-score rank (1 = best)",
            xlabel="Digit class",
            ylabel="Train / matched timestep (ms)",
            xticks=range(n_output),
            yticks=range(len(display_cases)),
            yticklabels=display_labels,
        )
        for index, case in enumerate(display_cases):
            decision = case["classification"]
            predicted = decision["prediction"]
            axes[1, column].plot(
                predicted,
                index,
                marker="s",
                ms=8,
                mfc="none",
                mec=theme.ELECTRIC_CYAN,
                mew=1,
            )
        # The panel title identifies scores. No vertical label outside the
        # right edge: it clipped at the final 180-mm publication width.
        colorbar = figure.colorbar(heat, ax=axes[1, column], fraction=0.035, pad=0.02)
        colorbar.set_ticks([1, n_output])
        lines = []
        for case, dt in zip(display_cases, display_labels):
            decision = case["classification"]
            common = case["common_clock_classification"]
            margin = decision["margin"]
            margin_text = f"{margin:.3g}" if margin is not None else "n/a"
            count = case.get("spike_count_classification")
            count_text = (
                f"{count['prediction']}{'*' if count['tie'] else ''}"
                if count
                else "n/a"
            )
            lines.append(
                f"{dt}: {decision['prediction']}{'*' if decision['tie'] else ''} / "
                f"{common['prediction']}{'*' if common['tie'] else ''} / {count_text}; Δ={margin_text}"
            )
        axes[1, column].text(
            0,
            -0.28,
            "case: mean / common / count prediction; margin\n" + "\n".join(lines),
            transform=axes[1, column].transAxes,
            va="top",
            fontsize=theme.SIZE_ANNOTATION,
        )
    handles = [Line2D([], [], color=color) for color in colors]
    labels = [case["label"] for case in summary["cases"]]
    if any("native_anchor" in summary["models"][model] for model in ("coba", "ping")):
        handles.append(Line2D([], [], color=theme.INK_BLACK, ls="-."))
        labels.append(summary.get("native_anchor_legend", "Trained anchor"))
    figure.legend(handles, labels, ncol=3, loc="outside upper center")
    figure.supxlabel(
        "Solid: true class; dashed: CT winner if different; dotted: true class, common-clock diagnostic.\n"
        "Discrete: native mean pre-reset score. CT: physical-time mean post-reset score (small-step limit).\n"
        "Common: arithmetic mean post-reset state on shared sample clock. Cyan square: final winner; * exact tie.\n"
        "Count: post-hoc alternative decoder, not trained; scores/margins are not calibrated probabilities.",
        fontsize=theme.SIZE_CAPTION,
    )
    theme.label_panels(axes.flat)
    return figure


def plot_all(summary, arrays, image, events, destination, *, label, image_index=0):
    """Write twelve flat publication exports; no CLI, runs, or neural execution.

    The input audit stays caller-owned. If ``summary['input']['plot_annotation']``
    is supplied, its measured audit text is shown without inventing counts.
    Return the exported paths for callers/tests; no mutation of inputs.
    """
    destination = Path(destination)
    if not destination.is_dir():
        raise ValueError("destination directory must already exist")
    if (
        not isinstance(label, (int, np.integer))
        or not 0 <= label < summary["configuration"]["n_output"]
    ):
        raise ValueError("true label is outside the recorded output classes")
    _style()
    paths = _save(
        _input_figure(summary, image, events, label=label, image_index=image_index),
        destination,
        "input_equivalence",
    )
    for model in ("coba", "ping"):
        paths.extend(
            _save(
                _comparison_figure(summary, arrays, model),
                destination,
                f"{model}_comparison",
            )
        )
    paths.extend(
        _save(
            _classification_figure(summary, arrays, label=label),
            destination,
            "classification",
        )
    )
    return paths
