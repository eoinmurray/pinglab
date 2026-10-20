"""Draw the three-condition MNIST activity comparison from pinned analysis.

Invoke present.py --source <analyse-run-id> after comparison or long analysis.
The input image, exact spike drive, native parameter snapshots and measurements
come from that run and its authenticated compute input. No network is executed.

The output is an immutable Pingstore presentation containing PNG, SVG and PDF
figures plus numbers.json. Seed 42 is the displayed diagnostic; long-protocol
analysis retains the additional seeds, which are not pooled into this figure.
"""

import argparse
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

import numpy as np
from experiments.exp144.recipe import validate_configuration
from pingstore.contracts import load_json, write_json_atomic
from pingstore.stages import source_run, stage_run
from snnlab.sim import NetworkSnapshot, RecordingData
from snnlab.viz import plot_raster


def validate_comparison(source, cfg):
    """Require the same image input and input weights across all displayed cases."""
    seed = cfg["presentation"]["seed"]
    cases = cfg["presentation"]["cases"]
    first = None
    for case in cases:
        snapshot = NetworkSnapshot.read(
            source.file(case, f"seed-{seed}", "simulation", "start", "snapshot.json")
        )
        weights = snapshot.load_parameters(names=("input_to_E.weight",))
        if first is None:
            first = weights["input_to_E.weight"]
        else:
            np.testing.assert_array_equal(first, weights["input_to_E.weight"])
    # Each case uses this single saved input; compute never resamples per case.
    with np.load(source.file(f"input-seed-{seed}.npz")) as data:
        if data["image"].shape != (
            round(cfg["duration_ms"] / cfg["dt_ms"]),
            1,
            cfg["n_input"],
        ):
            raise ValueError("invalid shared image drive")


def present(args):
    """Draw full-duration activity with the actual image and W_in."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from experiments import theme
    from experiments.exp144.custom.figures import save_figure
    from matplotlib.colors import ListedColormap
    from matplotlib.ticker import FuncFormatter

    source = source_run(
        REPO / ".pingstore", args.source, experiment="exp144", stage="analyse"
    )
    ref = source.record["inputs"]["compute"]
    computed = source_run(
        REPO / ".pingstore",
        ref["run_id"],
        experiment="exp144",
        stage="compute",
        reference=ref,
    )
    results = load_json(source.file("results.json"))
    cfg = source.record["execution"]["configuration"]
    validate_configuration(cfg)
    if cfg != computed.record["execution"]["configuration"]:
        raise ValueError("analysis and compute configurations disagree")
    if cfg["protocol"] == "screen":
        raise ValueError("presentation requires a comparison or long protocol analysis")
    inputs = {"analysis": source, "compute": computed}
    plot_cfg = cfg["presentation"]
    selected = []
    for case in plot_cfg["cases"]:
        rows = [
            row
            for row in results["rows"]
            if row["case"]["id"] == case and row["seed"] == plot_cfg["seed"]
        ]
        if len(rows) != 1:
            raise ValueError(f"expected one presentation row for {case}")
        selected.append((source, computed, cfg, rows[0]))
    validate_comparison(computed, cfg)
    short = all(c["duration_ms"] <= 1000 for _, _, c, _ in selected)
    with stage_run(
        REPO,
        "exp144",
        "present",
        inputs=inputs,
        run_id=args.run_id,
        configuration=cfg,
    ) as run:
        shutil.copyfile(computed.file("circuit.svg"), run.export / "circuit.svg")
        theme.set_paper_mode(True)
        theme.apply()
        plt.rcParams["svg.fonttype"] = "none"
        fig, axes = plt.subplots(
            5,
            3,
            figsize=(180 / 25.4, 220 / 25.4),
            gridspec_kw={"height_ratios": [1, 1.5, 0.8, 0.8, 0.8]},
        )
        _, computed, cfg, row = selected[-1]
        input_ms = cfg["duration_ms"] if short else plot_cfg["input_window_ms"]
        with np.load(computed.file("sample.npz")) as data:
            axes[0, 0].imshow(data["pixels"], cmap="gray_r", vmin=0, vmax=1)
        axes[0, 0].set(
            title=f"MNIST train[{cfg['sample_index']}]: {cfg['sample_label']}",
            xticks=[],
            yticks=[],
        )
        input_recording = RecordingData.read(
            computed.unit(row["case"]["id"], f"seed-{plot_cfg['seed']}", "recording")
        ).select_batch(0)
        plot_raster(
            axes[0, 1],
            input_recording,
            signal="image.value",
            s=1,
            marker="|",
            linewidths=0.35,
            color=theme.INK_BLACK,
            rasterized=True,
        )
        del input_recording
        axes[0, 1].set(
            title="Pixel-rate input",
            xlim=(0, input_ms),
            ylim=(0, cfg["n_input"]),
            xlabel="Time (ms)",
            ylabel="Pixel",
        )
        snapshot = NetworkSnapshot.read(
            computed.file(
                row["case"]["id"],
                f"seed-{plot_cfg['seed']}",
                "simulation",
                "start",
                "snapshot.json",
            )
        )
        weights = snapshot.load_parameters(names=("input_to_E.weight",))
        axes[0, 2].imshow(
            weights["input_to_E.weight"] > 0,
            cmap=ListedColormap([theme.PAPER, theme.INK_BLACK]),
            aspect="auto",
            origin="lower",
            interpolation="nearest",
        )
        axes[0, 2].set(title="Initial W_in", xlabel="E neuron", ylabel="Pixel")
        for col, (analysed, computed, cfg, row) in enumerate(selected):
            case = row["case"]["id"]
            summary = row["populations"]
            density = row["case"]["recurrence_density"]
            axes[1, col].set_title(
                f"Recurrence {density:g}; I {row['case']['i_rate_hz']:g} Hz\nE/I: {summary['E']['rate_hz']:.2f}/{summary['I']['rate_hz']:.2f} Hz",
                fontsize=theme.SIZE_TITLE,
            )
            recording = RecordingData.read(
                computed.unit(case, f"seed-{plot_cfg['seed']}", "recording")
            ).select_batch(0)
            for population, offset, color in (
                ("E", 0, theme.INK_BLACK),
                ("I", cfg["n_e"], theme.DEEP_RED),
            ):
                plot_raster(
                    axes[1, col],
                    recording,
                    signal=f"{population}.spikes",
                    offset=offset,
                    s=1,
                    marker="|",
                    linewidths=0.3,
                    color=color,
                    rasterized=True,
                )
            del recording
            axes[1, col].xaxis.set_major_formatter(
                FuncFormatter(lambda value, _: f"{value / 1000:g}")
            )
            axes[1, col].set(
                xlim=(0, cfg["duration_ms"]),
                ylim=(0, cfg["n_e"] + cfg["n_i"]),
                xlabel="Time (s)",
                ylabel="Neuron" if col == 0 else "",
            )
            with np.load(
                analysed.file(case, f"seed-{plot_cfg['seed']}", "measurements.npz")
            ) as data:
                rate_ms = (
                    cfg["duration_ms"] - cfg["burn_in_ms"]
                    if short
                    else plot_cfg["rate_window_ms"]
                )
                for pop, color in [("E", theme.INK_BLACK), ("I", theme.DEEP_RED)]:
                    rate = data[f"{pop}_rate_hz"][: round(rate_ms)]
                    axes[2, col].plot(
                        np.arange(len(rate)),
                        rate,
                        color=color,
                        lw=0.6,
                        label=pop,
                    )
                    axes[3, col].semilogy(
                        data[f"{pop}_frequencies_hz"][1:],
                        data[f"{pop}_power"][1:],
                        color=color,
                        lw=0.8,
                    )
                    if short:
                        widths = cfg["analysis"]["correlation_bins_ms"]
                        axes[4, col].plot(
                            widths,
                            [summary[pop]["correlation"][str(w)] for w in widths],
                            color=color,
                            marker="o",
                            markersize=2.5,
                            lw=0.8,
                        )
                        continue
                    cv = data[f"{pop}_cv"]
                    cv = cv[np.isfinite(cv)]
                    axes[4, col].hist(
                        cv,
                        bins=np.linspace(0, 2, 41),
                        density=True,
                        histtype="step",
                        color=color,
                        lw=0.8,
                    )
            axes[2, col].set(
                xlim=(0, rate_ms),
                xlabel=f"Time after {cfg['burn_in_ms']:g} ms burn-in (ms)",
                ylabel="Rate (Hz)" if col == 0 else "",
            )
            axes[2, col].legend(loc="upper right")
            axes[3, col].set(
                xlim=(1, 150),
                xlabel="Frequency (Hz)",
                ylabel="PSD (Hz²/Hz)" if col == 0 else "",
            )
            if short:
                axes[4, col].axhline(0, color=theme.INK_BLACK, lw=0.4, alpha=0.4)
                axes[4, col].set(
                    xlim=(0, 55),
                    ylim=(-0.05, 1.05),
                    xticks=cfg["analysis"]["correlation_bins_ms"],
                    xlabel="Count bin (ms)",
                    ylabel="Mean pairwise r" if col == 0 else "",
                )
            else:
                axes[4, col].set(
                    xlim=(0, 2), xlabel="ISI CV", ylabel="Density" if col == 0 else ""
                )
        for row_index in [2, 3, 4]:
            lows, highs = zip(*(a.get_ylim() for a in axes[row_index]))
            for ax in axes[row_index]:
                ax.set_ylim(min(lows), max(highs))
        theme.label_panels(axes.flat)
        fig.tight_layout(h_pad=1.0, w_pad=0.9, rect=(0, 0.035, 1, 1))
        fig.text(
            0.08,
            0.01,
            f"{cfg['max_pixel_rate_hz']:g} Hz max pixel rate; W_in parent {row['case']['input_parent_us']:g} µS. E black / I red; rates per neuron.\n"
            + (
                f"{cfg['duration_ms']:g} ms from rest, no burn-in. Correlations: 200 cells/population; constant cells omitted."
                if short
                else f"All weights are fixed. Statistics: {(cfg['duration_ms'] - cfg['burn_in_ms']) / 1000:g} s after {cfg['burn_in_ms'] / 1000:g} s burn-in."
            ),
            fontsize=theme.SIZE_CAPTION,
        )
        save_figure(fig, run.export / "mnist_ai", formats=("png", "svg", "pdf"))
        plt.close(fig)
        write_json_atomic(
            run.export / "numbers.json",
            dict(
                schema="exp144.presentation/v1",
                rows=[r for _, _, _, r in selected],
            ),
        )
    return run.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-id")
    print(present(parser.parse_args()))


if __name__ == "__main__":
    main()
