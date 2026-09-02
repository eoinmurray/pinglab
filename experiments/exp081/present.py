"""Render completed exp081 analysis; never simulate, analyse or materialize."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "experiments"), str(REPO / "tools")]

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from experiments.exp081 import inputs, recipe
from pingstore.contracts import PingstoreError, load_json, write_json_atomic
from pingstore.stages import stage_run

from helpers import theme


def plot_moments(
    empirical_mean: np.ndarray,
    empirical_sd: np.ndarray,
    cfg: dict,
    output: Path,
) -> None:
    theme.apply()
    colors = (theme.INK_BLACK, theme.DEEP_RED, theme.ELECTRIC_CYAN)
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.25), constrained_layout=True)
    for index, (probe, color) in enumerate(zip(cfg["probes_uS"], colors, strict=True)):
        axes[0].plot(
            cfg["input_rate_grid_hz"],
            empirical_mean[index],
            color=color,
            label=f"{probe:g} μS",
        )
        axes[1].plot(cfg["input_rate_grid_hz"], empirical_sd[index], color=color)
    axes[0].set(title="Mean feature", ylabel="Mean feature z (mV)")
    axes[1].set(title="Feature SD", ylabel="Feature SD (mV)")
    for axis in axes:
        axis.set(xlabel="Input rate (Hz)", xlim=(0, 25))
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.14)
    axes[0].legend(frameon=False)
    theme.label_panels(axes)
    fig.savefig(output / "empirical_moments.svg", metadata={"Date": None})
    plt.close(fig)


def plot_distributions(histograms: dict, cfg: dict, output: Path) -> None:
    theme.apply()
    fig, axes = plt.subplots(
        1, 3, figsize=(6.5, 2.75), constrained_layout=True, sharex=True, sharey=True
    )
    bins = histograms["edges_mV"]
    for axis, rate, probability in zip(
        axes, cfg["distribution_rates_hz"], histograms["probability"], strict=True
    ):
        axis.bar(
            bins[:-1],
            probability,
            width=np.diff(bins),
            align="edge",
            color=theme.INK_BLACK,
            alpha=0.72,
        )
        axis.set_title(f"{rate:g} Hz")
        axis.set_xlabel("Feature z (mV)")
        axis.set_yscale("log")
        axis.set_ylim(0.5 / cfg["distribution_draws"], 1.0)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Probability per bin (log scale)")
    theme.label_panels(axes)
    fig.savefig(output / "response_distributions.svg", metadata={"Date": None})
    plt.close(fig)


def plot_frequency_response(responses: dict, cfg: dict, output: Path) -> None:
    theme.apply()
    colors = (theme.INK_BLACK, theme.DEEP_RED, theme.ELECTRIC_CYAN)
    fig, axes = plt.subplots(
        1, 2, figsize=(6.5, 3.25), constrained_layout=True, sharey=True
    )
    for index, (rate, color) in enumerate(
        zip(cfg["frequency_response_rates_hz"], colors, strict=True)
    ):
        axes[0].semilogx(
            responses["frequency_hz"],
            responses["unaveraged_db"][index],
            color=color,
            label=f"{rate:g} Hz drive",
        )
        axes[1].semilogx(
            responses["frequency_hz"], responses["averaged_db"][index], color=color
        )
    axes[0].set(
        title="Synapse + membrane", ylabel="Magnitude relative to low-drive DC (dB)"
    )
    axes[1].set(title=f"After {cfg['presentation_ms']:g} ms averaging")
    for axis in axes:
        axis.set(xlabel="Frequency (Hz)", ylim=(-90, 4))
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=7)
    theme.label_panels(axes)
    fig.savefig(output / "frequency_response.svg", metadata={"Date": None})
    plt.close(fig)


def plot_comparison(
    empirical_mean: np.ndarray,
    empirical_sd: np.ndarray,
    analytical_mean: np.ndarray,
    analytical_sd: np.ndarray,
    cfg: dict,
    output: Path,
) -> None:
    theme.apply()
    colors = (theme.INK_BLACK, theme.DEEP_RED, theme.ELECTRIC_CYAN)
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.25), constrained_layout=True)
    for index, (probe, color) in enumerate(zip(cfg["probes_uS"], colors, strict=True)):
        axes[0].plot(
            cfg["input_rate_grid_hz"],
            analytical_mean[index],
            color=color,
            label=f"{probe:g} μS",
        )
        axes[0].scatter(
            cfg["input_rate_grid_hz"],
            empirical_mean[index],
            s=7,
            color=color,
            alpha=0.28,
            edgecolors="none",
        )
        axes[1].plot(cfg["input_rate_grid_hz"], analytical_sd[index], color=color)
        axes[1].scatter(
            cfg["input_rate_grid_hz"],
            empirical_sd[index],
            s=7,
            color=color,
            alpha=0.28,
            edgecolors="none",
        )
    axes[0].set(title="Mean feature", ylabel="Mean feature z (mV)")
    axes[1].set(title="Feature SD", ylabel="Feature SD (mV)")
    for axis in axes:
        axis.set(xlabel="Input rate (Hz)", xlim=(0, 25))
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.14)
    axes[0].legend(frameon=False)
    theme.label_panels(axes)
    fig.savefig(output / "analytical_empirical.svg", metadata={"Date": None})
    plt.close(fig)


def load_arrays(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as retained:
        return {key: retained[key] for key in retained.files}


def present(identity: str, *, run_id: str | None = None) -> str:
    analysis = inputs.source(REPO, identity, "analyse")
    cfg = inputs.configuration(analysis)
    refs = analysis.record["inputs"]
    if set(refs) != {"compute"}:
        raise PingstoreError("exp081 analysis must pin exactly one compute input")
    compute = inputs.source(
        REPO, refs["compute"]["run_id"], "compute", reference=refs["compute"]
    )
    if inputs.configuration(compute) != cfg or compute.record["inputs"]:
        raise PingstoreError("analysis recipe or compute lineage disagrees with source")
    results = load_json(analysis.export / "results.json")
    if (
        results.get("schema") != "exp081.analysis/v1"
        or results.get("parameters") != cfg
    ):
        raise PingstoreError("unsupported or inconsistent exp081 analysis payload")
    with stage_run(
        REPO,
        recipe.SLUG,
        "present",
        inputs={"analysis": analysis, "compute": compute},
        run_id=run_id,
        configuration=cfg,
    ) as run:
        moments = load_arrays(analysis.export / "moments.npz")
        histograms = load_arrays(analysis.export / "histograms.npz")
        responses = load_arrays(analysis.export / "frequency_response.npz")
        mean, sd = moments["empirical_mean_mV"], moments["empirical_sd_mV"]
        plot_moments(mean, sd, cfg, run.export)
        plot_distributions(histograms, cfg, run.export)
        plot_frequency_response(responses, cfg, run.export)
        plot_comparison(
            mean,
            sd,
            moments["analytical_mean_mV"],
            moments["analytical_sd_mV"],
            cfg,
            run.export,
        )
        write_json_atomic(run.export / "numbers.json", results)
    return run.run_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", required=True, help="completed exp081 v4 analyse run ID"
    )
    parser.add_argument("--run-id", help="unused v4 identity reserved before dispatch")
    args = parser.parse_args()
    present(args.source, run_id=args.run_id)


if __name__ == "__main__":
    main()
