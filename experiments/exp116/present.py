"""Render the four-part exp116 results story without recomputation."""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from experiments.exp116 import recipe
from experiments.exp116.compute import record_environment
from experiments.helpers import theme
from experiments.helpers.figsave import save_figure
from pingstore.contracts import PingstoreError, load_json, write_json_atomic
from pingstore.stages import source_run, stage_run


def finish(axis):
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=theme.SIZE_TICK)


def onset_figure(numbers, coordinates, destination):
    onset = numbers["reference"]["onset"]
    fig, axis = plt.subplots(figsize=(90 / 25.4, 68 / 25.4))
    axis.plot(
        coordinates["drive_nA"],
        coordinates["leading_real_per_ms"],
        color=theme.INK_BLACK,
        lw=1.1,
    )
    axis.axhline(0, color=theme.DEEP_RED, lw=0.7, ls="--")
    axis.axvline(onset["drive_nA"], color=theme.DEEP_RED, lw=0.7, ls=":")
    axis.plot(onset["drive_nA"], 0, "o", color=theme.DEEP_RED, ms=3)
    axis.annotate(
        f"{onset['drive_nA']:.3f} nA",
        (onset["drive_nA"], 0),
        xytext=(5, 7),
        textcoords="offset points",
        fontsize=theme.SIZE_ANNOTATION,
    )
    axis.set(xlabel="Tonic drive (nA)", ylabel=r"Leading Re$(\lambda_J)$ (ms$^{-1}$)")
    finish(axis)
    fig.subplots_adjust(left=0.23, right=0.96, bottom=0.22, top=0.95)
    save_figure(fig, destination / "hopf-onset", formats=("svg", "png"))
    plt.close(fig)


def criticality_figure(numbers, coordinates, destination):
    onset = numbers["reference"]["onset"]
    fig, axis = plt.subplots(figsize=(90 / 25.4, 68 / 25.4))
    axis.plot(
        coordinates["ramp_drive_nA"],
        coordinates["ramp_up_amplitude_per_ms"],
        color=theme.INK_BLACK,
        marker="o",
        ms=2.3,
        lw=1.1,
        label="Upward",
    )
    axis.plot(
        coordinates["ramp_drive_nA"],
        coordinates["ramp_down_amplitude_per_ms"],
        color=theme.DEEP_RED,
        marker="s",
        mfc="white",
        ms=2.0,
        lw=0.9,
        ls="--",
        label="Downward",
    )
    axis.axvline(onset["drive_nA"], color=theme.GREY_MID, lw=0.7, ls=":")
    axis.set(xlabel="Tonic drive (nA)", ylabel=r"E amplitude (pk–pk, ms$^{-1}$)")
    axis.legend(frameon=False, fontsize=theme.SIZE_LEGEND)
    finish(axis)
    fig.subplots_adjust(left=0.23, right=0.96, bottom=0.22, top=0.95)
    save_figure(fig, destination / "sampled-criticality", formats=("svg", "png"))
    plt.close(fig)


def frequency_figure(numbers, cfg, destination):
    fig, axis = plt.subplots(figsize=(90 / 25.4, 68 / 25.4))
    primary = [row for row in numbers["conditions"] if row["purpose"] == "gaba_sweep"]
    axis.plot(
        [row["condition"]["tau_GABA_ms"] for row in primary],
        [row["onset"]["frequency_Hz"] for row in primary],
        color=theme.INK_BLACK,
        marker="o",
        ms=2.7,
        lw=1.2,
        label="Reference closure",
    )
    robust = [
        row for row in numbers["conditions"] if row["purpose"] == "robustness_endpoint"
    ]
    for sigma_index, sigma in enumerate(cfg["robustness"]["sigma_corners_mV"]):
        for kappa_index, kappa in enumerate(cfg["robustness"]["kappa_corners"]):
            pair = [
                row
                for row in robust
                if row["condition"]["sigma_mV"] == sigma
                and row["condition"]["kappa"] == kappa
            ]
            pair.sort(key=lambda row: row["condition"]["tau_GABA_ms"])
            axis.plot(
                [row["condition"]["tau_GABA_ms"] for row in pair],
                [row["onset"]["frequency_Hz"] for row in pair],
                color=theme.GREY_LIGHT,
                lw=0.65,
                label="Closure corners"
                if sigma_index == 0 and kappa_index == 0
                else None,
            )
    axis.set(
        xlabel=r"Inhibitory decay $\tau_{GABA}$ (ms)",
        ylabel=r"Onset frequency $f_{Hopf}$ (Hz)",
    )
    axis.legend(frameon=False, fontsize=theme.SIZE_LEGEND)
    finish(axis)
    fig.subplots_adjust(left=0.23, right=0.96, bottom=0.22, top=0.95)
    save_figure(fig, destination / "frequency-vs-gaba", formats=("svg", "png"))
    plt.close(fig)


def figures(numbers, coordinates, cfg, destination):
    previous = theme.PAPER_MODE
    theme.set_paper_mode(True)
    theme.apply()
    try:
        onset_figure(numbers, coordinates, destination)
        criticality_figure(numbers, coordinates, destination)
        frequency_figure(numbers, cfg, destination)
    finally:
        theme.set_paper_mode(previous)
        theme.apply()


def present(identity, *, run_id=None):
    analysis = source_run(
        REPO / ".pingstore", identity, stage="analyse", experiment=recipe.SLUG
    )
    cfg = recipe.validate(analysis.record["execution"]["configuration"])
    if set(analysis.record["inputs"]) != {"compute"}:
        raise PingstoreError("exp116 analysis must identify one compute run")
    numbers = load_json(analysis.export / "results.json")
    if numbers.get("schema") != "exp116.analysis/v1":
        raise PingstoreError("unsupported exp116 analysis")
    if [row["condition"] for row in numbers["conditions"]] != recipe.conditions(cfg):
        raise PingstoreError("analysis condition order differs from the recipe")
    summary = numbers.get("summary", {})
    if not (
        summary.get("all_onsets_resolved")
        and summary.get("reference_criticality") == "consistent_with_supercritical"
        and summary.get("reference_gaba_frequencies_strictly_decrease")
        and summary.get("all_robustness_corners_support_direction")
    ):
        raise PingstoreError("exp116 evidence does not meet its prespecified goals")
    with stage_run(
        REPO,
        recipe.SLUG,
        "present",
        run_id=run_id,
        configuration=cfg,
        inputs={"analysis": analysis},
    ) as run:
        record_environment(run)
        run.record["execution"]["environment"]["matplotlib"] = matplotlib.__version__
        with np.load(
            analysis.export / "reference-branch.npz", allow_pickle=False
        ) as coordinates:
            figures(numbers, coordinates, cfg, run.export)
        write_json_atomic(run.export / "numbers.json", numbers)
        (run.export / "figure-captions.txt").write_text(
            "Hopf onset: leading Jacobian eigenvalue real part across tonic drive at the reference "
            "closure. Sampled criticality: upward and downward peak-to-peak excitatory-rate "
            "amplitudes over the final 500 ms of each 2-s step. Frequency versus GABA decay: the "
            "reference sweep and four low/high noise and relaxation endpoint checks. These are "
            "deterministic closure calculations, not measurements of the spiking network.\n"
        )
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="completed exp116 analysis run")
    parser.add_argument("--run-id", help="unused pre-reserved present identity")
    args = parser.parse_args()
    present(args.source, run_id=args.run_id)
