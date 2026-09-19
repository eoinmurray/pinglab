"""Render the single exp116 evidence figure without recomputation."""

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


def compound(numbers, coordinates, cfg, destination):
    previous = theme.PAPER_MODE
    theme.set_paper_mode(True)
    theme.apply()
    fig, axes = plt.subplots(1, 3, figsize=(180 / 25.4, 64 / 25.4))
    reference = numbers["reference"]
    onset = reference["onset"]
    axes[0].plot(
        coordinates["drive_nA"],
        coordinates["leading_real_per_ms"],
        color=theme.INK_BLACK,
        lw=1.1,
    )
    axes[0].axhline(0, color=theme.DEEP_RED, lw=0.7, ls="--")
    if onset:
        axes[0].axvline(onset["drive_nA"], color=theme.DEEP_RED, lw=0.7, ls=":")
        axes[0].plot(onset["drive_nA"], 0, "o", color=theme.DEEP_RED, ms=3)
    axes[0].set(
        xlabel="Tonic drive (nA)", ylabel=r"Leading Re$(\lambda_J)$ (ms$^{-1}$)"
    )

    axes[1].plot(
        coordinates["ramp_drive_nA"],
        coordinates["ramp_up_amplitude_per_ms"],
        color=theme.INK_BLACK,
        marker="o",
        ms=2.3,
        lw=1.1,
        label="Upward",
    )
    axes[1].plot(
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
    axes[1].axvline(onset["drive_nA"], color=theme.GREY_MID, lw=0.7, ls=":")
    axes[1].set(xlabel="Tonic drive (nA)", ylabel=r"E amplitude (pk–pk, ms$^{-1}$)")
    axes[1].legend(frameon=False, fontsize=theme.SIZE_LEGEND)

    primary = [row for row in numbers["conditions"] if row["purpose"] == "gaba_sweep"]
    axes[2].plot(
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
    for sigma in cfg["robustness"]["sigma_corners_mV"]:
        for kappa in cfg["robustness"]["kappa_corners"]:
            pair = [
                row
                for row in robust
                if row["condition"]["sigma_mV"] == sigma
                and row["condition"]["kappa"] == kappa
            ]
            pair.sort(key=lambda row: row["condition"]["tau_GABA_ms"])
            axes[2].plot(
                [row["condition"]["tau_GABA_ms"] for row in pair],
                [row["onset"]["frequency_Hz"] for row in pair],
                color=theme.GREY_LIGHT,
                lw=0.65,
            )
    axes[2].set(
        xlabel=r"Inhibitory decay $\tau_{GABA}$ (ms)",
        ylabel=r"Onset frequency $f_{Hopf}$ (Hz)",
    )
    for title, axis in zip(
        ("Equilibrium stability", "Amplitude ramps", "Inhibitory timescale"), axes
    ):
        axis.set_title(
            title, loc="left", fontsize=theme.SIZE_LABEL, fontweight="semibold"
        )
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=theme.SIZE_TICK)
    theme.label_panels(axes)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.25, top=0.83, wspace=0.47)
    save_figure(fig, destination / "minimal-hopf-evidence", formats=("svg", "png"))
    plt.close(fig)
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
            compound(numbers, coordinates, cfg, run.export)
        write_json_atomic(run.export / "numbers.json", numbers)
        (run.export / "figure-caption.txt").write_text(
            "A: leading Jacobian eigenvalue real part across tonic drive for the reference closure; "
            "the marker identifies the accepted stable-to-unstable crossing. B: upward and downward "
            "peak-to-peak excitatory-rate amplitudes over the final 500 ms of each 2-s step. C: onset "
            "frequency across the six inhibitory decay times at the reference closure; light endpoint "
            "segments show the four low/high noise and relaxation corner checks. These deterministic "
            "closure calculations are not measurements of the separate spiking network.\n"
        )
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="completed exp116 analysis run")
    parser.add_argument("--run-id", help="unused pre-reserved present identity")
    args = parser.parse_args()
    present(args.source, run_id=args.run_id)
