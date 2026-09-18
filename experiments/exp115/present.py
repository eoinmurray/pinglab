"""Render explicit analysed Hopf evidence without recomputing the model."""

import argparse
import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from experiments.exp115 import recipe
from experiments.exp115.compute import environment
from experiments.helpers import theme
from pingstore.contracts import PingstoreError, load_json, write_json_atomic
from pingstore.stages import source_run, stage_run


def compound(numbers, coordinates, cfg, destination):
    theme.set_paper_mode(True)
    theme.apply()
    fig, axes = plt.subplots(
        1, 3, figsize=(180 / 25.4, 77 / 25.4), layout="constrained"
    )
    ax = axes[0]
    ax.plot(
        coordinates["drive_nA"],
        coordinates["leading_real_per_ms"],
        color=theme.INK_BLACK,
        linewidth=1.1,
    )
    ax.axhline(0, color=theme.DEEP_RED, linewidth=0.7, linestyle="--")
    onset = numbers["reference"]["onset"]
    if onset is not None:
        ax.axvline(
            onset["drive_nA"], color=theme.DEEP_RED, linewidth=0.7, linestyle=":"
        )
        ax.plot(onset["drive_nA"], 0, "o", color=theme.DEEP_RED, markersize=3)
        ax.annotate(
            f"{onset['drive_nA']:.3f} nA",
            (onset["drive_nA"], 0),
            xytext=(5, 8),
            textcoords="offset points",
            fontsize=7,
        )
    else:
        ax.text(
            0.04, 0.96, "Onset unresolved", transform=ax.transAxes, va="top", fontsize=7
        )
    ax.set(xlabel="Tonic drive (nA)", ylabel=r"Leading Re$(\lambda_J)$ (ms$^{-1}$)")
    by_condition = {
        (
            r["condition"]["tau_GABA_ms"],
            r["condition"]["sigma_mV"],
            r["condition"]["kappa"],
        ): r
        for r in numbers["conditions"]
    }
    if "ramp_drive_nA" in coordinates:
        axes[1].plot(
            coordinates["ramp_drive_nA"],
            coordinates["ramp_up_amplitude_per_ms"],
            color=theme.INK_BLACK,
            linewidth=1.1,
            marker="o",
            markersize=2.3,
            label="Upward ramp",
        )
        axes[1].plot(
            coordinates["ramp_drive_nA"],
            coordinates["ramp_down_amplitude_per_ms"],
            color=theme.DEEP_RED,
            linewidth=0.9,
            marker="o",
            markersize=2.0,
            label="Downward ramp",
        )
        if onset is not None:
            axes[1].axvline(onset["drive_nA"], color=theme.GREY_MID, linewidth=0.7, linestyle=":")
        axes[1].legend(loc="best", fontsize=6, frameon=False)
    else:
        axes[1].text(0.04, 0.96, "Ramps unresolved", transform=axes[1].transAxes, va="top", fontsize=7)
    axes[1].set(xlabel="Tonic drive (nA)", ylabel=r"E amplitude (pk–pk, ms$^{-1}$)")
    reference = (cfg["reference"]["sigma_mV"], cfg["reference"]["kappa"])
    pairs = [(sigma, kappa) for sigma in cfg["sigma_grid_mV"] for kappa in cfg["kappa_grid"]]
    pairs.sort(key=lambda pair: pair == reference)
    for pair in pairs:
        values = []
        for tau in cfg["tau_GABA_grid_ms"]:
            result = by_condition[(tau, *pair)]["onset"]
            values.append(result["frequency_Hz"] if result else np.nan)
        is_reference = pair == reference
        axes[2].plot(
            cfg["tau_GABA_grid_ms"], values,
            color=theme.INK_BLACK if is_reference else theme.GREY_LIGHT,
            linewidth=1.2 if is_reference else 0.45,
            marker="o" if is_reference else None,
            markersize=2.7,
        )
    axes[2].set_xlabel(r"Inhibitory decay $\tau_{GABA}$ (ms)")
    axes[2].set_ylabel(r"Onset frequency $f_{Hopf}$ (Hz)")
    for letter, title, ax in zip(
        "ABC", ("Equilibrium stability", "Amplitude ramps", "Onset frequency"), axes
    ):
        ax.set_title(f"{letter}  {title}", fontsize=8, loc="left")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=7)
    from matplotlib.lines import Line2D

    axes[2].legend(
        handles=[
            Line2D([], [], color=theme.INK_BLACK, label="Reference closure"),
            Line2D([], [], color=theme.GREY_LIGHT, label="Other closure choices"),
        ],
        loc="best",
        fontsize=6,
        frameon=False,
    )
    fig.savefig(destination / "hopf-compound.svg")
    fig.savefig(destination / "hopf-compound.png", dpi=220)
    plt.close(fig)


def present(identity, *, run_id=None):
    analysis = source_run(
        REPO / ".pingstore", identity, stage="analyse", experiment=recipe.SLUG
    )
    cfg = recipe.validate(analysis.record["execution"]["configuration"])
    if set(analysis.record["inputs"]) != {"compute"}:
        raise PingstoreError("exp115 analysis must identify its compute evidence")
    pin = analysis.record["inputs"]["compute"]
    compute = source_run(
        REPO / ".pingstore",
        pin["run_id"],
        stage="compute",
        experiment=recipe.SLUG,
        reference=pin,
    )
    if compute.record["execution"]["configuration"] != cfg:
        raise PingstoreError("analysis and compute recipes disagree")
    numbers = load_json(analysis.export / "results.json")
    if numbers.get("schema") != "exp115.analysis/v2":
        raise PingstoreError("unsupported exp115 analysis measurements")
    if [r["condition"] for r in numbers["conditions"]] != recipe.conditions(cfg):
        raise PingstoreError("analysis is missing closure conditions")
    with stage_run(
        REPO,
        recipe.SLUG,
        "present",
        inputs={"analysis": analysis},
        run_id=run_id,
        configuration=cfg,
    ) as run:
        environment(run)
        run.record["execution"]["environment"]["matplotlib"] = matplotlib.__version__
        with np.load(
            analysis.export / "reference-branch.npz", allow_pickle=False
        ) as coordinates:
            compound(numbers, coordinates, cfg, run.export)
        write_json_atomic(run.export / "numbers.json", numbers)
        fields = [
            "tau_GABA_ms",
            "sigma_mV",
            "kappa",
            "status",
            "crossings",
            "onset_nA",
            "frequency_Hz",
            "criticality_assessment",
            "branch_gap_per_ms",
            "amplitude_squared_slope_per_ms2_nA",
            "amplitude_squared_r2",
            "chi_per_ms_nA",
            "max_equilibrium_residual",
        ]
        with (run.export / "sensitivity.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in numbers["conditions"]:
                onset = row["onset"] or {}
                criticality = row["criticality"] or {}
                writer.writerow(
                    {
                        **row["condition"],
                        "status": row["status"],
                        "crossings": len(row["crossings"]),
                        "onset_nA": onset.get("drive_nA"),
                        "frequency_Hz": onset.get("frequency_Hz"),
                        "chi_per_ms_nA": onset.get("chi_per_ms_nA"),
                        "criticality_assessment": criticality.get("assessment"),
                        "branch_gap_per_ms": criticality.get("branch_gap_per_ms"),
                        "amplitude_squared_slope_per_ms2_nA": criticality.get("amplitude_squared_slope_per_ms2_nA"),
                        "amplitude_squared_r2": criticality.get("amplitude_squared_r2"),
                        "max_equilibrium_residual": row["max_equilibrium_residual"],
                    }
                )
        (run.export / "figure-caption.txt").write_text(
            "A: leading Jacobian eigenvalue real part versus tonic drive at inhibitory decay 6 ms, "
            "effective voltage-noise scale 4 mV and rate-relaxation multiplier 1; the red marker "
            "identifies an accepted stable-to-unstable crossing, when available. B: upward and "
            "downward peak-to-peak excitatory-rate amplitudes over the final 500 ms of each 2-s "
            "drive step at the reference closure. "
            "C: corresponding eigenvalue-derived onset frequency. In C, the black curve fixes the "
            "reference noise scale and relaxation; grey curves show the other 19 closure choices. "
            "Missing or unresolved onsets break curves. These are deterministic sensitivity "
            "comparisons, not statistical uncertainty intervals. Finite ramps can miss narrow "
            "bistability intervals and unstable cycles.\n"
        )
        compute.check_unchanged()
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="completed exp115 analysis run")
    parser.add_argument("--run-id", help="unused pre-reserved present identity")
    args = parser.parse_args()
    present(args.source, run_id=args.run_id)
