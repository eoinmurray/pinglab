"""Render exp117 analysis evidence into flat article presentation files.

Invoke this stage explicitly from the repository root. Scientific outputs
and input provenance belong to its independently completed Pingstore run.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

import argparse
import shutil

import matplotlib.pyplot as plt
import numpy as np
from experiments import theme
from experiments.exp117.custom import inputs
from matplotlib.gridspec import GridSpec
from pingstore.contracts import PingstoreError, load_json, write_json_atomic


def present(identity, *, run_id=None):
    analysis = inputs.source(REPO, identity, "analyse")
    cfg = inputs.configuration(analysis)
    if set(analysis.record["inputs"]) != {"compute"}:
        raise PingstoreError("exp117 analysis must pin one compute run")
    results = load_json(analysis.export / "results.json")
    coordinates = load_json(analysis.export / "plot_coordinates.json")
    if results.get("schema") != "exp117.analysis/v3":
        raise PingstoreError("unsupported exp117 analysis payload")
    if coordinates.get("schema") != "exp117.plot-coordinates/v4":
        raise PingstoreError("unsupported exp117 plot coordinates")
    if results.get("configuration") != cfg:
        raise PingstoreError("exp117 analysis payload and recipe disagree")
    with inputs.execution(
        REPO,
        "present",
        sources={"analysis": analysis},
        run_id=run_id,
        configuration=cfg,
    ) as run:
        if cfg["schema"] == "exp117.recipe/v4":
            reference = analysis.record["inputs"]["compute"]
            computed = inputs.source(
                REPO, reference["run_id"], "compute", reference=reference
            )
            for filename in ("circuit.svg", "mean-field-model.json"):
                shutil.copyfile(computed.file(filename), run.export / filename)
        bifurcation_compound(coordinates, run.export / "bifurcation_compound.svg")
        write_json_atomic(run.export / "numbers.json", results)
    return run.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    print(present(args.source, run_id=args.run_id))


def bifurcation_compound(coordinates, output):
    """Render Hopf crossing, sampled criticality and inhibitory timescale."""
    rows = coordinates["rows"]
    hopf = coordinates["hopf"]
    criticality = coordinates["criticality"]
    tau_rows = coordinates["frequency_vs_tau_GABA"]

    drive = np.asarray([row["I_ext_nA"] for row in rows], dtype=float)
    eigenvalue_real = np.asarray(
        [[value[0] for value in row["eigenvalues_per_ms"]] for row in rows],
        dtype=float,
    )
    eigenvalue_imag = np.asarray(
        [[value[1] for value in row["eigenvalues_per_ms"]] for row in rows],
        dtype=float,
    )
    relative_drive = np.asarray(criticality["relative_drive_nA"], dtype=float)
    amplitude_up = np.asarray(criticality["amplitude_up_Hz"], dtype=float)
    amplitude_down = np.asarray(criticality["amplitude_down_Hz"], dtype=float)
    tau_gaba = np.asarray([row["tau_GABA_ms"] for row in tau_rows], dtype=float)
    frequency = np.asarray([row["f_Hopf_Hz"] for row in tau_rows], dtype=float)

    theme.apply()
    figure = plt.figure(figsize=(13.5, 4.6), dpi=150)
    grid = GridSpec(
        1,
        3,
        figure=figure,
        wspace=0.42,
        top=0.86,
        bottom=0.16,
        left=0.06,
        right=0.97,
    )
    axis_stability = figure.add_subplot(grid[0, 0])
    axis_criticality = figure.add_subplot(grid[0, 1])
    axis_frequency = figure.add_subplot(grid[0, 2])

    scatter = None
    for index in range(eigenvalue_real.shape[1]):
        scatter = axis_stability.scatter(
            eigenvalue_real[:, index],
            eigenvalue_imag[:, index],
            c=drive,
            cmap="pinglab_brand",
            s=5,
            linewidths=0,
        )
    axis_stability.axvline(0, color=theme.GREY_MID, ls=":", lw=0.8)
    omega = hopf["omega_Hopf_rad_per_ms"]
    axis_stability.scatter(
        [0, 0],
        [omega, -omega],
        facecolors="none",
        edgecolors=theme.ELECTRIC_CYAN,
        s=60,
        lw=1.4,
        zorder=5,
    )
    colorbar = figure.colorbar(scatter, ax=axis_stability, fraction=0.046, pad=0.02)
    colorbar.set_label(r"$I_{\mathrm{ext}}$ (nA)", fontsize=theme.SIZE_TICK - 1)
    colorbar.ax.tick_params(labelsize=theme.SIZE_TICK - 1)
    axis_stability.set_xlabel(r"Re$(\lambda_J)$ (ms$^{-1}$)")
    axis_stability.set_ylabel(r"Im$(\lambda_J)$ (rad/ms)")
    axis_stability.set_title(
        rf"Hopf crossing at $I^\star$ = {hopf['I_ext_star_nA']:.2f} nA",
        loc="left",
    )

    axis_criticality.plot(
        relative_drive,
        amplitude_up,
        "o-",
        color=theme.INK_BLACK,
        ms=4.5,
        label="ascending",
    )
    axis_criticality.plot(
        relative_drive,
        amplitude_down,
        "s--",
        color=theme.DEEP_RED,
        markerfacecolor="none",
        ms=4.5,
        label="descending",
    )
    axis_criticality.axvline(0, color=theme.AMBER, ls=":", lw=0.8)
    axis_criticality.set_xlabel(r"$I_{\mathrm{ext}}-I^\star$ (nA)")
    axis_criticality.set_ylabel(r"$A_{\mathrm{pp}}$ of $r_E$ (Hz)")
    axis_criticality.set_title(
        "Reversible sampled onset",
        loc="left",
    )
    axis_criticality.legend(loc="upper left")

    axis_frequency.plot(
        tau_gaba,
        frequency,
        "o-",
        color=theme.INK_BLACK,
        ms=4.5,
    )
    axis_frequency.set_xlabel(r"$\tau_{\mathrm{GABA}}$ (ms)")
    axis_frequency.set_ylabel(r"$f_{\mathrm{Hopf}}$ (Hz)")
    axis_frequency.set_title(
        "Slower inhibition, lower frequency",
        loc="left",
    )

    theme.label_panels((axis_stability, axis_criticality, axis_frequency))
    figure.savefig(output)
    plt.close(figure)


if __name__ == "__main__":
    main()
