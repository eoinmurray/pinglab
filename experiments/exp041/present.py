"""Render saved analysis only; no simulation, measurement or publication.

Invoke this stage explicitly from the repository root. Scientific outputs
and input provenance belong to its independently completed Pingstore run.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "libraries")]

import argparse
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from experiments import theme
from experiments.exp041.custom import inputs
from experiments.exp041.custom.network_artifacts import copy_circuits
from experiments.exp041.recipe import F_GAMMA_BAND_HZ, TAU_GABA_REFERENCE_MS
from pingstore.library import PingstoreError, load_json, write_json_atomic
from snnlab import viz

DEFAULT_FORMATS = ("svg", "pdf")


def schematic_diagrams():
    """Lay out the documented model; these are illustrations, not execution evidence."""
    specifications = [
        {
            "name": "ping",
            "title": "PING classifier",
            "nodes": [
                ("X", "Poisson spikes", "input", (-360, 0)),
                ("E", "Excitatory neurons", "population", (0, 0)),
                ("I", "Inhibitory neurons", "population", (0, 300)),
                ("R", "Class readout", "population", (360, 0)),
            ],
            "edges": [
                ("X", "E", "excitatory"),
                ("E", "I", "excitatory"),
                ("I", "E", "inhibitory"),
                ("E", "R", "signal"),
            ],
        }
    ]
    diagrams = {}
    ports = dict(
        top_left=(0.25, 0),
        top_right=(0.75, 0),
        bottom_left=(0.25, 1),
        bottom_right=(0.75, 1),
    )
    for spec in specifications:
        visual = viz.Diagram(
            name="exp041_" + spec["name"],
            title=spec["title"],
            nodes=tuple(
                viz.DiagramNode(identifier, identifier, detail, "", kind=kind)
                for identifier, detail, kind, _ in spec["nodes"]
            ),
            edges=tuple(
                viz.DiagramEdge(
                    source,
                    target,
                    id=source + "_to_" + target,
                    role="signal" if role == "average" else role,
                )
                for source, target, role in spec["edges"]
            ),
            metadata=dict(
                view="illustrative documented model; no historical bundle reconstructed"
            ),
        )
        positions = {
            identifier: position for identifier, _, _, position in spec["nodes"]
        }
        routes = {}
        for source, target, role in spec["edges"]:
            label = (
                r"$\langle V-E_L\rangle_t$"
                if role == "average"
                else rf"$W_{{{target},{source}}}$"
            )
            if (source, target) == ("E", "I"):
                route = dict(source_port="bottom_left", target_port="top_left")
            elif (source, target) == ("I", "E"):
                route = dict(source_port="top_right", target_port="bottom_right")
            elif source == "J":
                route = dict(source_port="top", target_port="bottom")
            elif positions[source][0] == positions[target][0]:
                route = dict(source_port="bottom", target_port="top")
            elif positions[source][0] > positions[target][0]:
                route = dict(source_port="left", target_port="right")
            else:
                route = dict(source_port="right", target_port="left")
            routes[source + "_to_" + target] = viz.EdgeLayout(
                label=viz.DiagramLabel(label), **route
            )
        layout = viz.DiagramLayout(
            nodes={
                identifier: viz.NodeLayout(
                    position=position,
                    label=f"${identifier}$",
                    min_size=(240, 160),
                    ports=ports,
                )
                for identifier, position in positions.items()
            },
            edges=routes,
            defaults=viz.LayoutDefaults(
                node=viz.NodeLayout(content_align="center", padding=16), gap=30
            ),
        )
        with matplotlib.rc_context({"savefig.bbox": None}):
            diagrams[spec["name"]] = viz.resolve_diagram(
                visual, layout=layout, theme=viz.DiagramTheme(secondary_size=13)
            )
    return diagrams


def draw_schematics():
    """Regenerate report-local SVG illustrations without executing scientific stages."""
    destination = Path(__file__).parent / "report" / "images"
    destination.mkdir(exist_ok=True)
    with matplotlib.rc_context({"savefig.bbox": None}):
        for name, diagram in schematic_diagrams().items():
            viz.render_diagram(diagram, destination / (name + "-network.svg"))


def save_figure(fig, stem, formats=DEFAULT_FORMATS) -> list[Path]:
    """Write *fig* to <stem>.<ext> for each ext in *formats*.

    *stem* is a path without an extension (a trailing .png/.pdf/.svg is
    stripped, so existing call sites that pass "foo.png" keep working).
    Format is inferred from the extension; rcParams (dpi, bbox, fonttype)
    come from the active theme profile.
    """
    stem = Path(stem)
    if stem.suffix:
        stem = stem.with_suffix("")
    out: list[Path] = []
    for ext in formats:
        p = stem.with_suffix(f".{ext}")
        fig.savefig(p)
        out.append(p)
    return out


def format_duration(seconds: float) -> str:
    s = int(round(seconds))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60:02d}s"
    return f"{s // 3600}h {(s % 3600) // 60:02d}m"


def present(identity: str, *, run_id: str | None = None) -> str:

    analysis = inputs.source(REPO, identity, "analyse")
    if set(analysis.record["inputs"]) != {"compute", "bank"}:
        raise PingstoreError(
            "exp041 analysis must pin its computation and training bank"
        )
    ref = analysis.record["inputs"]["compute"]
    compute = inputs.source(REPO, ref["run_id"], "compute", reference=ref)
    cfg = inputs.configuration(compute)
    if compute.record["inputs"]["bank"] != analysis.record["inputs"]["bank"]:
        raise PingstoreError("analysis and compute disagree on the training bank")
    result = load_json(analysis.export / "results.json")
    if (
        result.get("schema") != "exp041.analysis/v1"
        or result.get("recipe") != cfg
        or result.get("measurement") != analysis.record["execution"]["configuration"]
    ):
        raise PingstoreError("unsupported or inconsistent exp041 analysis payload")
    samples = []
    with np.load(analysis.export / "rasters.npz", allow_pickle=False) as data:
        for row in result["rasters"]:
            samples.append(
                {
                    **row,
                    "e": np.array(data[row["cell_name"] + "__e"]),
                    "i": np.array(data[row["cell_name"] + "__i"]),
                }
            )
    started = time.monotonic()
    with inputs.execution(
        REPO,
        "present",
        sources={"analysis": analysis},
        run_id=run_id,
        configuration={
            "schema": "exp041.presentation/v1",
            "paper_mode": True,
            "raster_window_ms": cfg["raster"]["window_ms"],
        },
    ) as run:
        theme.set_paper_mode(True)
        plot_quantitative_law(
            result["aggregate"], result["fit"], run.export / "rate_vs_fgamma"
        )
        plot_psd_panel(result["aggregate"], run.export / "psds")
        plot_per_trial_peaks(result["aggregate"], run.export / "per_trial_peaks")
        plot_raster_strip(
            samples, run.export / "raster_strip", cfg["raster"]["window_ms"]
        )
        plot_training_curves(
            result["curves"],
            cfg["tau_gaba_sweep_ms"],
            cfg["seeds"],
            run.export / "training_curves",
        )
        run.record["presentation_lineage"] = {
            "analysis": analysis.reference,
            "operation": "render saved measurements and selected raster samples",
            "historical_training_curves": analysis.record["inputs"]["bank"],
            "compute_evidence": analysis.record["inputs"]["compute"],
        }
        write_json_atomic(
            run.export / "numbers.json",
            {
                **{
                    k: v
                    for k, v in result.items()
                    if k not in ("aggregate", "rasters", "curves")
                },
                "run_id": run.run_id,
                "duration_s": time.monotonic() - started,
                "duration": format_duration(time.monotonic() - started),
                "git_sha": run.record["provenance"]["git_commit"],
                "circuit_files": copy_circuits(run),
            },
        )
    return run.run_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="completed exp041 analyse ID")
    parser.add_argument("--run-id", help="unused v4 reservation")
    args = parser.parse_args()
    try:
        present(args.source, run_id=args.run_id)
    except PingstoreError as exc:
        parser.exit(1, f"exp041 present: {exc}\n")


def plot_quantitative_law(
    rows: list[dict], fit: dict, out_path: Path, *, axes=None
) -> None:
    """Seed-mean E rate and accuracy versus gamma frequency, with the saved affine fit."""
    theme.apply()

    standalone = axes is None
    if standalone:
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(6.5, 4.2),
            sharex=True,
            gridspec_kw={"hspace": 0.12, "height_ratios": [1.6, 1.0]},
        )
    ax_rate, ax_acc = axes
    fig = ax_rate.figure

    f_gammas: list[float] = []
    for row in rows:
        tau = row["tau_gaba_ms"]
        fg_mu, fg_se = row["f_gamma_hz"]["mean"], row["f_gamma_hz"]["sem"]
        er_mu, er_se = row["e_rate_hz"]["mean"], row["e_rate_hz"]["sem"]
        ac_mu, ac_se = row["acc"]["mean"], row["acc"]["sem"]
        f_gammas.append(fg_mu)
        ax_rate.errorbar(
            fg_mu,
            er_mu,
            xerr=fg_se,
            yerr=er_se,
            fmt="o",
            markersize=6 if standalone else 3,
            color=theme.INK_BLACK,
            capsize=3 if standalone else 2,
            label=f"τ_GABA = {tau:g} ms"
            if standalone and tau == TAU_GABA_REFERENCE_MS
            else None,
        )
        ax_rate.annotate(
            f" {tau:g} ms",
            (fg_mu, er_mu),
            fontsize=theme.SIZE_ANNOTATION,
            color=theme.MUTED,
            xytext=(9 if standalone else 4, -3),
            textcoords="offset points",
            va="center",
        )
        ax_acc.errorbar(
            fg_mu,
            ac_mu,
            xerr=fg_se,
            yerr=ac_se,
            fmt="o",
            markersize=6 if standalone else 3,
            color=theme.INK_BLACK,
            capsize=3 if standalone else 2,
        )

    fg_arr = np.array(f_gammas)
    p_fit, a_fit, r2 = fit["p_affine"], fit["a_affine"], fit["r2_affine"]
    if p_fit is not None:
        xs = np.linspace(0, fg_arr.max() * 1.1, 200)
        ax_rate.plot(
            xs,
            p_fit * xs + a_fit,
            color=theme.DEEP_RED,
            lw=1.2,
            ls="--",
            label=(
                "$r_E = a + p · f_γ$"
                + ("  " if standalone else "\n")
                + f"(a = {a_fit:.2f} Hz, p = {p_fit:.3f},"
                + (" " if standalone else "\n")
                + f"R² = {r2:.3f})"
            ),
        )
    ax_rate.set_ylabel("Hidden E rate (Hz)", fontsize=theme.SIZE_LABEL)
    ax_acc.set_ylabel("Test accuracy (%)", fontsize=theme.SIZE_LABEL)
    if standalone:
        ax_acc.set_xlabel(
            "Measured $f_γ$ (Hz) — peak of trained-network population PSD",
            fontsize=theme.SIZE_LABEL,
        )
    else:
        for ax in axes:
            ax.set_xlabel("Measured $f_γ$ (Hz)", fontsize=theme.SIZE_LABEL)
    ax_acc.set_ylim(0, 100)
    ax_acc.axhline(10.0, color=theme.GREY_MID, lw=0.5, ls=":", alpha=0.6)
    for ax in (ax_rate, ax_acc):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=theme.SIZE_TICK)
    if not standalone:
        ax_rate.margins(x=0.15, y=0.35)
        ax_acc.set_xlim(ax_rate.get_xlim())
    if not standalone:
        ax_rate.set_ylim(bottom=0)
        for line in ax_rate.lines:
            if line.get_label().startswith("$r_E"):
                line.set_label("Affine fit")
    ax_rate.legend(fontsize=theme.SIZE_LEGEND, frameon=False, loc="upper left")
    theme.label_panels((ax_rate, ax_acc))
    if standalone:
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_figure(fig, out_path)
        plt.close(fig)


def plot_training_curves(
    curves: list[dict], taus: list[float], seeds: list[int], out_path: Path
) -> None:
    """Per-cell training-trajectory curves so convergence is auditable
    by inspection. One line per (τ_GABA, seed); colour by τ_GABA."""
    theme.apply()
    cmap = plt.get_cmap("viridis")
    taus_sorted = list(taus)
    fig, (ax_acc, ax_rate) = plt.subplots(
        2,
        1,
        figsize=(6.5, 5.175),
        sharex=True,
        gridspec_kw={"hspace": 0.15},
    )
    for i, tau in enumerate(taus_sorted):
        color = cmap(i / max(1, len(taus_sorted) - 1))
        for j, seed in enumerate(seeds):
            m = next(
                row
                for row in curves
                if row["tau_gaba_ms"] == tau and row["seed"] == seed
            )
            eps = [e["ep"] for e in m["epochs"]]
            accs = [e.get("acc", 0) for e in m["epochs"]]
            rates = [e.get("test_rate_e", 0) for e in m["epochs"]]
            label = f"τ_GABA = {tau:g} ms" if j == 0 else None
            ax_acc.plot(eps, accs, color=color, lw=1.0, alpha=0.85, label=label)
            ax_rate.plot(eps, rates, color=color, lw=1.0, alpha=0.85)
    ax_acc.set_ylabel("Validation accuracy (%)", fontsize=theme.SIZE_LABEL)
    ax_rate.set_ylabel("Validation E rate (Hz)", fontsize=theme.SIZE_LABEL)
    ax_rate.set_xlabel("Epoch", fontsize=theme.SIZE_LABEL)
    ax_acc.legend(fontsize=theme.SIZE_LEGEND, frameon=False, ncol=2, loc="lower right")
    ax_acc.set_ylim(0, 100)
    for ax in (ax_acc, ax_rate):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.15, lw=0.4)
    theme.label_panels((ax_acc, ax_rate))
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path)
    plt.close(fig)


def plot_psd_panel(rows: list[dict], out_path: Path) -> None:
    """One PSD curve per τ_GABA (mean across seeds), to verify the
    peak-frequency identification is reading the right feature."""
    theme.apply()
    fig, ax = plt.subplots(figsize=(6.5, 3.15))
    cmap = plt.get_cmap("viridis")
    taus_sorted = [r["tau_gaba_ms"] for r in rows]
    for i, row in enumerate(rows):
        tau = row["tau_gaba_ms"]
        freqs = np.array(row["freqs_hz"])
        psd_mean = np.array(row["psd"])
        band_mask = (freqs >= F_GAMMA_BAND_HZ[0]) & (freqs <= F_GAMMA_BAND_HZ[1])
        ax.plot(
            freqs[band_mask],
            psd_mean[band_mask],
            color=cmap(i / max(1, len(taus_sorted) - 1)),
            label=f"τ_GABA = {tau:g} ms",
            lw=1.2,
        )
        peak_f = row["psd_marker"]["frequency_hz"]
        peak_p = row["psd_marker"]["power"]
        ax.scatter(
            [peak_f],
            [peak_p],
            color=cmap(i / max(1, len(taus_sorted) - 1)),
            s=20,
            zorder=5,
        )
    ax.set_xlabel("Frequency (Hz)", fontsize=theme.SIZE_LABEL)
    ax.set_ylabel("Population PSD (a.u.)", fontsize=theme.SIZE_LABEL)
    ax.set_xlim(F_GAMMA_BAND_HZ)
    ax.legend(fontsize=theme.SIZE_LEGEND, frameon=False, ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path)
    plt.close(fig)


def plot_per_trial_peaks(
    rows: list[dict],
    out_path: Path,
) -> None:
    """Sanity check: per-trial PSD peak distribution per τ_GABA.

    Pools per-trial peak frequencies across seeds for each τ_GABA value
    and shows their histogram alongside the trial-mean-PSD peak.
    Narrow histograms → trial-mean-PSD f_γ is unbiased; wide histograms
    → trial-mean PSD is a centroid, and per-trial median would differ.
    """
    theme.apply()
    taus_sorted = [r["tau_gaba_ms"] for r in rows]
    n_taus = len(rows)
    cmap = plt.get_cmap("viridis")

    fig, axes = plt.subplots(
        n_taus,
        1,
        figsize=(6.5, (1.5 * n_taus + 1.0) * 0.8625),
        sharex=True,
    )
    if n_taus == 1:
        axes = [axes]
    for ax, row in zip(axes, rows, strict=True):
        tau = row["tau_gaba_ms"]
        stats = row["trial_peaks"]
        color = cmap(taus_sorted.index(tau) / max(1, n_taus - 1))
        if stats["median_hz"] is not None:
            ax.stairs(
                stats["counts"],
                stats["bins_hz"],
                fill=True,
                color=color,
                alpha=0.85,
                edgecolor=theme.INK_BLACK,
                lw=0.4,
            )
            median = stats["median_hz"]
            iqr = stats["iqr_hz"]
            ax.axvline(median, color=theme.INK_BLACK, ls="--", lw=1.0)
            ax.text(
                0.98,
                0.85,
                f"τ_GABA = {tau:g} ms\n"
                f"per-trial: median {median:.1f} Hz, IQR {iqr:.1f} Hz",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=theme.SIZE_LABEL,
            )
        mean_peak = row["f_gamma_hz"]["mean"]
        ax.axvline(
            mean_peak,
            color=theme.DEEP_RED,
            ls="-",
            lw=1.2,
            label=f"trial-mean PSD peak: {mean_peak:.1f} Hz",
        )
        ax.legend(fontsize=theme.SIZE_LEGEND, frameon=False, loc="upper left")
        ax.set_ylabel("trials", fontsize=theme.SIZE_LABEL)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xlabel("Per-trial PSD peak frequency (Hz)", fontsize=theme.SIZE_LABEL)
    theme.label_panels(axes)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path)
    plt.close(fig)


def plot_raster_strip(
    samples: list[dict],
    out_path: Path,
    t_window_ms: float,
) -> None:
    """One stacked single-trial raster per τ_GABA cluster. X-axis is
    physical time in ms (not steps), so the gamma cadence shift with
    τ_GABA is read by eye — shorter τ_GABA gives faster bursts gives
    more E spikes per unit time."""
    theme.apply()
    # Show τ_GABA in ascending value, top-down (so faster gamma at the top).
    samples = sorted(samples, key=lambda s: s["tau_gaba_ms"])
    n = len(samples)
    n_e = samples[0]["e"].shape[1]
    n_i = samples[0]["i"].shape[1]
    gap = 6
    fig, axes = plt.subplots(
        n,
        1,
        figsize=(6.5, (1.0 * n + 1.0) * 0.69),
        sharex=True,
        gridspec_kw={"hspace": 0.22},
    )
    if n == 1:
        axes = [axes]
    for i, (ax, s) in enumerate(zip(axes, samples, strict=True)):
        T = s["e"].shape[0]
        t_axis = np.arange(T) * s["dt_ms"]
        mask = t_axis <= t_window_ms
        e_t, e_n = np.where(s["e"][mask])
        i_t, i_n = np.where(s["i"][mask])
        ax.scatter(
            t_axis[mask][e_t], e_n, s=2.0, c=theme.INK_BLACK, marker="|", linewidths=0.4
        )
        ax.scatter(
            t_axis[mask][i_t],
            i_n + n_e + gap,
            s=2.0,
            c=theme.DEEP_RED,
            marker="|",
            linewidths=0.4,
        )
        ax.set_ylim(-2, n_e + n_i + gap + 2)
        ax.set_yticks([n_e / 2, n_e + gap + n_i / 2])
        ax.set_yticklabels(["E", "I"])
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(0, t_window_ms)
        ax.text(
            1.012,
            0.5,
            f"τ_GABA = {s['tau_gaba_ms']:g} ms\nE = {s['e_rate_hz']:.1f} Hz",
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=theme.SIZE_LABEL,
        )
        if i < n - 1:
            ax.tick_params(axis="x", labelbottom=False)
    axes[-1].set_xlabel("time (ms)")
    theme.label_panels(axes)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path, formats=("svg", "pdf"))
    plt.close(fig)


if __name__ == "__main__":
    if "--draw-schematics" in sys.argv[1:]:
        parser = argparse.ArgumentParser(
            description="Render illustrative model diagrams only."
        )
        parser.add_argument("--draw-schematics", action="store_true", required=True)
        parser.parse_args()
        draw_schematics()
    else:
        main()
