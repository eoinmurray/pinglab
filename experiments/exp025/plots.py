"""Render saved exp025 analysis; all scientific aggregation occurs upstream."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from experiments.helpers import theme
from experiments.helpers.figsave import save_figure

from .recipe import MODELS

MODEL_COLORS = {"coba": theme.DEEP_RED, "ping": theme.INK_BLACK}
MODEL_MARKERS = {"coba": "s", "ping": "D"}


def render_raster(npz_path: Path, out_path: Path, title: str) -> None:
    """Population spike raster from snapshot.npz."""
    theme.apply()
    data = np.load(npz_path)
    dt = float(data["dt"])
    T, n_e, n_i = int(data["T"]), int(data["n_e"]), int(data["n_i"])
    t_ms = np.arange(T) * dt
    has_i = data["i_t"].size > 0
    if has_i:
        fig, (ax_e, ax_i) = plt.subplots(
            2,
            1,
            figsize=(5.6, 3.15),
            sharex=True,
            gridspec_kw={"height_ratios": [4, 1]},
        )
    else:
        fig, ax_e = plt.subplots(1, 1, figsize=(5.6, 3.15))
        ax_i = None
    e_idx, e_t = data["e_cell"], data["e_t"]
    ax_e.scatter(t_ms[e_t], e_idx, s=1.0, c=theme.INK_BLACK, marker="|", linewidths=0.5)
    ax_e.set_ylabel("E neuron")
    ax_e.set_ylim(0, n_e)
    ax_e.set_xlim(0, T * dt)
    ax_e.set_title(title)
    if has_i:
        assert ax_i is not None  # has_i is only True when the 2-axes branch ran
        i_idx, i_t = data["i_cell"], data["i_t"]
        ax_i.scatter(
            t_ms[i_t], i_idx, s=1.0, c=theme.DEEP_RED, marker="|", linewidths=0.5
        )
        ax_i.set_ylabel("I neuron")
        ax_i.set_ylim(0, n_i)
        ax_i.set_xlim(0, T * dt)
        ax_i.set_xlabel("time (ms)")
    else:
        ax_e.set_xlabel("time (ms)")
    fig.tight_layout()
    save_figure(fig, out_path, formats=("png", "pdf"))  # dense raster: PNG, not SVG
    plt.close(fig)


def plot_rate_target_p_fgamma(
    rows: list[dict],
    out_path: Path,
    run_id: str,
) -> None:
    """4-panel decomposition vs rate target (Hz):
    (top-left)  p vs rate target (PING only) — the per-cycle participation gate
    (top-right) f_γ vs rate target (PING only) — biophysics, untouched by rate target
    (bottom-left)  E rate vs rate target (both architectures); overlay p · f_γ
    (bottom-right) accuracy vs rate target (both architectures)"""
    theme.apply()

    def by_model(model: str) -> list[dict]:
        sub = [
            r for r in rows if r["model"] == model and r["rate_target_hz"] is not None
        ]
        sub.sort(key=lambda r: r["rate_target_hz"])
        return sub

    ping = by_model("ping")
    coba = by_model("coba")

    fig, axes = plt.subplots(2, 2, figsize=(6.9, 5.018), dpi=150)
    (ax_p, ax_fg), (ax_r, ax_a) = axes

    if ping:
        ping_pf = [
            r for r in ping if r.get("p") is not None and r.get("f_gamma") is not None
        ]
        xs = [r["rate_target_hz"] for r in ping_pf]
        ax_p.plot(
            xs, [r["p"] for r in ping_pf], marker="o", color=theme.INK_BLACK, lw=1.5
        )
        ax_fg.plot(
            xs,
            [r["f_gamma"] for r in ping_pf],
            marker="s",
            color=theme.DEEP_RED,
            lw=1.5,
        )
        xs_all = [r["rate_target_hz"] for r in ping]
        ax_r.plot(
            xs_all,
            [r["e_rate"] for r in ping],
            marker="o",
            color=theme.INK_BLACK,
            lw=1.5,
            label="PING E (measured)",
        )
        # Predicted overlay stays greyscale-safe: near-black ink is reserved for the
        # measured trace, red is the single accent (COBA), so the prediction is grey
        # dashed with its own marker rather than a second chromatic hue (H13).
        ax_r.plot(
            xs,
            [r["p_times_f_gamma"] for r in ping_pf],
            marker="^",
            color=theme.GREY_MID,
            lw=1.5,
            ls="--",
            label="p × f_γ (predicted)",
        )
        ax_a.plot(
            xs_all,
            [r["acc"] for r in ping],
            marker="o",
            color=theme.INK_BLACK,
            lw=1.5,
            label="PING",
        )

    if coba:
        xs = [r["rate_target_hz"] for r in coba]
        ax_r.plot(
            xs,
            [r["e_rate"] for r in coba],
            marker="s",
            color=theme.DEEP_RED,
            lw=1.5,
            label="COBA E",
        )
        ax_a.plot(
            xs,
            [r["acc"] for r in coba],
            marker="s",
            color=theme.DEEP_RED,
            lw=1.5,
            label="COBA",
        )

    for ax in (ax_p, ax_fg, ax_r, ax_a):
        ax.set_xlabel("rate target (Hz)", fontsize=theme.SIZE_LABEL)
        ax.invert_xaxis()  # tightest penalty on the right read left → right
    ax_p.set_ylabel("p (per-cycle participation)", fontsize=theme.SIZE_LABEL)
    ax_p.set_title(
        "Participation gate vs rate target (PING)", fontsize=theme.SIZE_TITLE
    )
    ax_p.set_ylim(bottom=0)
    ax_fg.set_ylabel("f_γ (Hz)", fontsize=theme.SIZE_LABEL)
    ax_fg.set_title("Gamma frequency vs rate target (PING)", fontsize=theme.SIZE_TITLE)
    ax_fg.set_ylim(bottom=0)
    ax_r.set_ylabel("E firing rate (Hz)", fontsize=theme.SIZE_LABEL)
    ax_r.set_title(
        "E rate vs rate target — measured vs p · f_γ", fontsize=theme.SIZE_TITLE
    )
    ax_r.set_ylim(bottom=0)
    ax_r.legend(fontsize=theme.SIZE_LABEL, frameon=False, loc="upper left")
    ax_a.set_ylabel("Test accuracy (%)", fontsize=theme.SIZE_LABEL)
    ax_a.set_title("Accuracy vs rate target", fontsize=theme.SIZE_TITLE)
    ax_a.set_ylim(0, 100)
    ax_a.legend(fontsize=theme.SIZE_LABEL, frameon=False, loc="lower left")

    # No baked-in suptitle: the caption carries the takeaway (HOUSESTYLE H17).
    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


def plot_low_w_in(rows: list[dict], curves: dict, out_path: Path, run_id: str) -> None:
    """2 rows × 3 cols. One column per --w-in init. Top row: per-epoch
    accuracy. Bottom row: per-epoch firing rates with E (black) and I
    (red) overlaid. Reads per-epoch traces from each run's metrics.json."""
    theme.apply()
    n = len(rows)
    # Fixed 16:9 at column width (HOUSESTYLE H12) — one aspect across the writeup.
    # The earlier n-dependent height collapsed to a wide strip whose two stacked
    # y-axis labels overprinted each other; a taller frame gives each row room.
    fig, axes = plt.subplots(
        2,
        n,
        figsize=(6.9, 3.881),
        dpi=150,
        sharex=True,
    )
    rate_max = 0.0
    for col, row in enumerate(rows):
        curve = curves[f"{row['w_in']:g}"]
        epochs = curve["epochs"]
        acc_mean, rate_e_mean, rate_i_mean = [
            np.array(curve[k]) for k in ("acc_mean", "rate_e_mean", "rate_i_mean")
        ]
        rate_max = max(rate_max, curve["rate_max"])
        ax_acc = axes[0, col]
        ax_rate = axes[1, col]
        ax_acc.plot(epochs, acc_mean, color=theme.INK_BLACK, lw=1.5)
        if row["n_seeds"] > 1:
            acc_sem = np.array(curve["acc_sem"])
            ax_acc.fill_between(
                epochs,
                acc_mean - acc_sem,
                acc_mean + acc_sem,
                color=theme.INK_BLACK,
                alpha=0.16,
                linewidth=0,
            )
        ax_acc.axhline(10.0, color=theme.GREY_MID, lw=0.6, ls=":", alpha=0.5)
        ax_acc.set_ylim(0, 100)
        ax_acc.set_title(
            f"$W_\\text{{in}}$ = {row['w_in']:g}",
            fontsize=theme.SIZE_TITLE,
        )
        ax_rate.plot(epochs, rate_e_mean, color=theme.INK_BLACK, lw=1.5, label="E")
        ax_rate.plot(epochs, rate_i_mean, color=theme.DEEP_RED, lw=1.5, label="I")
        if row["n_seeds"] > 1:
            for mean, key, color in (
                (rate_e_mean, "rate_e_sem", theme.INK_BLACK),
                (rate_i_mean, "rate_i_sem", theme.DEEP_RED),
            ):
                sem = np.array(curve[key])
                ax_rate.fill_between(
                    epochs,
                    mean - sem,
                    mean + sem,
                    color=color,
                    alpha=0.14,
                    linewidth=0,
                )
        ax_rate.set_xlabel("Epoch", fontsize=theme.SIZE_LABEL)
        if col == 0:
            ax_acc.set_ylabel("Test accuracy (%)", fontsize=theme.SIZE_LABEL)
            ax_rate.set_ylabel("Firing rate (Hz)", fontsize=theme.SIZE_LABEL)
            ax_rate.legend(fontsize=theme.SIZE_LABEL, frameon=False, loc="upper left")

    for col in range(n):
        axes[1, col].set_ylim(0, rate_max * 1.1 if rate_max > 0 else 1.0)

    # No baked-in suptitle: the caption carries the takeaway (HOUSESTYLE H17). The
    # per-column $W_"in"$ headers identify the panels.
    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


def plot_w_in_scale_sweep(
    rows: list[dict], f_star_s: float | None, out_path: Path, run_id: str
) -> None:
    """Six-panel: CE loss, penalty, total loss, accuracy, E rate, I rate
    vs W_in scale. One curve per (model, rate_target_hz) cell."""
    theme.apply()
    fig, axes_2d = plt.subplots(2, 3, figsize=(6.9, 4.6), dpi=150)
    axes = axes_2d.flatten()
    styles = {
        "coba@rt1hz": ("COBA (1 Hz target)", theme.DEEP_RED, "s", "-"),
        "ping@rt1hz": ("PING (1 Hz target)", theme.INK_BLACK, "o", "-"),
    }
    for ax in axes:
        ax.set_xlabel("$W_\\text{in}$ scale $s$", fontsize=theme.SIZE_LABEL)
        ax.axvline(1.0, color=theme.GREY_MID, lw=0.6, ls="--", alpha=0.7)
        if f_star_s is not None:
            ax.axvline(f_star_s, color=theme.INK_BLACK, lw=0.8, ls=":", alpha=0.7)
    for cell, (label, color, marker, ls) in styles.items():
        msel = [r for r in rows if r["cell"] == cell]
        if not msel:
            continue
        xs = [r["scale"] for r in msel]
        axes[0].plot(
            xs,
            [r["loss"] for r in msel],
            marker=marker,
            color=color,
            lw=1.5,
            ls=ls,
            label=label,
        )
        axes[1].plot(
            xs,
            [r["penalty"] for r in msel],
            marker=marker,
            color=color,
            lw=1.5,
            ls=ls,
            label=label,
        )
        axes[2].plot(
            xs,
            [r["total_loss"] for r in msel],
            marker=marker,
            color=color,
            lw=1.5,
            ls=ls,
            label=label,
        )
        axes[3].plot(
            xs,
            [r["acc"] for r in msel],
            marker=marker,
            color=color,
            lw=1.5,
            ls=ls,
            label=label,
        )
        axes[4].plot(
            xs,
            [r["rate_e"] for r in msel],
            marker=marker,
            color=color,
            lw=1.5,
            ls=ls,
            label=label,
        )
        axes[5].plot(
            xs,
            [r["rate_i"] for r in msel],
            marker=marker,
            color=color,
            lw=1.5,
            ls=ls,
            label=label,
        )
    axes[0].set_ylabel("Test cross-entropy", fontsize=theme.SIZE_LABEL)
    axes[0].set_title("CE loss", fontsize=theme.SIZE_TITLE)
    if f_star_s is not None:
        ylim = axes[0].get_ylim()
        axes[0].text(
            f_star_s,
            ylim[1] * 0.95,
            "$\\approx f^\\star$",
            ha="left",
            va="top",
            fontsize=theme.SIZE_ANNOTATION,
            color=theme.INK_BLACK,
        )
    axes[1].set_ylabel("Spike-budget penalty", fontsize=theme.SIZE_LABEL)
    axes[1].set_title("Penalty", fontsize=theme.SIZE_TITLE)
    axes[1].set_ylim(0, 4.0)
    axes[2].set_ylabel("CE + penalty", fontsize=theme.SIZE_LABEL)
    axes[2].set_title("Training-objective loss", fontsize=theme.SIZE_TITLE)
    axes[2].set_ylim(0, 4.0)
    axes[3].set_ylabel("Test accuracy (%)", fontsize=theme.SIZE_LABEL)
    axes[3].set_ylim(0, 100)
    axes[3].axhline(10.0, color=theme.GREY_MID, lw=0.6, ls=":", alpha=0.5)
    axes[3].set_title("Accuracy", fontsize=theme.SIZE_TITLE)
    axes[4].set_ylabel("E rate (Hz)", fontsize=theme.SIZE_LABEL)
    axes[4].set_title("E rate", fontsize=theme.SIZE_TITLE)
    axes[5].set_ylabel("I rate (Hz)", fontsize=theme.SIZE_LABEL)
    axes[5].set_title("I rate", fontsize=theme.SIZE_TITLE)
    axes[0].legend(fontsize=theme.SIZE_LABEL, frameon=False, loc="upper right")
    # No baked-in suptitle: the caption carries the takeaway (HOUSESTYLE H17). The
    # s = 1 dashed line and ≈ f* dotted line are already annotated in-panel.
    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


def plot_w_in_scale_sweep_vs_rate(
    rows: list[dict], out_path: Path, run_id: str
) -> None:
    """Same data as plot_w_in_scale_sweep, but x-axis is E rate instead
    of W_in scale s. Y-axes: CE | penalty | total loss | accuracy |
    I rate | s. Each cell's trained s=1 point marked with a filled star
    on every curve so the reader sees where on the rate axis training
    landed."""
    theme.apply()
    fig, axes_2d = plt.subplots(2, 3, figsize=(6.9, 4.6), dpi=150)
    axes = axes_2d.flatten()
    styles = {
        "coba@rt1hz": ("COBA (1 Hz target)", theme.DEEP_RED, "s", "-"),
        "ping@rt1hz": ("PING (1 Hz target)", theme.INK_BLACK, "o", "-"),
    }
    for ax in axes:
        ax.set_xlabel("Hidden E rate (Hz)", fontsize=theme.SIZE_LABEL)
    # Order each cell by E rate so lines don't backtrack.
    for cell, (label, color, marker, ls) in styles.items():
        msel = sorted(
            (r for r in rows if r["cell"] == cell),
            key=lambda r: r["rate_e"],
        )
        if not msel:
            continue
        xs = [r["rate_e"] for r in msel]
        axes[0].plot(
            xs,
            [r["loss"] for r in msel],
            marker=marker,
            color=color,
            lw=1.5,
            ls=ls,
            label=label,
        )
        axes[1].plot(
            xs,
            [r["penalty"] for r in msel],
            marker=marker,
            color=color,
            lw=1.5,
            ls=ls,
            label=label,
        )
        axes[2].plot(
            xs,
            [r["total_loss"] for r in msel],
            marker=marker,
            color=color,
            lw=1.5,
            ls=ls,
            label=label,
        )
        axes[3].plot(
            xs,
            [r["acc"] for r in msel],
            marker=marker,
            color=color,
            lw=1.5,
            ls=ls,
            label=label,
        )
        axes[4].plot(
            xs,
            [r["rate_i"] for r in msel],
            marker=marker,
            color=color,
            lw=1.5,
            ls=ls,
            label=label,
        )
        axes[5].plot(
            xs,
            [r["scale"] for r in msel],
            marker=marker,
            color=color,
            lw=1.5,
            ls=ls,
            label=label,
        )
        # Mark each cell's trained operating point (s = 1) with a star.
        trained = next((r for r in msel if abs(r["scale"] - 1.0) < 1e-6), None)
        if trained is not None:
            star_kwargs = dict(
                marker="*",
                color=color,
                markersize=16,
                markeredgecolor=theme.INK_BLACK,
                markeredgewidth=0.7,
                linestyle="None",
                zorder=5,
            )
            axes[0].plot([trained["rate_e"]], [trained["loss"]], **star_kwargs)
            axes[1].plot([trained["rate_e"]], [trained["penalty"]], **star_kwargs)
            axes[2].plot([trained["rate_e"]], [trained["total_loss"]], **star_kwargs)
            axes[3].plot([trained["rate_e"]], [trained["acc"]], **star_kwargs)
            axes[4].plot([trained["rate_e"]], [trained["rate_i"]], **star_kwargs)
            axes[5].plot([trained["rate_e"]], [trained["scale"]], **star_kwargs)
    axes[0].set_ylabel("Test cross-entropy", fontsize=theme.SIZE_LABEL)
    axes[0].set_title("CE loss", fontsize=theme.SIZE_TITLE)
    axes[1].set_ylabel("Spike-budget penalty", fontsize=theme.SIZE_LABEL)
    axes[1].set_title("Penalty", fontsize=theme.SIZE_TITLE)
    axes[1].set_ylim(0, 4.0)
    axes[2].set_ylabel("CE + penalty", fontsize=theme.SIZE_LABEL)
    axes[2].set_title("Training-objective loss", fontsize=theme.SIZE_TITLE)
    axes[2].set_ylim(0, 4.0)
    axes[3].set_ylabel("Test accuracy (%)", fontsize=theme.SIZE_LABEL)
    axes[3].set_ylim(0, 100)
    axes[3].axhline(10.0, color=theme.GREY_MID, lw=0.6, ls=":", alpha=0.5)
    axes[3].set_title("Accuracy", fontsize=theme.SIZE_TITLE)
    axes[4].set_ylabel("I rate (Hz)", fontsize=theme.SIZE_LABEL)
    axes[4].set_title("I rate", fontsize=theme.SIZE_TITLE)
    axes[5].set_ylabel("$W_\\text{in}$ scale $s$", fontsize=theme.SIZE_LABEL)
    axes[5].set_title("$W_\\text{in}$ scale", fontsize=theme.SIZE_TITLE)
    axes[0].legend(fontsize=theme.SIZE_LABEL, frameon=False, loc="upper right")
    # No baked-in suptitle: the caption carries the takeaway (HOUSESTYLE H17). The
    # trained-point stars are already annotated in-panel.
    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


def _despine(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def fig_results_compound(frontier_stats, curves, npz_coba, npz_ping, out_path, run_id):
    """exp023-Figure-1-style super figure (replotted from cache, no retraining):
    top row two trained-baseline rasters (COBA | PING), bottom row four small
    plots — train loss, test accuracy, accuracy–rate frontier, accuracy/rate
    bars."""
    theme.apply()
    plt.rcParams["savefig.bbox"] = "standard"  # keep the saved 16:9 exact
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(6.9, 3.881), dpi=150)  # 16:9
    gs = GridSpec(
        2,
        2,
        figure=fig,
        height_ratios=[3.0, 2.6],
        hspace=0.45,
        wspace=0.2,
        top=0.93,
        bottom=0.1,
        left=0.07,
        right=0.96,
    )

    # --- top row: two rasters side by side (E black, I red above) ---
    for col, (npz_path, title) in enumerate(
        [
            (npz_coba, "COBA — loop off"),
            (npz_ping, "PING — loop on"),
        ]
    ):
        ax = fig.add_subplot(gs[0, col])
        data = np.load(npz_path)
        dt = float(data["dt"])
        T, N_E, N_I = int(data["T"]), int(data["n_e"]), int(data["n_i"])
        t_ms = np.arange(T) * dt
        gap = max(8, N_E // 40)
        e_t, e_n = data["e_t"], data["e_cell"]
        ax.scatter(
            t_ms[e_t], e_n, s=0.8, c=theme.INK_BLACK, marker="|", linewidths=0.35
        )
        if N_I > 0 and data["i_t"].size > 0:
            i_t, i_n = data["i_t"], data["i_cell"]
            ax.scatter(
                t_ms[i_t],
                i_n + N_E + gap,
                s=1.0,
                c=theme.DEEP_RED,
                marker="|",
                linewidths=0.45,
            )
            ax.set_ylim(-2, N_E + N_I + gap + 2)
            ax.set_yticks([N_E / 2, N_E + gap + N_I / 2])
            ax.set_yticklabels(["E", "I"])
        else:
            ax.set_ylim(-2, N_E + 2)
            ax.set_yticks([N_E / 2])
            ax.set_yticklabels(["E"])
            ax.text(
                T * dt * 0.985,
                N_E - 30,
                "I silent (loop off)",
                ha="right",
                va="top",
                fontsize=theme.SIZE_LABEL - 1,
                color=theme.MUTED,
                fontstyle="italic",
                # Opaque backing so the note reads over the dense raster
                # instead of smearing into it like a ghost watermark.
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.75,
                ),
            )
        ax.set_xlim(0, T * dt)
        ax.set_xlabel("time (ms)")
        ax.tick_params(axis="y", length=0)
        ax.set_title(title, loc="left", fontweight="semibold")
        _despine(ax)

    # --- bottom-left: validation accuracy per epoch (checkpoint selection) ---
    ax_acc = fig.add_subplot(gs[1, 0])
    for m in MODELS:
        curve = curves[m]
        ax_acc.plot(
            curve["epochs"],
            curve["acc_mean"],
            marker=MODEL_MARKERS[m],
            ms=3.2,
            markevery=10,
            lw=1.4,
            color=MODEL_COLORS[m],
            label=m.upper(),
        )
    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylabel("validation accuracy (%)")
    ax_acc.set_ylim(0, 100)
    # No baked-in title: the caption carries the takeaway (HOUSESTYLE H17).
    ax_acc.legend(fontsize=theme.SIZE_LEGEND, frameon=False, loc="lower right")
    _despine(ax_acc)

    # --- bottom-right: accuracy–rate frontier, operating points annotated ---
    ax_fr = fig.add_subplot(gs[1, 1])
    model_curves = {}
    xmax = 1.0
    for m in MODELS:
        pts = [point for point in frontier_stats if point["model"] == m]
        pts.sort(key=lambda point: point["rate_mean"])
        base_point = next(
            (point for point in pts if point["rate_target_hz"] is None), None
        )
        base = (
            (base_point["rate_mean"], base_point["acc_mean"])
            if base_point is not None
            else None
        )
        model_curves[m] = (pts, base)
        if pts:
            xmax = max(xmax, max(point["rate_mean"] for point in pts))
    ax_fr.set_xlim(
        -xmax * 0.03, xmax * 1.12
    )  # left margin so near-zero points read; right headroom for the COBA label
    ax_fr.set_ylim(40, 100)  # all points ≥ 54%; crop dead space to grow the frontier
    for m in MODELS:
        pts, base = model_curves[m]
        if pts:
            ax_fr.errorbar(
                [point["rate_mean"] for point in pts],
                [point["acc_mean"] for point in pts],
                xerr=[point["rate_sem"] for point in pts],
                yerr=[point["acc_sem"] for point in pts],
                marker=MODEL_MARKERS[m],
                ms=4,
                lw=1.4,
                capsize=2,
                color=MODEL_COLORS[m],
                label=m.upper(),
            )
    for m in MODELS:
        base = model_curves[m][1]
        if base is None:
            continue
        ax_fr.scatter(
            [base[0]],
            [base[1]],
            s=130,
            marker="*",
            color=MODEL_COLORS[m],
            edgecolor=theme.INK_BLACK,
            linewidths=0.7,
            zorder=6,
        )
        # PING star: label up-right into open plot space; COBA star sits top-
        # right, so label down-left to avoid clipping the axis and the title.
        if m == "ping":
            # Label up-and-right of the star, into the open space above the trace,
            # with enough offset that it never crowds the frontier line.
            dxdy, ha, va = (14, 5), "left", "bottom"
        else:
            dxdy, ha, va = (-6, -8), "right", "top"
        ax_fr.annotate(
            f"{m.upper()}\n{base[1]:.0f}% @ {base[0]:.0f} Hz",
            (base[0], base[1]),
            xytext=dxdy,
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=theme.SIZE_ANNOTATION,
            color=MODEL_COLORS[m],
        )
    ax_fr.set_xlabel("hidden-E firing rate (Hz)")
    ax_fr.set_ylabel("test accuracy (%)")
    # No baked-in title: the caption carries the takeaway (HOUSESTYLE H17).
    ax_fr.legend(
        fontsize=theme.SIZE_LEGEND,
        frameon=False,
        loc="lower right",
        title="★ = spike budget off",
        title_fontsize=theme.SIZE_ANNOTATION,
    )
    _despine(ax_fr)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path, formats=("png", "pdf"))  # dense rasters: PNG, not SVG
    plt.close(fig)
