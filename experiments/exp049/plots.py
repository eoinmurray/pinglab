"""Render saved analysis arrays only; aggregation is owned by measurements."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from experiments.helpers import theme
from experiments.helpers.figsave import save_figure

from .recipe import COND_ORDER, CONDITIONS, F_GAMMA_BAND_HZ

COND_COLOURS = {
    "frozen_ping": theme.MUTED,
    "trainable_ping_init": theme.INK_BLACK,
    "trainable_zero_init": theme.DEEP_RED,
    "trainable_small_init": theme.AMBER,
}
COND_MARKERS = {
    "frozen_ping": "o",
    "trainable_ping_init": "s",
    "trainable_zero_init": "^",
    "trainable_small_init": "D",
}


def plot_weight_matrices(
    cond: str,
    data: dict,
    out_path: Path,
    run_id: str,
) -> None:
    """Per-condition weight-distribution card: W^EI and W^IE, init vs trained.

    Two panels side-by-side. Each panel shows the *surviving* (>0) weights as
    overlaid histograms (init outline, trained fill). The fraction of entries
    pruned to zero by Dale's-law clamping is shown as a small bar chart inset
    so the spike-at-zero doesn't dominate the histogram visually.
    """
    theme.apply()
    from matplotlib.gridspec import GridSpec

    label = CONDITIONS[cond]["label"]
    seeds_sorted = data["seeds"]
    canon_ei, canon_ie = 1 / 1024, 2 / 256

    fig = plt.figure(figsize=(6.9, 3.05), dpi=150)
    gs = GridSpec(
        2,
        2,
        figure=fig,
        width_ratios=[1.0, 1.0],
        height_ratios=[0.18, 1.0],
        hspace=0.34,
        wspace=0.42,
        top=0.94,
        bottom=0.17,
        left=0.07,
        right=0.98,
    )
    ax_hdr = fig.add_subplot(gs[0, :])
    ax_ei = fig.add_subplot(gs[1, 0])
    ax_ie = fig.add_subplot(gs[1, 1])

    # --- header ---
    ax_hdr.set_axis_off()
    ax_hdr.text(
        0.0,
        0.72,
        label,
        transform=ax_hdr.transAxes,
        ha="left",
        va="center",
        fontsize=theme.SIZE_TITLE + 1,
        fontweight="semibold",
        color=COND_COLOURS[cond],
    )
    ax_hdr.text(
        0.0,
        0.10,
        f"Recurrent-weight distributions · pooled across {len(seeds_sorted)} seeds · post-projection values",
        transform=ax_hdr.transAxes,
        ha="left",
        va="center",
        fontsize=theme.SIZE_CAPTION,
        color=theme.LABEL,
        fontfamily="monospace",
    )

    def _panel(ax, entry, title, color, canon_mean):
        bins = np.asarray(entry["bins"])
        stats = entry["stats"]
        eff_init = stats["init_mean"]
        eff_trained = stats["trained_mean"]
        frac_pruned_init = stats["init_zero_fraction"]
        frac_pruned_trained = stats["trained_zero_fraction"]

        # Surviving (>0) distributions only — keeps the histogram readable.
        if entry["has_initial"]:
            ax.stairs(entry["initial"], bins, color=color, lw=1.6, label="init")
        if entry["has_trained"]:
            ax.stairs(
                entry["trained"],
                bins,
                fill=True,
                color=color,
                alpha=0.32,
                lw=0.8,
                label="trained",
            )

        # Reference vertical at the canonical biophysical mean.
        ax.axvline(canon_mean, color=theme.GREY_MID, lw=0.9, ls=":")
        y_hi = ax.get_ylim()[1]
        ax.text(
            canon_mean,
            y_hi * 0.5,
            "  canonical",
            ha="left",
            va="center",
            fontsize=theme.SIZE_CAPTION,
            color=theme.GREY_MID,
            rotation=90,
        )

        ax.set_title(title, fontsize=theme.SIZE_LABEL, loc="left", pad=4)
        ax.set_xlabel(
            "conductance magnitude (surviving > 0)", fontsize=theme.SIZE_LABEL
        )
        ax.set_ylabel("entry count", fontsize=theme.SIZE_LABEL)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=theme.SIZE_LABEL - 1, direction="out", length=3)
        ax.set_xlim(0, bins[-1])
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(
                handles,
                labels,
                fontsize=theme.SIZE_LEGEND,
                frameon=False,
                loc="lower right",
            )

        # Stats box (upper-right): the two key numbers per row.
        stat_text = (
            f"init     mean = {eff_init:.4f}    pruned = {frac_pruned_init:5.1%}\n"
            f"trained  mean = {eff_trained:.4f}    pruned = {frac_pruned_trained:5.1%}"
        )
        ax.text(
            0.99,
            0.78,
            stat_text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=theme.SIZE_CAPTION,
            color=theme.LABEL,
            fontfamily="monospace",
            bbox=dict(
                facecolor="white",
                edgecolor=theme.GREY_MID,
                lw=0.5,
                boxstyle="round,pad=0.4",
                alpha=0.95,
            ),
        )

    _panel(
        ax_ei,
        data["weights"]["ei"],
        r"$W^{EI}$  (1024 × 256)",
        theme.INK_BLACK,
        canon_ei,
    )
    _panel(
        ax_ie,
        data["weights"]["ie"],
        r"$W^{IE}$  (256 × 1024)",
        theme.DEEP_RED,
        canon_ie,
    )

    theme.label_panels((ax_ei, ax_ie))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path, formats=("svg", "pdf"))
    plt.close(fig)


def plot_condition_card(
    cond: str,
    data: dict,
    raster: dict | None,
    out_path: Path,
    run_id: str,
) -> None:
    """Per-condition diagnostic card: trajectory strip on top, PSD + raster below."""
    theme.apply()
    from matplotlib.gridspec import GridSpec
    from matplotlib.lines import Line2D

    color = COND_COLOURS[cond]
    label = CONDITIONS[cond]["label"]

    header = data["header"]
    acc_final, e_rate_f, i_rate_f, f_gamma_f = [
        float(header[k]) if header[k] is not None else float("nan")
        for k in ("acc", "e_rate_hz", "i_rate_hz", "f_gamma_hz")
    ]

    fig = plt.figure(figsize=(6.9, 5.1), dpi=150)
    gs = GridSpec(
        3,
        4,
        figure=fig,
        height_ratios=[0.32, 1.0, 1.4],
        hspace=0.85,
        wspace=0.40,
        top=0.96,
        bottom=0.07,
        left=0.06,
        right=0.97,
    )
    ax_hdr = fig.add_subplot(gs[0, :])
    ax_wei = fig.add_subplot(gs[1, 0])
    ax_wie = fig.add_subplot(gs[1, 1])
    ax_rate = fig.add_subplot(gs[1, 2])
    ax_acc = fig.add_subplot(gs[1, 3])
    ax_psd = fig.add_subplot(gs[2, 0:2])
    ax_rast = fig.add_subplot(gs[2, 2:4])

    # --- header strip ---
    ax_hdr.set_axis_off()
    ax_hdr.text(
        0.0,
        1.0,
        label,
        transform=ax_hdr.transAxes,
        ha="left",
        va="top",
        fontsize=theme.SIZE_TITLE + 2,
        fontweight="semibold",
        color=color,
    )
    stat_pieces = [
        f"test acc = {acc_final:5.2f}%",
        f"E = {e_rate_f:5.1f} Hz" if e_rate_f == e_rate_f else "E = —",
        f"I = {i_rate_f:5.1f} Hz" if i_rate_f == i_rate_f else "I = —",
        f"mean peak = {f_gamma_f:4.1f} Hz"
        if f_gamma_f == f_gamma_f
        else "mean peak = —",
    ]
    ax_hdr.text(
        0.0,
        0.0,
        "    ".join(stat_pieces),
        transform=ax_hdr.transAxes,
        ha="left",
        va="bottom",
        fontsize=theme.SIZE_LABEL + 1,
        color=theme.LABEL,
        fontfamily="monospace",
    )

    # --- trajectory strip (per-seed alpha + mean) ---
    def _per_seed_curves(key, sub_key=None):
        curve = data["curves"][sub_key or key]
        return curve["ep"], curve

    def _plot_traj(ax, xs, curve, *, ls="-", lw_mean=1.8):
        for row in curve["rows"]:
            ax.plot(xs, row, color=color, lw=0.7, ls=ls, alpha=0.28)
        ax.plot(
            xs, curve["mean"], color=color, lw=lw_mean, ls=ls, solid_capstyle="round"
        )

    def _style_panel(ax, *, ylabel, last_val=None, last_val_fmt="{:.4f}"):
        ax.set_ylabel(ylabel, fontsize=theme.SIZE_LABEL)
        ax.set_xlabel("Epoch", fontsize=theme.SIZE_LABEL)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=theme.SIZE_LABEL - 1, direction="out", length=3)
        if last_val is not None and last_val == last_val:
            ax.text(
                0.99,
                0.97,
                last_val_fmt.format(last_val),
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=theme.SIZE_LABEL - 1,
                color=color,
                fontweight="semibold",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.0),
            )

    def _empty_panel(ax, ylabel, message):
        ax.set_ylabel(ylabel, fontsize=theme.SIZE_LABEL)
        ax.set_xlabel("Epoch", fontsize=theme.SIZE_LABEL)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=theme.SIZE_LABEL - 1, direction="out", length=3)
        ax.set_xlim(0, data["curves"]["acc"]["ep"][-1])
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.text(
            0.5,
            0.5,
            message,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=theme.SIZE_LABEL,
            color=theme.GREY_MID,
            fontstyle="italic",
        )

    # ‖W^EI‖
    xs, arr = _per_seed_curves("weight_norms", "W_ei.1")
    if xs is not None and arr["visible"]:
        _plot_traj(ax_wei, xs, arr)
        _style_panel(
            ax_wei,
            ylabel=r"$\|W^{EI}\|_F$",
            last_val=arr["last"],
            last_val_fmt="end={:.3f}",
        )
    else:
        msg = "frozen" if cond == "frozen_ping" else "no positive\nnorm recorded"
        _empty_panel(ax_wei, ylabel=r"$\|W^{EI}\|_F$", message=msg)
    # ‖W^IE‖
    xs, arr = _per_seed_curves("weight_norms", "W_ie.1")
    if xs is not None and arr["visible"]:
        _plot_traj(ax_wie, xs, arr)
        _style_panel(
            ax_wie,
            ylabel=r"$\|W^{IE}\|_F$",
            last_val=arr["last"],
            last_val_fmt="end={:.3f}",
        )
    else:
        msg = "frozen" if cond == "frozen_ping" else "no positive\nnorm recorded"
        _empty_panel(ax_wie, ylabel=r"$\|W^{IE}\|_F$", message=msg)
    # Firing rates: E (solid) + I (dashed)
    xs_e, arr_e = _per_seed_curves("rate_e")
    xs_i, arr_i = _per_seed_curves("rate_i")
    if xs_e is not None:
        _plot_traj(ax_rate, xs_e, arr_e, ls="-")
    if xs_i is not None:
        _plot_traj(ax_rate, xs_i, arr_i, ls="--")
    _style_panel(ax_rate, ylabel="Reference rate (Hz)")
    ax_rate.legend(
        handles=[
            Line2D([0], [0], color=color, lw=1.8, ls="-", label="E"),
            Line2D([0], [0], color=color, lw=1.8, ls="--", label="I"),
        ],
        fontsize=theme.SIZE_LEGEND,
        frameon=False,
        loc="upper left",
    )
    # Accuracy
    xs, arr = _per_seed_curves("acc")
    if xs is not None:
        _plot_traj(ax_acc, xs, arr)
        _style_panel(
            ax_acc,
            ylabel="Validation accuracy (%)",
            last_val=arr["last"],
            last_val_fmt="end={:.1f}%",
        )
        ax_acc.set_ylim(0, 100)
        ax_acc.axhline(10.0, color=theme.GREY_MID, lw=0.6, ls=":", alpha=0.6)
        ax_acc.text(
            0.02,
            0.13,
            "chance",
            transform=ax_acc.transAxes,
            fontsize=theme.SIZE_CAPTION,
            color=theme.GREY_MID,
        )

    # --- final PSD ---
    if data["psd"]:
        freqs = data["psd"]["frequencies"]
        for row in data["psd"]["rows"]:
            ax_psd.plot(freqs, row, color=color, lw=0.7, alpha=0.3)
        ax_psd.plot(freqs, data["psd"]["mean"], color=color, lw=1.8)
        # Mark the retained mean of seed-wise spectral peak bins if defined.
        if (
            f_gamma_f == f_gamma_f
            and F_GAMMA_BAND_HZ[0] <= f_gamma_f <= F_GAMMA_BAND_HZ[1]
        ):
            ax_psd.axvline(f_gamma_f, color=color, lw=0.9, ls="--", alpha=0.55)
            ax_psd.text(
                0.98,
                0.95,
                f"mean peak = {f_gamma_f:.1f} Hz",
                transform=ax_psd.transAxes,
                ha="right",
                va="top",
                fontsize=theme.SIZE_LABEL - 1,
                color=color,
                fontweight="semibold",
            )
    ax_psd.set_xlabel("Frequency (Hz)", fontsize=theme.SIZE_LABEL)
    ax_psd.set_ylabel("Population E PSD (a.u.)", fontsize=theme.SIZE_LABEL)
    ax_psd.set_xlim(F_GAMMA_BAND_HZ)
    ax_psd.spines["top"].set_visible(False)
    ax_psd.spines["right"].set_visible(False)
    ax_psd.tick_params(labelsize=theme.SIZE_LABEL - 1, direction="out", length=3)
    ax_psd.set_title(
        "Trained-network E-population PSD", fontsize=theme.SIZE_LABEL, loc="left", pad=4
    )

    # --- single-trial raster ---
    if raster is not None:
        n_e, n_i = raster["n_e"], raster["n_i"]
        n_e_plot, n_i_plot = raster["n_e_plot"], raster["n_i_plot"]
        ax_rast.scatter(
            raster["e_t"],
            raster["e_n"],
            s=1.6,
            c=theme.INK_BLACK,
            marker="|",
            linewidths=0.35,
        )
        if n_i > 0:
            divider_y = n_e_plot + 4
            ax_rast.axhline(divider_y, color=theme.GREY_MID, lw=0.5, alpha=0.5)
            ax_rast.scatter(
                raster["i_t"],
                raster["i_n"] + divider_y + 4,
                s=1.6,
                c=theme.DEEP_RED,
                marker="|",
                linewidths=0.35,
            )
            ax_rast.set_ylim(-2, divider_y + 4 + n_i_plot + 2)
            ax_rast.set_yticks([n_e_plot / 2, divider_y + 4 + n_i_plot / 2])
            ax_rast.set_yticklabels(
                [f"E ({n_e})", f"I ({n_i})"],
                fontsize=theme.SIZE_LABEL - 1,
            )
        else:
            ax_rast.set_ylim(-2, n_e_plot + 2)
            ax_rast.set_yticks([n_e_plot / 2])
            ax_rast.set_yticklabels([f"E ({n_e})"], fontsize=theme.SIZE_LABEL - 1)
    ax_rast.set_xlabel("Time (ms)", fontsize=theme.SIZE_LABEL)
    ax_rast.spines["top"].set_visible(False)
    ax_rast.spines["right"].set_visible(False)
    ax_rast.tick_params(labelsize=theme.SIZE_LABEL - 1, direction="out", length=3)
    ax_rast.set_title(
        "Single-trial raster (seed 42)", fontsize=theme.SIZE_LABEL, loc="left", pad=4
    )

    theme.label_panels((ax_wei, ax_wie, ax_rate, ax_acc, ax_psd, ax_rast))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path, formats=("png", "pdf"))
    plt.close(fig)


def fig_attractor(data, out_path, run_id):
    """Final-checkpoint official-test E/I rates; points retain individual seeds."""
    theme.apply()
    fig, ax = plt.subplots(figsize=(5.6, 3.5), dpi=150)
    ax.axhline(0.0, color=theme.FAINT, lw=0.8, ls=":")
    for cond in COND_ORDER:
        entry = data[cond]
        E, I, acc = entry["e"], entry["i"], entry["acc"]
        ax.scatter(
            E,
            I,
            s=95,
            color=COND_COLOURS[cond],
            marker=COND_MARKERS[cond],
            edgecolor="white",
            linewidths=0.7,
            zorder=5,
            label=f"{CONDITIONS[cond]['label']}  ·  {acc:.1f}%",
        )
    ax.set_xlabel("E firing rate (Hz)", fontsize=theme.SIZE_LABEL)
    ax.set_ylabel("I firing rate (Hz)", fontsize=theme.SIZE_LABEL)
    ax.legend(
        fontsize=theme.SIZE_LEGEND,
        frameon=False,
        loc="upper right",
        title="condition · test acc",
        title_fontsize=theme.SIZE_LEGEND,
    )
    ax.margins(0.13)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path, formats=("svg", "pdf"))
    plt.close(fig)


def fig_training_curves(data, out_path: Path, run_id: str) -> None:
    """Validation accuracy/rates and reference-image contrast from retained epochs."""
    theme.apply()
    plt.rcParams["savefig.bbox"] = "standard"  # keep the saved 16:9 exact

    panels = [
        ("A", "Validation accuracy (%)", "acc", (0, 100)),
        ("B", "Validation E rate (Hz)", "rate_e", None),
        ("C", "Validation I rate (Hz)", "rate_i", None),
        ("D", "Reference contrast R", "contrast", (-0.02, 1.02)),
    ]
    xmax = data["last_epoch"]
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(6.9, 3.88),
        dpi=200,
        gridspec_kw={"hspace": 0.38, "wspace": 0.38},
    )
    axes = axes.ravel()
    for k, (letter, ylabel, key, ylim) in enumerate(panels):
        ax = axes[k]
        for cond in COND_ORDER:
            entry = data["trajectories"][cond]["panels"].get(key)
            if entry is None:
                continue
            ep, mean, lo, hi = (entry[k] for k in ("ep", "mean", "lo", "hi"))
            color = COND_COLOURS.get(cond, theme.INK_BLACK)
            frozen = cond == "frozen_ping"
            ax.fill_between(ep, lo, hi, color=color, alpha=0.13, lw=0)
            ax.plot(ep, mean, color=color, lw=2.0, ls="--" if frozen else "-")
        ax.set_ylabel(ylabel, fontsize=theme.SIZE_LABEL)
        if ylim is None:
            ax.set_ylim(bottom=0)
        else:
            ax.set_ylim(*ylim)
        ax.set_xlim(0, xmax)
        ax.tick_params(labelsize=theme.SIZE_TICK)
        if k >= 2:
            ax.set_xlabel("training epoch", fontsize=theme.SIZE_LABEL)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    # One legend entry per condition, mean-line style, in the accuracy panel.
    for cond in COND_ORDER:
        if cond in data["trajectories"]:
            axes[0].plot(
                [],
                [],
                color=COND_COLOURS.get(cond, theme.INK_BLACK),
                lw=2.0,
                ls="--" if cond == "frozen_ping" else "-",
                label=CONDITIONS[cond]["label"],
            )
    axes[0].legend(frameon=False, fontsize=theme.SIZE_LEGEND - 1, loc="lower right")
    theme.label_panels(axes)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.14, top=0.97)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Crop surrounding whitespace — this is a standalone publication figure, so
    # trim to content rather than holding the fixed 16:9 frame. save_figure takes
    # no kwargs, so the crop is applied via rcParams here.
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["savefig.pad_inches"] = 0.04
    save_figure(fig, out_path, formats=("svg", "pdf"))
    plt.close(fig)


def fig_phase_portrait(data, out_path: Path, run_id: str) -> None:
    """Unsmoothed validation E-rate/reference-contrast trajectories; frozen endpoint cluster."""
    theme.apply()
    plt.rcParams["savefig.bbox"] = "standard"

    from matplotlib.collections import LineCollection
    from matplotlib.colors import to_rgba

    fig, ax = plt.subplots(figsize=(5.6, 3.15), dpi=200)

    e_max = 0.0
    final_accs: dict[str, float] = {}

    for cond in COND_ORDER:
        entry = data["trajectories"][cond]["phase"]
        if entry is None:
            continue
        e_mean, p_mean, a_mean = (np.asarray(entry[k]) for k in ("e", "p", "a"))
        e_max = max(e_max, entry["max_e"])
        final_accs[cond] = entry["final_acc"]

        color = COND_COLOURS.get(cond, theme.INK_BLACK)
        marker = COND_MARKERS.get(cond, "o")
        frozen = cond == "frozen_ping"
        label = f"{CONDITIONS[cond]['label']}  ·  {final_accs[cond]:.1f}%"

        if frozen:
            # Show individual frozen endpoints and their mean, not a trajectory.
            ax.scatter(
                entry["final_e"],
                entry["final_p"],
                s=55,
                color=color,
                marker=marker,
                alpha=0.40,
                edgecolor="none",
                zorder=6,
            )
            ax.scatter(
                e_mean[-1],
                p_mean[-1],
                s=170,
                color=color,
                marker=marker,
                edgecolor="white",
                linewidths=1.3,
                zorder=10,
                label=label,
            )
        else:
            # Faded-to-saturated trajectory: alpha encodes epoch progress.
            points = np.array([e_mean, p_mean]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            alphas = np.linspace(0.22, 1.0, len(segments))
            seg_colors = [to_rgba(color, alpha=a) for a in alphas]
            lc = LineCollection(
                segments.tolist(), colors=seg_colors, linewidths=1.6, zorder=4
            )
            ax.add_collection(lc)
            # Start marker — hollow
            ax.scatter(
                e_mean[0],
                p_mean[0],
                s=70,
                facecolor="white",
                edgecolor=color,
                marker=marker,
                linewidths=1.4,
                zorder=10,
            )
            # End marker — filled
            ax.scatter(
                e_mean[-1],
                p_mean[-1],
                s=70,
                color=color,
                marker=marker,
                edgecolor="white",
                linewidths=0.9,
                zorder=10,
                label=label,
            )

    ax.set_xlabel("Validation E rate (Hz)", fontsize=theme.SIZE_LABEL)
    ax.set_ylabel("Reference contrast R", fontsize=theme.SIZE_LABEL)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(0, max(e_max * 1.08, 60.0))
    ax.tick_params(labelsize=theme.SIZE_TICK)
    ax.legend(
        fontsize=theme.SIZE_LEGEND - 1,
        frameon=False,
        loc="center right",
        title="condition · final validation acc",
        title_fontsize=theme.SIZE_LEGEND - 1,
    )
    ax.text(
        1.0,
        -0.16,
        "Trainable: ○ epoch 1   ● final epoch",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=theme.SIZE_CAPTION,
        color=theme.GREY_MID,
    )

    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["savefig.pad_inches"] = 0.04
    fig.tight_layout(pad=0.4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path, formats=("svg", "pdf"))
    plt.close(fig)


def fig_acc_rate_trajectory(data, out_path: Path, run_id: str) -> None:
    """Validation accuracy versus E rate, coloured by retained reference-image contrast."""
    theme.apply()
    plt.rcParams["savefig.bbox"] = "standard"

    from matplotlib.cm import ScalarMappable
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize

    fig, ax = plt.subplots(figsize=(5.6, 3.15), dpi=200)

    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=0.0, vmax=1.0)

    e_max = 0.0
    final_pings: dict[str, float] = {}

    for cond in COND_ORDER:
        entry = data["trajectories"][cond]["phase"]
        if entry is None:
            continue
        e_mean, p_mean, a_mean = (np.asarray(entry[k]) for k in ("e", "p", "a"))
        e_max = max(e_max, entry["max_e"])
        final_pings[cond] = p_mean[-1]

        marker = COND_MARKERS.get(cond, "o")
        # Per-segment pingness colour: average of the segment's two endpoint pingnesses.
        points = np.array([e_mean, a_mean]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        seg_pings = np.asarray(entry["segment_pings"])
        lc = LineCollection(
            segments.tolist(),
            cmap=cmap,
            norm=norm,
            array=seg_pings,
            linewidths=2.2,
            zorder=4,
        )
        ax.add_collection(lc)

        # Start (epoch 1) and end (final epoch) markers, coloured by their pingness.
        start_color = cmap(norm(p_mean[0]))
        end_color = cmap(norm(p_mean[-1]))
        ax.scatter(
            e_mean[0],
            a_mean[0],
            s=70,
            facecolor="white",
            edgecolor=start_color,
            marker=marker,
            linewidths=1.6,
            zorder=10,
        )
        label = f"{CONDITIONS[cond]['label']}  ·  final R {final_pings[cond]:.2f}"
        ax.scatter(
            e_mean[-1],
            a_mean[-1],
            s=85,
            color=end_color,
            marker=marker,
            edgecolor="white",
            linewidths=0.9,
            zorder=10,
            label=label,
        )

    # Colorbar for pingness — the third dimension.
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.045, aspect=22)
    cbar.set_label("Reference contrast R", fontsize=theme.SIZE_LABEL)
    cbar.ax.tick_params(labelsize=theme.SIZE_TICK)

    ax.set_xlabel("Validation E rate (Hz)", fontsize=theme.SIZE_LABEL)
    ax.set_ylabel("Validation accuracy (%)", fontsize=theme.SIZE_LABEL)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(60, 100)
    ax.set_xlim(0, max(e_max * 1.08, 60.0))
    ax.tick_params(labelsize=theme.SIZE_TICK)
    ax.legend(
        fontsize=theme.SIZE_LEGEND - 1,
        frameon=False,
        loc="lower right",
        title="condition · final reference contrast",
        title_fontsize=theme.SIZE_LEGEND - 1,
    )
    ax.text(
        1.0,
        -0.16,
        "○ epoch 1   ● final epoch",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=theme.SIZE_CAPTION,
        color=theme.GREY_MID,
    )

    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["savefig.pad_inches"] = 0.04
    fig.tight_layout(pad=0.4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path, formats=("svg", "pdf"))
    plt.close(fig)


def fig_training_summary(data, out_path: Path, run_id: str) -> None:
    """Seven-panel endpoint and recurrent-weight comparison from saved analysis."""
    import snnviz
    from matplotlib.patches import Patch

    conds = [
        "frozen_ping",
        "trainable_ping_init",
        "trainable_small_init",
        "trainable_zero_init",
    ]
    x = np.arange(4)
    theme.set_paper_mode(True)
    theme.apply()
    grid = snnviz.FigureGrid(
        rows=2, columns=1, bounds=(0.09, 0.16, 0.88, 0.70), row_gap=0.14
    )
    grid.place("top", row=0, column=0)
    grid.place("bottom", row=1, column=0)
    top = grid.subgrid("top", rows=1, columns=3, column_gap=0.075)
    bottom = grid.subgrid("bottom", rows=1, columns=4, column_gap=0.075)
    for i in range(3):
        top.place(str(i), row=0, column=i)
    for i in range(4):
        bottom.place(str(i), row=0, column=i)
    fig = grid.figure(figsize=(180 / 25.4, 4.8), dpi=240)
    theme.apply()
    axes = [top.add_axes(fig, str(i)) for i in range(3)] + [
        bottom.add_axes(fig, str(i)) for i in range(4)
    ]
    fig.legend(
        handles=[
            Patch(facecolor=theme.GREY_LIGHT, label="Weights before"),
            Patch(facecolor=theme.DEEP_RED, edgecolor="none", label="Weights after"),
        ],
        bbox_to_anchor=(0.09, 0.525),
        loc="upper left",
        ncol=2,
        frameon=False,
        borderaxespad=0,
        fontsize=theme.SIZE_LEGEND,
    )
    for ax, title, letter in zip(
        axes,
        [
            "Test accuracy",
            "E/I firing rates",
            "Lobe–trough contrast",
            "E→I non-zero",
            "I→E non-zero",
            "E→I mean",
            "I→E mean",
        ],
        "ABCDEFG",
    ):
        ax.set_title(
            title, fontsize=theme.SIZE_LABEL, fontweight="semibold", loc="left", pad=6
        )
        theme.label_panel(ax, letter)
        ax.set_xlim(-0.6, 3.6)
        ax.set_xticks(
            x, ["Frozen", "Std.", "10%", "Zero"], fontsize=theme.SIZE_ANNOTATION,
            rotation=25, ha="right"
        )
        ax.tick_params(axis="x", length=0, pad=4)
        ax.tick_params(
            axis="y", labelsize=theme.SIZE_TICK, direction="in", length=3, width=0.8
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.8)
        ax.spines["bottom"].set_linewidth(0.8)

    def bars(ax, key, positions, color, width=0.6, label=None):
        m = [data["outcomes"][c][key]["mean"] for c in conds]
        sem = [data["outcomes"][c][key]["sem"] for c in conds]
        ax.bar(
            positions,
            m,
            width=width,
            color=color,
            edgecolor="none",
            label=label,
            zorder=2,
        )
        ax.errorbar(
            positions,
            m,
            yerr=sem,
            fmt="none",
            ecolor=theme.INK_BLACK,
            elinewidth=0.8,
            capsize=2,
            capthick=0.8,
            zorder=4,
        )
        return m, sem

    m, se = bars(axes[0], "acc", x, theme.INK_BLACK)
    axes[0].set_ylim(0, 104)
    axes[0].set_yticks([0, 25, 50, 75, 100])
    axes[0].set_ylabel("Accuracy (%)")
    bars(axes[1], "e_rate_hz", x - 0.17, theme.INK_BLACK, width=0.30, label="E")
    bars(axes[1], "i_rate_hz", x + 0.17, theme.DEEP_RED, width=0.30, label="I")
    axes[1].set_ylim(0, 125)
    axes[1].set_yticks([0, 40, 80, 120])
    axes[1].set_ylabel("Rate (Hz)")
    axes[1].legend(loc="upper right", frameon=False, ncol=2, fontsize=theme.SIZE_LEGEND)
    m, se = bars(axes[2], "contrast", x, theme.INK_BLACK)
    axes[2].set_ylim(0, 1.13)
    axes[2].set_yticks([0, 0.5, 1])
    axes[2].set_ylabel("Contrast R")
    for ax, dr, kind in [
        (axes[3], "ei", "fraction"),
        (axes[4], "ie", "fraction"),
        (axes[5], "ei", "mean"),
        (axes[6], "ie", "mean"),
    ]:
        before = []
        after = []
        for cond in conds:
            s = data["weights"][cond]["weights"][dr]["stats"]
            if kind == "fraction":
                before.append(100 * (1 - s["init_zero_fraction"]))
                after.append(100 * (1 - s["trained_zero_fraction"]))
            else:
                before.append(1000 * s["init_mean"])
                after.append(1000 * s["trained_mean"])
        ax.bar(
            x, before, width=0.72, color=theme.GREY_LIGHT, edgecolor="none", zorder=1
        )
        ax.bar(
            x,
            after,
            width=0.43,
            facecolor=theme.DEEP_RED,
            edgecolor="none",
            linewidth=0,
            zorder=3,
        )
        if kind == "fraction":
            ax.set_ylim(0, 116)
            ax.set_yticks([0, 50, 100])
            ax.set_ylabel("Non-zero (%)")
        else:
            ax.set_ylim(0, 1.22 if dr == "ei" else 11.4)
            ax.set_ylabel("Mean weight (×10⁻³)")
            ax.set_yticks([0, 0.5, 1] if dr == "ei" else [0, 5, 10])
        for xx, v0, v1 in zip(x, before, after):
            if abs(v1 - v0) >= 0.05 * abs(v0) and not np.isclose(
                v0, v1, rtol=1e-7, atol=1e-12
            ):
                ax.annotate(
                    "",
                    xy=(xx, v1),
                    xytext=(xx, v0),
                    arrowprops=dict(
                        arrowstyle="->",
                        color=theme.INK_BLACK,
                        lw=0.7,
                        shrinkA=0,
                        shrinkB=0,
                        mutation_scale=5,
                    ),
                    zorder=6,
                )
    for ax in axes[3:]:
        ax.set_xticks(
            x,
            ["Frozen", "Std.", "10%", "Zero"],
            fontsize=theme.SIZE_ANNOTATION,
            rotation=25,
            ha="right",
        )
    save_figure(fig, out_path, formats=("svg", "pdf", "png"))
    plt.close(fig)
