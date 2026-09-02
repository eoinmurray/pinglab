"""Render saved arrays and summaries without running inference or aggregation."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from experiments.helpers import theme
from experiments.helpers.figsave import save_figure

from .recipe import EI_RASTER_N_E_PLOT, EI_RASTER_N_I_PLOT, MODELS


def plot_rate_rasters(samples: list[dict], out_path: Path, run_id: str) -> None:
    """One row per input-rate value; same E-over-I stacked layout as
    plot_ei_rasters so the two figures are visually comparable."""
    theme.apply()
    n = len(samples)
    n_e = EI_RASTER_N_E_PLOT
    n_i = EI_RASTER_N_I_PLOT
    gap = 6
    fig, axes = plt.subplots(
        n,
        1,
        figsize=(7.2, max(3.15, 0.62 * n + 0.7)),
        sharex=True,
        gridspec_kw={"hspace": 0.18},
    )
    if n == 1:
        axes = [axes]
    for i, (ax, s) in enumerate(zip(axes, samples)):
        T = s["e"].shape[0]
        t_axis = np.arange(T) * s["dt"]
        e_t, e_n = np.where(s["e"])
        i_t, i_n = np.where(s["i"])
        ax.scatter(
            t_axis[e_t],
            e_n,
            s=2.0,
            c=theme.INK_BLACK,
            marker="|",
            linewidths=0.4,
        )
        ax.scatter(
            t_axis[i_t],
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
        ax.set_xlim(0, s["t_ms"])
        i_rate_str = f"\nI = {s['i_rate_hz']:.1f} Hz" if "i_rate_hz" in s else ""
        ax.text(
            1.012,
            0.5,
            f"input = {s['spike_rate']:.1f} Hz\nE = {s['e_rate_hz']:.1f} Hz"
            + i_rate_str,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=theme.SIZE_ANNOTATION,
        )
        if i == 0:
            ax.set_title(
                f"Trained PING input-rate sweep — MNIST label {s['label']}\n"
                "E spikes black; I spikes red",
                fontsize=theme.SIZE_TITLE,
            )
        if i < n - 1:
            ax.tick_params(axis="x", labelbottom=False)
    axes[-1].set_xlabel("time (ms)")
    theme.label_panels(axes)
    fig.subplots_adjust(left=0.07, right=0.76, top=0.9, bottom=0.08)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path, formats=("png", "pdf"))  # dense raster: PNG, not SVG
    plt.close(fig)


def plot_fi_curve_uniform(
    rows: list[dict],
    out_path: Path,
    run_id: str,
    zoom_rows: list[dict] | None = None,
) -> None:
    """Two-panel f-I figure under spatially uniform Poisson input:
    COBA (E + I) on the left, PING (E + I) on the right. If `zoom_rows`
    is provided, a third panel below adds the 0-10 Hz zoom overlaying
    both models' E curves to expose the recruitment cliff."""
    theme.apply()
    if zoom_rows is None:
        fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.1))
        top_axes = list(axes)
    else:
        fig, axes = plt.subplots(2, 2, figsize=(5.6, 4.2))
        top_axes = list(axes[0])
    titles = {"ping": "PING (I-loop active)", "coba": "COBA (no I-loop)"}
    for ax, model in zip(top_axes, MODELS):
        msel = sorted(
            [r for r in rows if r["model"] == model],
            key=lambda r: r["input_rate_hz"],
        )
        xs = [r["input_rate_hz"] for r in msel]
        e_ys = [r["e_rate_hz"] for r in msel]
        i_ys = [r["i_rate_hz"] for r in msel]
        ax.plot(xs, e_ys, marker="o", color=theme.INK_BLACK, lw=1.5, label="E")
        ax.plot(xs, i_ys, marker="s", color=theme.DEEP_RED, lw=1.5, label="I")
        if model == "ping":
            ax.plot(
                xs,
                [r["e_plus_i_rate_hz"] for r in msel],
                marker="^",
                color=theme.AMBER,
                lw=1.5,
                ls="--",
                label="E + I (sum)",
            )
        ax.set_xlabel("Input Poisson rate (Hz, per channel)", fontsize=theme.SIZE_LABEL)
        ax.set_ylabel("Per-cell firing rate (Hz)", fontsize=theme.SIZE_LABEL)
        ax.set_title(titles[model], fontsize=theme.SIZE_TITLE)
        ax.legend(fontsize=theme.SIZE_LABEL, frameon=False, loc="upper left")

    # share the y-axis across the two panels so COBA's saturation and PING's
    # compression are read on one scale (the whole point of the comparison)
    top_max = max(
        (
            max(
                r["e_rate_hz"],
                r["i_rate_hz"],
                r["e_plus_i_rate_hz"] if r["model"] == "ping" else 0.0,
            )
            for r in rows
        ),
        default=1.0,
    )
    for ax in top_axes:
        ax.set_ylim(0, top_max * 1.05)

    if zoom_rows is not None:
        # Bottom row: zoom 0-10 Hz, one panel per model, same scheme
        # as the top row.
        for ax, model in zip(axes[1], MODELS):
            msel = sorted(
                [r for r in zoom_rows if r["model"] == model],
                key=lambda r: r["input_rate_hz"],
            )
            xs = [r["input_rate_hz"] for r in msel]
            e_ys = [r["e_rate_hz"] for r in msel]
            i_ys = [r["i_rate_hz"] for r in msel]
            ax.plot(xs, e_ys, color=theme.INK_BLACK, lw=1.5, label="E")
            ax.plot(xs, i_ys, color=theme.DEEP_RED, lw=1.5, label="I")
            if model == "ping":
                ax.plot(
                    xs,
                    [r["e_plus_i_rate_hz"] for r in msel],
                    color=theme.AMBER,
                    lw=1.5,
                    ls="--",
                    label="E + I (sum)",
                )
            ax.set_xlabel(
                "Input Poisson rate (Hz, per channel)", fontsize=theme.SIZE_LABEL
            )
            ax.set_ylabel("Per-cell firing rate (Hz)", fontsize=theme.SIZE_LABEL)
            ax.set_title(
                f"{titles[model]} — 0–10 Hz zoom",
                fontsize=theme.SIZE_TITLE,
            )
            ax.set_xlim(0, 10)
            ax.legend(fontsize=theme.SIZE_LABEL, frameon=False, loc="upper left")

    fig.suptitle(
        "Population f-I curves: trained PING and COBA, uniform Poisson input",
        fontsize=theme.SIZE_TITLE,
    )
    theme.label_panels(axes.flat if zoom_rows is not None else axes)
    fig.tight_layout()
    save_figure(fig, out_path)  # line/curve plot: SVG + PDF
    plt.close(fig)


def plot_fi_curve(samples: list[dict], out_path: Path, run_id: str) -> None:
    """f-I curve from the same data that plot_rate_rasters consumed.
    x-axis: input Poisson rate (Hz, per channel). y-axis: per-cell mean
    firing rate of E (black) and I (red) populations across the trial."""
    theme.apply()
    fig, ax = plt.subplots(figsize=(5.6, 3.15))
    xs = [s["spike_rate"] for s in samples]
    e_ys = [s["e_rate_hz"] for s in samples]
    i_ys = [s["i_rate_hz"] for s in samples]
    ax.plot(xs, e_ys, marker="o", color=theme.INK_BLACK, lw=1.5, label="E")
    ax.plot(xs, i_ys, marker="s", color=theme.DEEP_RED, lw=1.5, label="I")
    ax.set_xlabel("Input Poisson rate (Hz, per channel)", fontsize=theme.SIZE_LABEL)
    ax.set_ylabel("Per-cell firing rate (Hz)", fontsize=theme.SIZE_LABEL)
    ax.legend(fontsize=theme.SIZE_LABEL, frameon=False)
    fig.suptitle(
        f"Trained PING f-I curve (MNIST label {samples[0]['label']})",
        fontsize=theme.SIZE_TITLE,
    )
    fig.tight_layout()
    save_figure(fig, out_path)  # line/curve plot: SVG + PDF
    plt.close(fig)


def plot_ei_rasters(samples: list[dict], out_path: Path, run_id: str) -> None:
    """One row per ei value; I units stack over E units so the PING-style
    E-then-I cadence reads as alternating bursts when it appears."""
    theme.apply()
    n = len(samples)
    n_e = EI_RASTER_N_E_PLOT
    n_i = EI_RASTER_N_I_PLOT
    gap = 6
    fig, axes = plt.subplots(
        n,
        1,
        figsize=(5.6, 3.15),
        sharex=True,
        gridspec_kw={"hspace": 0.18},
    )
    if n == 1:
        axes = [axes]
    for i, (ax, s) in enumerate(zip(axes, samples)):
        T = s["e"].shape[0]
        t_axis = np.arange(T) * s["dt"]
        e_t, e_n = np.where(s["e"])
        i_t, i_n = np.where(s["i"])
        ax.scatter(
            t_axis[e_t],
            e_n,
            s=2.0,
            c=theme.INK_BLACK,
            marker="|",
            linewidths=0.4,
        )
        ax.scatter(
            t_axis[i_t],
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
        ax.set_xlim(0, s["t_ms"])
        ax.text(
            1.012,
            0.5,
            f"s = {s['ei_strength']:g}",
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=theme.SIZE_LABEL,
        )
        if i < n - 1:
            ax.tick_params(axis="x", labelbottom=False)
    axes[-1].set_xlabel("time (ms)")
    theme.label_panels(axes)
    fig.subplots_adjust(left=0.07, right=0.88, top=0.98, bottom=0.12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path, formats=("png", "pdf"))  # dense raster: PNG, not SVG
    plt.close(fig)


def _despine(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def fig_loop_transfer_compound(points, raster_lo, raster_hi, out_path, run_id):
    """Two illustrative rasters with saved mean and sample-SD transfer curves."""
    theme.apply()
    plt.rcParams["savefig.bbox"] = "standard"  # keep the saved 16:9 exact
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(6.9, 3.88))  # 16:9, full text width
    gs = GridSpec(
        2,
        2,
        figure=fig,
        height_ratios=[3.0, 2.6],
        hspace=0.5,
        wspace=0.22,
        top=0.93,
        bottom=0.1,
        left=0.07,
        right=0.96,
    )

    n_e, n_i, gap = EI_RASTER_N_E_PLOT, EI_RASTER_N_I_PLOT, 6
    raster_axes = []
    for col, s in enumerate((raster_lo, raster_hi)):
        ax = fig.add_subplot(gs[0, col])
        raster_axes.append(ax)
        T = s["e"].shape[0]
        t_axis = np.arange(T) * s["dt"]
        e_t, e_n = np.where(s["e"])
        i_t, i_n = np.where(s["i"])
        ax.scatter(
            t_axis[e_t], e_n, s=1.6, c=theme.INK_BLACK, marker="|", linewidths=0.4
        )
        ax.scatter(
            t_axis[i_t],
            i_n + n_e + gap,
            s=1.6,
            c=theme.DEEP_RED,
            marker="|",
            linewidths=0.4,
        )
        ax.set_ylim(-2, n_e + n_i + gap + 2)
        ax.set_yticks([n_e / 2, n_e + gap + n_i / 2])
        ax.set_yticklabels(["E", "I"])
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(0, s["t_ms"])
        ax.set_xlabel("time (ms)")
        tag = "loop off" if s["ei_strength"] == 0 else "loop enabled after training"
        ax.set_title(
            f"s = {s['ei_strength']:g}: {tag}", loc="left", fontweight="semibold"
        )
        _despine(ax)

    summary = points

    eis = np.asarray([p["ei_strength"] for p in summary])
    ax_r = fig.add_subplot(gs[1, 0])
    hid = np.asarray([p["hid_rate_hz"] for p in summary])
    inh = np.asarray([p["inh_rate_hz"] for p in summary])
    hid_sd = np.asarray([p["hid_rate_hz_sd"] for p in summary])
    inh_sd = np.asarray([p["inh_rate_hz_sd"] for p in summary])
    ax_r.plot(eis, hid, marker="o", ms=3, color=theme.INK_BLACK, label="E (hidden)")
    ax_r.plot(eis, inh, marker="s", ms=3, color=theme.DEEP_RED, label="I")
    ax_r.fill_between(
        eis, hid - hid_sd, hid + hid_sd, color=theme.INK_BLACK, alpha=0.15, linewidth=0
    )
    ax_r.fill_between(
        eis, inh - inh_sd, inh + inh_sd, color=theme.DEEP_RED, alpha=0.15, linewidth=0
    )
    ax_r.set_xlabel("inference E↔I strength s")
    ax_r.set_ylabel("rate (Hz)")
    ax_r.legend(fontsize=theme.SIZE_LEGEND, frameon=False)
    _despine(ax_r)

    ax_a = fig.add_subplot(gs[1, 1])
    accs = np.asarray([p["acc"] for p in summary])
    acc_sds = np.asarray([p["acc_sd"] for p in summary])
    base_acc = summary[0]["acc"]
    ax_a.axhline(
        base_acc,
        color=theme.LABEL,
        lw=1.0,
        ls="--",
        label=f"COBA baseline {base_acc:.0f}%",
    )
    ax_a.plot(eis, accs, marker="o", ms=3, color=theme.DEEP_RED, label="transfer")
    ax_a.fill_between(
        eis,
        accs - acc_sds,
        accs + acc_sds,
        color=theme.DEEP_RED,
        alpha=0.15,
        linewidth=0,
    )
    ax_a.set_ylim(0, 100)
    ax_a.set_xlabel("inference E↔I strength s")
    ax_a.set_ylabel("test accuracy (%)")
    ax_a.legend(fontsize=theme.SIZE_LEGEND, frameon=False, loc="lower left")
    _despine(ax_a)

    # Compound contains dense single-trial raster panels: rasterise as PNG, not SVG.
    theme.label_panels((*raster_axes, ax_r, ax_a))
    save_figure(fig, out_path, formats=("png", "pdf"))
    plt.close(fig)
