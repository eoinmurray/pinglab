"""Draw exp023's existing panels from retained analysis; no measurements or simulation."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from experiments.helpers import theme
from experiments.helpers.figsave import save_figure
from matplotlib.patches import FancyArrowPatch, Rectangle


def _despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_raster_compound(
    snaps: dict,
    fi: dict,
    out_path: Path,
    titles: dict,
    spectra: dict,
    peaks: dict,
    band_hz: tuple[float, float],
    include_arch: bool = False,
) -> None:
    """Super figure: COBA vs PING side by side.

    Each condition is a column-pair: a single raster (I stacked above E, no gap)
    spans the pair; below it the population-E Welch PSD sits next to the
    free-running f–I curve. Silent I populations are omitted from the raster.

    With include_arch=True a top row carries each column's architecture
    schematic (COBA / PING) directly above its plots.
    """
    theme.apply()
    plt.rcParams["savefig.bbox"] = "standard"  # keep the saved 16:9 exact
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(6.9, 3.88))  # 16:9, full print width
    if include_arch:
        # Nested gridspecs so the (small) schematic→plots gap is independent of
        # the (larger) raster→PSD gap that has to clear the time-axis label.
        outer = GridSpec(
            2,
            1,
            figure=fig,
            height_ratios=[2.5, 7.4],
            hspace=0.10,
            top=0.92,
            bottom=0.13,
            left=0.08,
            right=0.955,
        )
        arch_gs = outer[0].subgridspec(1, 2, wspace=0.5)
        plot_gs = outer[1].subgridspec(
            2,
            4,
            height_ratios=[4.4, 2.6],
            hspace=0.4,
            wspace=0.5,
        )
    else:
        arch_gs = None
        plot_gs = GridSpec(
            2,
            4,
            figure=fig,
            height_ratios=[4.4, 2.6],
            hspace=0.4,
            wspace=0.5,
            top=0.92,
            bottom=0.13,
            left=0.08,
            right=0.955,
        )

    arch_axes = []
    raster_axes = []
    lower_axes = []
    for col, cell in enumerate(("coba", "ping")):
        c0 = 2 * col
        s = snaps[cell]
        spk_e, spk_i, dt = s["spk_e"], s["spk_i"], s["dt"]
        T = spk_e.shape[0]
        t_ms = np.arange(T) * dt
        has_i = spk_i.size > 0 and spk_i.shape[0] == T and spk_i.any()

        if include_arch:
            # arch_gs is only built (non-None) in the include_arch branch above.
            assert arch_gs is not None
            ax_arch = fig.add_subplot(arch_gs[0, col])
            arch_axes.append(ax_arch)
            _draw_schematic(ax_arch, cell)
            ax_arch.set_title(titles[cell], loc="left", fontweight="semibold")

        ax_r = fig.add_subplot(plot_gs[0, c0 : c0 + 2])  # one raster, I above E
        ax_psd = fig.add_subplot(plot_gs[1, c0])  # PSD next to f–I
        ax_fi = fig.add_subplot(plot_gs[1, c0 + 1])
        raster_axes.append(ax_r)
        lower_axes.extend((ax_psd, ax_fi))

        # Combined raster: E (black) at the bottom, I (red) stacked directly
        # above it in the same axes — no vertical gap between the populations.
        n_e = spk_e.shape[1]
        e_idx, e_t = np.where(spk_e.T)
        ax_r.scatter(
            t_ms[e_t], e_idx, s=1.0, c=theme.INK_BLACK, marker="|", linewidths=0.5
        )
        if has_i:
            n_i = spk_i.shape[1]
            i_idx, i_t = np.where(spk_i.T)
            ax_r.scatter(
                t_ms[i_t],
                n_e + i_idx,
                s=1.0,
                c=theme.DEEP_RED,
                marker="|",
                linewidths=0.5,
            )
            ax_r.axhline(n_e, color=theme.GREY_MID, lw=0.6, alpha=0.6)
            total = n_e + n_i
            ax_r.set_yticks([n_e / 2, n_e + n_i / 2])
            ax_r.set_yticklabels(["E", "I"])
        else:
            total = n_e
            ax_r.set_yticks([n_e / 2])
            ax_r.set_yticklabels(["E"])
            ax_r.text(
                0.99,
                0.97,
                "no I activity",
                transform=ax_r.transAxes,
                ha="right",
                va="top",
                color=theme.GREY_MID,
                fontstyle="italic",
                fontsize=theme.SIZE_LABEL - 1,
                # Opaque backing so the note reads over the dense raster instead of
                # smearing into it like a ghost watermark.
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.75,
                ),
            )
        ax_r.set_ylim(0, total)
        ax_r.set_xlim(0, T * dt)
        ax_r.set_xlabel("time (ms)")
        if not include_arch:
            ax_r.set_title(titles[cell], loc="left", fontweight="semibold")
        _despine(ax_r)

        # Numerical measurements are retained analysis outputs, never recomputed here.
        freqs, psd = spectra[cell]["frequency_hz"], spectra[cell]["density"]
        f_peak = peaks[cell]
        band = (freqs >= band_hz[0]) & (freqs <= band_hz[1])
        ax_psd.plot(freqs[band], psd[band], color=theme.INK_BLACK, lw=1.4)
        ax_psd.set_xlim(band_hz)
        ax_psd.set_xlabel("frequency (Hz)")
        ax_psd.set_ylabel("E PSD (a.u.)")
        _despine(ax_psd)
        # Only the loop-on condition has a recurrent gamma peak to mark; the
        # loop-off control has scattered input-driven power, not a rhythm.
        if has_i and f_peak is not None:
            ax_psd.axvline(f_peak, color=theme.DEEP_RED, lw=0.9, ls="--", alpha=0.8)
            ax_psd.text(
                0.97,
                0.94,
                f"  $f_\\gamma$ = {f_peak:.1f} Hz",
                transform=ax_psd.transAxes,
                ha="right",
                va="top",
                fontsize=theme.SIZE_LABEL - 1,
                color=theme.DEEP_RED,
                fontweight="semibold",
            )
        elif has_i:
            # Loop on but no clean peak found. COBA (loop off) gets no label —
            # the flat spectrum speaks for itself.
            ax_psd.text(
                0.97,
                0.95,
                "no clear peak",
                transform=ax_psd.transAxes,
                ha="right",
                va="top",
                fontsize=theme.SIZE_LABEL - 1,
                color=theme.GREY_MID,
                fontstyle="italic",
            )

        # f–I curve (bottom-right of the pair). Per-condition y-scale: COBA runs
        # to its ~400+ Hz ceiling; PING is clamped by the loop, so it gets its
        # own smaller y-axis where the E-clamp and climbing I are legible (a
        # shared axis buried PING's curves at the bottom).
        f = fi[cell]
        ax_fi.plot(
            f["in"], f["e"], color=theme.INK_BLACK, marker="o", ms=3, lw=1.3, label="E"
        )
        ax_fi.plot(
            f["in"], f["i"], color=theme.DEEP_RED, marker="s", ms=3, lw=1.3, label="I"
        )
        cell_max = max(f["e"] + f["i"])
        ax_fi.set_ylim(0, max(cell_max * 1.08, 1.0))
        ax_fi.set_xlim(0, max(f["in"]))
        ax_fi.set_xlabel("input rate (Hz)")
        ax_fi.set_ylabel("rate (Hz)")
        _despine(ax_fi)
        if col == 1:
            ax_fi.legend(frameon=False, fontsize=theme.SIZE_LABEL - 2, loc="upper left")

    theme.label_panels((*arch_axes, *raster_axes, *lower_axes))
    save_figure(fig, out_path, formats=("png", "pdf"))  # dense raster: PNG, not SVG
    plt.close(fig)


def plot_traces(
    data: dict, selected: dict, biophysics: dict, out_path: Path, title: str
) -> None:
    """Draw analysis-selected voltage, conductance and current traces unchanged."""
    theme.apply()
    for population, index in (("e", selected["e_index"]), ("i", selected["i_index"])):
        if index is None:
            continue
        color = theme.INK_BLACK if population == "e" else theme.DEEP_RED
        time_ms = data["time_ms"]
        for panel in ("v", "g", "i"):
            fig, ax = plt.subplots(figsize=(4.0, 2.25))
            if panel == "v":
                ax.plot(
                    time_ms,
                    data[f"v_{population}"],
                    color=color,
                    lw=0.8,
                    label=f"V_{population.upper()}",
                )
                ax.axhline(
                    biophysics["threshold_mV"],
                    color=theme.FAINT,
                    lw=0.5,
                    ls="--",
                    label="V_th",
                )
                label, unit = "voltage", "V [mV]"
            elif panel == "g":
                ax.plot(
                    time_ms,
                    data[f"ge_{population}"],
                    color=theme.INK_BLACK,
                    lw=0.9,
                    label="g_E (exc)",
                )
                if population == "e" and selected["has_gi_e"]:
                    ax.plot(
                        time_ms,
                        data["gi_e"],
                        color=theme.DEEP_RED,
                        lw=0.9,
                        label="g_I (inh)",
                    )
                ax.axhline(
                    biophysics[f"g_L_{population.upper()}_uS"],
                    color=theme.FAINT,
                    lw=0.7,
                    ls=":",
                    label="g_L (leak)",
                )
                label, unit = "conductances", "g [µS]"
            else:
                ax.axhline(0, color=theme.FAINT, lw=0.5)
                ax.plot(
                    time_ms,
                    data[f"ie_{population}"],
                    color=theme.INK_BLACK,
                    lw=0.9,
                    label="I_E in",
                )
                if population == "e" and selected["has_gi_e"]:
                    ax.plot(
                        time_ms,
                        data["ii_e"],
                        color=theme.DEEP_RED,
                        lw=0.9,
                        label="I_I in",
                    )
                ax.plot(
                    time_ms,
                    data[f"il_{population}"],
                    color=theme.FAINT,
                    lw=0.7,
                    ls=":",
                    label="I_L in",
                )
                label, unit = "currents", "inward current [nA]"
            ax.set_xlim(0, time_ms[-1] + (time_ms[1] - time_ms[0]))
            ax.set_xlabel("time (ms)")
            ax.set_ylabel(unit)
            ax.set_title(
                f"{title} · {population.upper()} {label} (cell {index})",
                loc="left",
                fontsize=theme.SIZE_LABEL,
            )
            ax.legend(loc="upper right", fontsize=theme.SIZE_LEGEND)
            fig.tight_layout()
            save_figure(
                fig, out_path.with_name(f"{out_path.name}__{panel}_{population}")
            )
            plt.close(fig)


def _arch_box(ax, cx, cy, w, h, label, fontsize=15):
    """A black-edged population box with a centred monospace label."""
    ax.add_patch(
        Rectangle(
            (cx - w / 2, cy - h / 2),
            w,
            h,
            fill=False,
            edgecolor=theme.INK_BLACK,
            lw=1.8,
            zorder=3,
        )
    )
    # va="center" leaves the font's descender gap below the glyph, so a
    # capital (E/I, no descender) sits high and hugs the box top — exaggerated
    # when the panel isn't equal-aspect. Nudge down a few points (aspect-
    # independent) to optically centre the letter.
    ax.annotate(
        label,
        (cx, cy),
        textcoords="offset points",
        xytext=(0, -0.18 * fontsize),
        ha="center",
        va="center",
        fontsize=fontsize,
        color=theme.INK_BLACK,
        zorder=4,
    )


def _arch_arrow(ax, x0, y0, x1, y1):
    """A solid black arrow from (x0, y0) to (x1, y1)."""
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle="-|>",
            mutation_scale=14,
            lw=1.6,
            color=theme.INK_BLACK,
            shrinkA=0,
            shrinkB=0,
            zorder=3,
        )
    )


def _arch_label(ax, x, y, text, fontsize=12):
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=theme.INK_BLACK,
        zorder=4,
    )


def _draw_schematic(ax, kind: str) -> None:
    """Draw one architecture schematic (kind = 'coba' or 'ping') into ax.

    The frame fills the axes (16 × 9, no forced square) so the schematic is
    large; both kinds share the frame and box sizes so the COBA and PING
    panels read at the same scale. Weight labels sit clear of the boxes.
    """
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    bw, bh = 3.0, 2.6  # shared box size (taller: give E/I room off the top edge)
    bf, lf = 13, 10  # box-label / weight-label font sizes
    if kind == "coba":
        _arch_box(ax, 8.0, 4.5, bw, bh, "E", fontsize=bf)
        _arch_arrow(ax, 2.4, 4.5, 6.4, 4.5)  # input → E
        _arch_label(ax, 4.2, 5.7, "W_in", fontsize=lf)
        _arch_arrow(ax, 9.6, 4.5, 13.6, 4.5)  # E → output
        _arch_label(ax, 11.6, 5.7, "W_out", fontsize=lf)
    else:  # ping
        _arch_box(ax, 8.0, 6.4, bw, bh, "E", fontsize=bf)
        _arch_box(ax, 8.0, 2.0, bw, bh, "I", fontsize=bf)
        _arch_arrow(ax, 2.4, 6.4, 6.4, 6.4)  # input → E
        _arch_label(ax, 4.2, 7.6, "W_in", fontsize=lf)
        _arch_arrow(ax, 9.6, 6.4, 13.6, 6.4)  # E → output
        _arch_label(ax, 11.6, 7.6, "W_out", fontsize=lf)
        _arch_arrow(ax, 6.9, 5.1, 6.9, 3.3)  # E → I (down, left)
        _arch_label(ax, 5.0, 4.2, "W_ei", fontsize=lf)
        _arch_arrow(ax, 9.1, 3.3, 9.1, 5.1)  # I → E (up, right)
        _arch_label(ax, 11.0, 4.2, "W_ie", fontsize=lf)


def plot_architecture(out_path: Path) -> None:
    """Draw the COBA (left) and PING (right) schematics on one 16:9 figure."""
    theme.apply()
    plt.rcParams["savefig.bbox"] = "standard"  # keep the saved 16:9 exact

    fig, ax = plt.subplots(figsize=(6.9, 3.88))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.set_aspect("equal")  # 16:9 data range in a 16:9 frame, undistorted
    ax.axis("off")

    # ── COBA (left), centred on x = 4 ──────────────────────────────
    ax.text(
        4.0, 7.6, "COBA", ha="center", va="center", fontsize=14, color=theme.INK_BLACK
    )
    _arch_box(ax, 4.0, 5.0, 2.4, 1.7, "E")
    _arch_arrow(ax, 0.7, 5.0, 2.75, 5.0)  # input → E
    _arch_label(ax, 1.7, 5.5, "W_in")
    _arch_arrow(ax, 5.25, 5.0, 7.3, 5.0)  # E → output
    _arch_label(ax, 6.25, 5.5, "W_out")

    # ── PING (right), centred on x = 12 ────────────────────────────
    ax.text(
        12.0, 7.6, "PING", ha="center", va="center", fontsize=14, color=theme.INK_BLACK
    )
    _arch_box(ax, 12.0, 5.5, 2.6, 1.7, "E")
    _arch_box(ax, 12.0, 2.5, 2.6, 1.7, "I")
    _arch_arrow(ax, 8.5, 5.5, 10.65, 5.5)  # input → E
    _arch_label(ax, 9.55, 6.0, "W_in")
    _arch_arrow(ax, 13.3, 5.5, 15.4, 5.5)  # E → output
    _arch_label(ax, 14.35, 6.0, "W_out")
    _arch_arrow(ax, 11.3, 4.65, 11.3, 3.35)  # E → I (down, left side)
    _arch_label(ax, 10.25, 4.0, "W_ei")
    _arch_arrow(ax, 12.7, 3.35, 12.7, 4.65)  # I → E (up, right side)
    _arch_label(ax, 13.75, 4.0, "W_ie")

    save_figure(fig, out_path)  # schematic line art: SVG + PDF
    plt.close(fig)
