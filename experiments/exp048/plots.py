"""Render saved measurements using the existing figure recipes."""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from experiments.helpers import theme
from experiments.helpers.figsave import save_figure
from experiments.helpers.stamp import stamp_figure
from PIL import Image, ImageDraw, ImageFont

from .recipe import (
    DT,
    INPUT_RATE_HZ,
    N_CLASSES,
    N_E,
    N_I,
    RASTER_N_E_PLOT,
    RASTER_N_I_PLOT,
    SEED,
    TAU_SWEEP_MS,
)


def label_retained_stream(source: Path, png_out: Path, pdf_out: Path) -> None:
    """Add canonical panel letters to a retained four-row stream rendering."""
    image = Image.open(source).convert("RGB")
    width, height = image.size
    font_path = Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans-Bold.ttf"
    font = ImageFont.truetype(str(font_path), max(14, round(width * 0.018)))
    draw = ImageDraw.Draw(image)
    for label, y_fraction in zip("ABCD", (0.055, 0.270, 0.600, 0.800), strict=True):
        x, y = round(width * 0.068), round(height * y_fraction)
        bounds = draw.textbbox((x, y), label, font=font)
        draw.rectangle(
            (bounds[0] - 4, bounds[1] - 2, bounds[2] + 4, bounds[3] + 2),
            fill="white",
        )
        draw.text((x, y), label, fill=theme.INK_BLACK, font=font)
    image.save(png_out, dpi=(150, 150))
    image.save(pdf_out, "PDF", resolution=150)


def plot_headline_stream(s: dict, out_path: Path, run_id: str) -> None:
    """4-panel headline figure for a 5-digit τ=50ms stream."""
    theme.apply()
    n_dig = s["n_digits"]
    tau_ms = s["tau_ms"]
    tau_steps = s["tau_steps"]
    T_stream_steps = s["T_stream_steps"]
    t_axis = np.arange(T_stream_steps) * DT  # ms
    seg_starts_ms = np.arange(n_dig) * tau_ms
    seg_ends_ms = (np.arange(n_dig) + 1) * tau_ms
    labels = s["labels"]
    seg_pred = s["pred_per_t"][np.arange(1, n_dig + 1) * tau_steps - 1]
    pixels = s["pixels"]
    spk_e = s["spk_e"]
    spk_i = s["spk_i"]
    probs = s["probs"]

    fig = plt.figure(figsize=(6.9, 5.33), dpi=150)
    gs = fig.add_gridspec(
        4,
        1,
        height_ratios=[0.9, 2.2, 1.2, 2.0],
        hspace=0.18,
    )

    # ── Panel A: digit thumbnails + class labels
    ax_a = fig.add_subplot(gs[0])
    ax_a.set_xlim(0, T_stream_steps * DT)
    ax_a.set_ylim(0, 1)
    ax_a.set_yticks([])
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)
    ax_a.spines["left"].set_visible(False)
    for d in range(n_dig):
        # Thumbnail glued to the segment band, scaled to ~tau_ms wide.
        x_lo = seg_starts_ms[d]
        x_hi = seg_ends_ms[d]
        img = pixels[d].reshape(28, 28)
        # Build a tiny axes inset for the digit image to avoid axes-coord
        # warping with axhspan.
        sub_w = (x_hi - x_lo) / (T_stream_steps * DT) * 0.88
        sub_l = ax_a.get_position().x0 + (
            (x_lo / (T_stream_steps * DT))
            * (ax_a.get_position().x1 - ax_a.get_position().x0)
        )
        # add_axes([l,b,w,h]) rect form is valid at runtime; matplotlib stub
        # overloads are too strict → library-stub false positive.
        sub = fig.add_axes(  # ty: ignore[no-matching-overload]
            [
                sub_l,
                ax_a.get_position().y0 + 0.005,
                sub_w,
                ax_a.get_position().height - 0.01,
            ]
        )
        sub.imshow(img, cmap="Greys", interpolation="nearest", aspect="auto")
        sub.set_xticks([])
        sub.set_yticks([])
        sub.set_title(
            f"true {labels[d]} · pred {int(seg_pred[d])}",
            fontsize=theme.SIZE_LABEL,
            color=(
                theme.INK_BLACK if int(seg_pred[d]) == labels[d] else theme.DEEP_RED
            ),
            pad=2,
        )
    ax_a.set_xticks([])
    ax_a.tick_params(axis="x", labelbottom=False)

    # ── Panel B: hidden E raster
    ax_b = fig.add_subplot(gs[1])
    rng = np.random.default_rng(SEED)
    e_idx = np.sort(rng.choice(N_E, RASTER_N_E_PLOT, replace=False))
    e_t, e_n = np.where(spk_e[:, e_idx])
    ax_b.scatter(
        t_axis[e_t],
        e_n,
        s=2.0,
        c=theme.INK_BLACK,
        marker="|",
        linewidths=0.4,
    )
    for seg in seg_starts_ms[1:]:
        ax_b.axvline(seg, color=theme.GREY_MID, lw=0.5, ls=":", alpha=0.7)
    ax_b.set_xlim(0, T_stream_steps * DT)
    ax_b.set_ylim(0, RASTER_N_E_PLOT)
    ax_b.set_yticks([0, RASTER_N_E_PLOT])
    ax_b.set_yticklabels(["0", f"{N_E}"])
    ax_b.set_ylabel("E cell", fontsize=theme.SIZE_LABEL)
    ax_b.tick_params(axis="x", labelbottom=False)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)

    # ── Panel C: hidden I raster
    ax_c = fig.add_subplot(gs[2])
    i_idx = np.sort(rng.choice(N_I, min(RASTER_N_I_PLOT, N_I), replace=False))
    i_t, i_n = np.where(spk_i[:, i_idx])
    ax_c.scatter(
        t_axis[i_t],
        i_n,
        s=2.0,
        c=theme.DEEP_RED,
        marker="|",
        linewidths=0.4,
    )
    for seg in seg_starts_ms[1:]:
        ax_c.axvline(seg, color=theme.GREY_MID, lw=0.5, ls=":", alpha=0.7)
    ax_c.set_xlim(0, T_stream_steps * DT)
    ax_c.set_ylim(0, len(i_idx))
    ax_c.set_yticks([0, len(i_idx)])
    ax_c.set_yticklabels(["0", f"{N_I}"])
    ax_c.set_ylabel("I cell", fontsize=theme.SIZE_LABEL)
    ax_c.tick_params(axis="x", labelbottom=False)
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)

    # ── Panel D: readout probabilities
    ax_d = fig.add_subplot(gs[3])
    # Plot every class in grey, then identify the true-class trace with the
    # single paper accent. Line weight, not colour alone, carries the emphasis.
    for c in range(N_CLASSES):
        ax_d.plot(
            t_axis,
            probs[:, c],
            color=theme.GREY_MID,
            lw=0.6,
            alpha=0.45,
        )
    # Heavy line: true-class trace per segment
    for d in range(n_dig):
        a = d * tau_steps
        b = (d + 1) * tau_steps
        c = labels[d]
        ax_d.plot(
            t_axis[a:b],
            probs[a:b, c],
            color=theme.DEEP_RED,
            lw=2.2,
        )
    for seg in seg_starts_ms[1:]:
        ax_d.axvline(seg, color=theme.GREY_MID, lw=0.5, ls=":", alpha=0.7)
    ax_d.axhline(0.5, color=theme.GREY_MID, lw=0.5, ls="--", alpha=0.6)
    ax_d.set_xlim(0, T_stream_steps * DT)
    ax_d.set_ylim(0, 1)
    ax_d.set_xlabel("time (ms)", fontsize=theme.SIZE_LABEL)
    ax_d.set_ylabel("readout p(class)", fontsize=theme.SIZE_LABEL)
    ax_d.spines["top"].set_visible(False)
    ax_d.spines["right"].set_visible(False)

    theme.label_panels((ax_a, ax_b, ax_c, ax_d))
    stamp_figure(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path, formats=("png", "pdf"))  # dense raster: PNG, not SVG
    plt.close(fig)


def plot_acc_vs_tau(
    rows: list[dict],
    out_path: Path,
    run_id: str,
) -> None:
    theme.apply()
    fig, ax = plt.subplots(figsize=(5.6, 3.15), dpi=150)
    constant = sorted(
        [r for r in rows if not r["rate_compensate"]],
        key=lambda r: r["tau_ms"],
    )
    compensated = sorted(
        [r for r in rows if r["rate_compensate"]],
        key=lambda r: r["tau_ms"],
    )
    if constant:
        ax.errorbar(
            [r["tau_ms"] for r in constant],
            [r["acc"] for r in constant],
            yerr=[r.get("acc_sem", 0.0) for r in constant],
            marker="o",
            color=theme.INK_BLACK,
            lw=1.5,
            capsize=4,
            label=f"constant input ({INPUT_RATE_HZ:g} Hz)",
        )
    if compensated:
        ax.errorbar(
            [r["tau_ms"] for r in compensated],
            [r["acc"] for r in compensated],
            yerr=[r.get("acc_sem", 0.0) for r in compensated],
            marker="s",
            color=theme.DEEP_RED,
            lw=1.5,
            capsize=4,
            label=r"rate-compensated ($25 \cdot 200/\tau$ Hz)",
        )
    ax.set_xlabel(r"Segment duration $\tau$ (ms)", fontsize=theme.SIZE_LABEL)
    ax.set_ylabel("Per-segment accuracy (%)", fontsize=theme.SIZE_LABEL)
    ax.set_ylim(0, 100)
    ax.axhline(10.0, color=theme.GREY_MID, lw=0.5, ls=":", alpha=0.6)
    # Annotate the gamma cycle ≈ 28 ms.
    ax.axvline(28.0, color=theme.AMBER, lw=0.7, ls="--", alpha=0.8)
    ax.text(
        28.0,
        92,
        " ≈ 1 gamma cycle",
        fontsize=theme.SIZE_ANNOTATION,
        color=theme.AMBER,
        va="top",
    )
    ax.set_xticks(TAU_SWEEP_MS)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=theme.SIZE_LEGEND, frameon=False, loc="lower right")
    fig.suptitle(
        "Streaming accuracy vs digit duration on trained PING",
        fontsize=theme.SIZE_TITLE,
    )
    fig.tight_layout()
    stamp_figure(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path, formats=("svg", "pdf"))
    plt.close(fig)


def plot_varying_headline_stream(s: dict, out_path: Path, run_id: str) -> None:
    """4-panel headline with per-segment (τ, rate) varying."""
    theme.apply()
    segments = s["segments"]
    n_dig = len(segments)
    segment_steps = s["segment_steps"]
    T_stream_steps = s["T_stream_steps"]
    t_axis = np.arange(T_stream_steps) * DT
    seg_starts_steps = np.concatenate([[0], np.cumsum(segment_steps)[:-1]])
    seg_starts_ms = seg_starts_steps * DT
    seg_ends_ms = (seg_starts_steps + np.array(segment_steps)) * DT
    labels = s["labels"]
    seg_pred = s["seg_preds"]
    pixels = s["pixels"]
    spk_e = s["spk_e"]
    spk_i = s["spk_i"]
    probs = s["probs"]
    T_ms = T_stream_steps * DT

    fig = plt.figure(figsize=(6.9, 5.33), dpi=150)
    gs = fig.add_gridspec(
        4,
        1,
        height_ratios=[1.35, 2.2, 1.2, 2.0],
        hspace=0.18,
    )

    # Panel A: digit thumbnails with per-segment (τ, rate) labels.
    ax_a = fig.add_subplot(gs[0])
    ax_a.set_xlim(0, T_ms)
    ax_a.set_ylim(0, 1)
    ax_a.set_yticks([])
    ax_a.set_xticks([])
    for sp in ("top", "right", "left", "bottom"):
        ax_a.spines[sp].set_visible(False)

    rates_all = [seg[1] for seg in segments]
    log_rmin = np.log(min(rates_all))
    log_rmax = np.log(max(rates_all))

    for d in range(n_dig):
        tau_ms, rate_hz = segments[d]
        x_lo = seg_starts_ms[d]
        x_hi = seg_ends_ms[d]
        img = pixels[d].reshape(28, 28)
        sub_w = (x_hi - x_lo) / T_ms * 0.88
        sub_l = ax_a.get_position().x0 + (
            (x_lo / T_ms) * (ax_a.get_position().x1 - ax_a.get_position().x0)
        )
        # add_axes([l,b,w,h]) rect form is valid at runtime; matplotlib stub
        # overloads are too strict → library-stub false positive.
        sub = fig.add_axes(  # ty: ignore[no-matching-overload]
            [
                sub_l,
                ax_a.get_position().y0 + 0.005,
                sub_w,
                ax_a.get_position().height - 0.02,
            ]
        )
        # Opacity ∈ [0.2, 1.0] (log-rate) so the weakest drive is faintly
        # visible and the strongest is bold — input rate becomes a visual cue.
        if log_rmax > log_rmin:
            alpha = 0.2 + 0.8 * (np.log(rate_hz) - log_rmin) / (log_rmax - log_rmin)
        else:
            alpha = 1.0
        sub.imshow(
            img,
            cmap="Greys",
            interpolation="nearest",
            aspect="auto",
            alpha=alpha,
        )
        sub.set_xticks([])
        sub.set_yticks([])
        ok_color = theme.INK_BLACK if seg_pred[d] == labels[d] else theme.DEEP_RED
        # (τ, rate) caption. A per-inset title is centred over an axes whose width
        # scales with the segment's *time span*, so short segments get titles wider
        # than their thumbnail that overprint neighbours. Draw it instead on ax_a at
        # the segment centre (x in data ms), single line, and stagger adjacent
        # segments across two rows so labels never collide however narrow a segment
        # is. τ first (the structural knob), rate second.
        x_c = 0.5 * (x_lo + x_hi)
        y_row = 1.02 if d % 2 == 0 else 1.15
        ax_a.text(
            x_c,
            y_row,
            f"{tau_ms:g} ms · {rate_hz:g} Hz",
            transform=ax_a.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=theme.SIZE_LABEL - 2,
            color=theme.MUTED,
            clip_on=False,
        )
        # Per-segment prediction badge inset into the thumbnail's top
        # — same colour scheme as the readout traces below.
        sub.text(
            0.05,
            0.95,
            f"{labels[d]}→{seg_pred[d]}",
            transform=sub.transAxes,
            ha="left",
            va="top",
            fontsize=theme.SIZE_LABEL,
            color="white",
            weight="bold",
            bbox=dict(
                facecolor=ok_color,
                edgecolor="none",
                boxstyle="round,pad=0.2",
                alpha=0.95,
            ),
        )

    # Panel B: hidden E raster.
    ax_b = fig.add_subplot(gs[1])
    rng = np.random.default_rng(SEED)
    e_idx = np.sort(rng.choice(N_E, RASTER_N_E_PLOT, replace=False))
    e_t, e_n = np.where(spk_e[:, e_idx])
    ax_b.scatter(
        t_axis[e_t],
        e_n,
        s=2.0,
        c=theme.INK_BLACK,
        marker="|",
        linewidths=0.4,
    )
    for seg in seg_starts_ms[1:]:
        ax_b.axvline(seg, color=theme.GREY_MID, lw=0.5, ls=":", alpha=0.7)
    ax_b.set_xlim(0, T_ms)
    ax_b.set_ylim(0, RASTER_N_E_PLOT)
    ax_b.set_yticks([0, RASTER_N_E_PLOT])
    ax_b.set_yticklabels(["0", f"{N_E}"])
    ax_b.set_ylabel("E cell", fontsize=theme.SIZE_LABEL)
    ax_b.tick_params(axis="x", labelbottom=False)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)

    # Panel C: hidden I raster.
    ax_c = fig.add_subplot(gs[2])
    i_idx = np.sort(rng.choice(N_I, min(RASTER_N_I_PLOT, N_I), replace=False))
    i_t, i_n = np.where(spk_i[:, i_idx])
    ax_c.scatter(
        t_axis[i_t],
        i_n,
        s=2.0,
        c=theme.DEEP_RED,
        marker="|",
        linewidths=0.4,
    )
    for seg in seg_starts_ms[1:]:
        ax_c.axvline(seg, color=theme.GREY_MID, lw=0.5, ls=":", alpha=0.7)
    ax_c.set_xlim(0, T_ms)
    ax_c.set_ylim(0, len(i_idx))
    ax_c.set_yticks([0, len(i_idx)])
    ax_c.set_yticklabels(["0", f"{N_I}"])
    ax_c.set_ylabel("I cell", fontsize=theme.SIZE_LABEL)
    ax_c.tick_params(axis="x", labelbottom=False)
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)

    # Panel D: readout probabilities.
    ax_d = fig.add_subplot(gs[3])
    for c in range(N_CLASSES):
        ax_d.plot(
            t_axis,
            probs[:, c],
            color=theme.GREY_MID,
            lw=0.6,
            alpha=0.45,
        )
    for d in range(n_dig):
        a = seg_starts_steps[d]
        b = a + segment_steps[d]
        c = labels[d]
        ax_d.plot(
            t_axis[a:b],
            probs[a:b, c],
            color=theme.DEEP_RED,
            lw=2.2,
        )
    for seg in seg_starts_ms[1:]:
        ax_d.axvline(seg, color=theme.GREY_MID, lw=0.5, ls=":", alpha=0.7)
    ax_d.axhline(0.5, color=theme.GREY_MID, lw=0.5, ls="--", alpha=0.6)
    ax_d.set_xlim(0, T_ms)
    ax_d.set_ylim(0, 1)
    ax_d.set_xlabel("time (ms)", fontsize=theme.SIZE_LABEL)
    ax_d.set_ylabel("readout p(class)", fontsize=theme.SIZE_LABEL)
    ax_d.spines["top"].set_visible(False)
    ax_d.spines["right"].set_visible(False)

    theme.label_panels((ax_a, ax_b, ax_c, ax_d))
    stamp_figure(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path, formats=("png", "pdf"))  # dense raster: PNG, not SVG
    plt.close(fig)


def plot_grid_and_rate(
    rows: list[dict],
    rate_rows: list[dict],
    out_path: Path,
    run_id: str,
) -> None:
    theme.apply()
    taus = sorted(set(r["tau_ms"] for r in rows))
    rates = sorted(set(r["input_rate_hz"] for r in rows))
    grid = np.zeros((len(rates), len(taus)), dtype=np.float32)
    for r in rows:
        i = rates.index(r["input_rate_hz"])
        j = taus.index(r["tau_ms"])
        grid[i, j] = r["acc"]

    fig, (ax, curve_ax) = plt.subplots(
        1,
        2,
        figsize=(8.0, 3.5),
        gridspec_kw={"width_ratios": (1.15, 1)},
    )
    im = ax.imshow(
        grid,
        origin="lower",
        aspect="auto",
        cmap="magma",
        vmin=0,
        vmax=100,
    )
    ax.set_xticks(range(len(taus)))
    ax.set_xticklabels([f"{t:g}" for t in taus])
    ax.set_yticks(range(len(rates)))
    ax.set_yticklabels([f"{r:g}" for r in rates])
    ax.set_xlabel(r"Segment duration $\tau$ (ms)", fontsize=theme.SIZE_LABEL)
    ax.set_ylabel("Input Poisson rate (Hz / channel)", fontsize=theme.SIZE_LABEL)
    for i in range(len(rates)):
        for j in range(len(taus)):
            ax.text(
                j,
                i,
                f"{grid[i, j]:.0f}",
                ha="center",
                va="center",
                fontsize=theme.SIZE_LABEL,
                color=("white" if grid[i, j] < 55 else theme.INK_BLACK),
            )
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Per-segment accuracy (%)", fontsize=theme.SIZE_LABEL)
    curve_rates = np.array([row["input_rate_hz"] for row in rate_rows])
    curve_acc = 100 * np.array([row["accuracy"] for row in rate_rows])
    curve_sem = 100 * np.array([row["accuracy_sem"] for row in rate_rows])
    curve_ax.plot(curve_rates, curve_acc, color=theme.INK_BLACK, lw=1.8)
    curve_ax.fill_between(
        curve_rates,
        curve_acc - curve_sem,
        curve_acc + curve_sem,
        color=theme.INK_BLACK,
        alpha=0.15,
        linewidth=0,
    )
    curve_ax.scatter(
        curve_rates, curve_acc, color=theme.INK_BLACK, marker="o", zorder=3
    )
    curve_ax.axhline(10, color=theme.DEEP_RED, ls="--", lw=1.2, label="chance (10%)")
    curve_ax.axvline(
        INPUT_RATE_HZ,
        color=theme.GREY_MID,
        ls=":",
        lw=1.2,
        label=f"trained rate ({INPUT_RATE_HZ:g} Hz)",
    )
    curve_ax.set(
        xlabel="Poisson encoding rate (Hz)",
        ylabel="P(correct) (%)",
        xlim=(0, max(curve_rates)),
        ylim=(0, 101),
    )
    curve_ax.legend(
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=1,
        fontsize=7,
        loc="lower right",
    )
    # Keep the complete linear-rate curve while making its informative
    # low-rate transition readable without a logarithmic axis.
    zoom = curve_ax.inset_axes([0.43, 0.30, 0.53, 0.48])
    low = curve_rates <= 10
    zoom.plot(curve_rates[low], curve_acc[low], color=theme.INK_BLACK, lw=1.2)
    zoom.fill_between(
        curve_rates[low],
        curve_acc[low] - curve_sem[low],
        curve_acc[low] + curve_sem[low],
        color=theme.INK_BLACK,
        alpha=0.15,
        linewidth=0,
    )
    zoom.scatter(
        curve_rates[low],
        curve_acc[low],
        color=theme.INK_BLACK,
        marker="o",
        s=12,
        zorder=3,
    )
    zoom.axhline(10, color=theme.DEEP_RED, ls="--", lw=0.8)
    zoom.set(xlim=(0, 10), ylim=(0, 101))
    zoom.set_xlabel("0–10 Hz detail", fontsize=7)
    zoom.tick_params(labelsize=6)
    theme.label_panels((ax, curve_ax))
    fig.tight_layout()
    stamp_figure(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Equal breathing room on all four sides (tight bbox + uniform pad),
    # so the plot sits centered in the exported image.
    with plt.rc_context({"savefig.pad_inches": 0.15}):
        save_figure(fig, out_path, formats=("png", "pdf"))
    plt.close(fig)
