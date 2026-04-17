"""All matplotlib plotting code for the oscilloscope.

Contains panel layout, figure builders, draw functions, and profiler.
"""
from __future__ import annotations

import sys
import time as _time
from pathlib import Path
from contextlib import contextmanager

# Ensure src/pinglab/ is first on sys.path
_pkg_dir = str(Path(__file__).parent)
if _pkg_dir in sys.path:
    sys.path.remove(_pkg_dir)
sys.path.insert(0, _pkg_dir)

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import models as M
from metrics import (
    compute_pop_rate, compute_psd, find_fundamental_nondiff,
)


# =============================================================================
# Profiling
# =============================================================================

class _Profiler:
    """Lightweight accumulator for sim / render / encode timings."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.sim = 0.0
        self.render = 0.0
        self.encode = 0.0

    @contextmanager
    def track_sim(self):
        t0 = _time.monotonic()
        yield
        self.sim += _time.monotonic() - t0

    @contextmanager
    def track_render(self):
        t0 = _time.monotonic()
        yield
        self.render += _time.monotonic() - t0

    @contextmanager
    def track_encode(self):
        t0 = _time.monotonic()
        yield
        self.encode += _time.monotonic() - t0

    def report(self, n_frames=None):
        total = self.sim + self.render + self.encode
        if total == 0:
            return
        n = n_frames or 1
        def _bar(val, label, width=20):
            frac = val / total if total > 0 else 0
            filled = int(frac * width)
            bar = "\u2588" * filled + "\u2591" * (width - filled)
            avg = val / n
            return (f"  {label:>8s} {bar} {val:>6.1f}s  ({frac:>4.0%})"
                    f"  avg {avg*1000:>6.1f}ms/frame")

        print("  \u250c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2510")
        print(_bar(self.sim, "sim"))
        print(_bar(self.render, "render"))
        print(_bar(self.encode, "encode"))
        fps_str = ""
        if n_frames and total > 0:
            fps_str = f"  {n_frames / total:.1f} fps"
        print(f"  \u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500 total {total:>6.1f}s{fps_str} \u2500\u2500\u2518")


prof = _Profiler()


# =============================================================================
# Style
# =============================================================================

plt.rcParams.update({
    "font.family": "monospace",
    "font.size": 14,
    "font.weight": "normal",
    "axes.labelsize": 14,
    "axes.labelweight": "normal",
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "axes.titlesize": 14,
    "figure.titlesize": 18,
    "savefig.dpi": 200,
})

CLR = "#1a1a1a"
CLR_LIGHT = "#999999"
CLR_ACCENT = "#FF2020"


# =============================================================================
# Panel layout engine
# =============================================================================

# Each panel: (span, height_ratio)
# span: "full" = full width, "left"/"right" = half width
PANEL_CATALOG = {
    "header":        ("full",    0.25),
    "e_raster":      ("full",    3.0),
    "drive":         ("left",    0.8),
    "weights":       ("right",   0.8),
    "i_raster":      ("left",    0.6),
    "participation": ("right",   0.6),
    "output":        ("left",    0.6),
    "psd":           ("right",   0.6),
    "sweep":         ("sidebar", 0.8),
    "digit_image":   ("sidebar", 0.8),
    "acc_curve":     ("sidebar", 0.8),
    "grad_flow":     ("sidebar", 0.8),
    "rate_curve":    ("sidebar", 0.8),
}

# Named presets
LAYOUT_PRESETS = {
    "full": ["header", "e_raster",
             "drive", "weights", "i_raster", "participation",
             "output", "psd"],
    "video": ["header", "e_raster",
              "drive", "weights", "i_raster", "participation",
              "output", "psd", "sweep"],
    "dataset": ["header", "e_raster",
                "drive", "weights", "i_raster", "participation",
                "output", "psd", "digit_image"],
    "dataset_video": ["header", "e_raster",
                      "drive", "weights", "i_raster", "participation",
                      "output", "psd", "sweep", "digit_image"],
    "train": ["header", "e_raster",
              "drive", "weights", "i_raster", "participation",
              "output", "psd",
              "digit_image", "acc_curve", "grad_flow", "rate_curve"],
    "compact": ["header", "e_raster",
                "i_raster", "psd"],
    "minimal": ["header", "e_raster"],
}

ACTIVE_PANELS = list(LAYOUT_PRESETS["full"])


# =============================================================================
# Raster helper
# =============================================================================

RASTER_MODE = "scatter"  # "scatter" or "imshow"


def plot_raster(ax, spikes, color, n_neurons, dt, x_max=None):
    """Plot spike raster using scatter (crisp) or imshow (fast)."""
    if RASTER_MODE == "imshow":
        from matplotlib.colors import LinearSegmentedColormap
        from scipy.ndimage import maximum_filter1d
        img = spikes.T.astype(np.float32)
        spread = max(1, int(0.8 / dt))
        if spread > 1:
            img = maximum_filter1d(img, size=spread, axis=1)
        cmap = LinearSegmentedColormap.from_list("raster", ["white", color])
        vis_ms = x_max if x_max is not None else len(spikes) * dt
        ax.imshow(img, aspect="auto", origin="lower", cmap=cmap,
                  vmin=0, vmax=1, interpolation="nearest",
                  extent=[0, vis_ms, -0.5, n_neurons - 0.5])
    else:
        t_ms = np.arange(len(spikes)) * dt
        dot_size = max(12, min(35, 25 * (dt / 0.1)))
        step = max(1, n_neurons // 256)
        for n in range(0, n_neurons, step):
            times = t_ms[spikes[:, n] > 0]
            ax.scatter(times, np.full_like(times, n), s=dot_size, c=color,
                       marker="|", linewidths=0.5)
    if x_max is not None:
        ax.set_xlim(0, x_max)
    ax.set_ylim(-0.5, n_neurons - 0.5)
    ax.tick_params(labelsize=13)
    if x_max is not None:
        ax.set_xlim(0, x_max)
    ax.set_ylim(-0.5, n_neurons - 0.5)
    ax.tick_params(labelsize=13)


# =============================================================================
# Axes helpers
# =============================================================================

def _clear_axes(axes):
    """Clear and restyle all axes (handles lists of axes too)."""
    for ax in axes:
        if ax is None:
            continue
        if isinstance(ax, list):
            for a in ax:
                a.clear()
                for spine in a.spines.values():
                    spine.set_linewidth(2)
                    spine.set_color("black")
                    spine.set_visible(True)
            continue
        ax.clear()
        for spine in ax.spines.values():
            spine.set_linewidth(2)
            spine.set_color("black")
            spine.set_visible(True)


def _style_all_axes(axes):
    """Apply shared tick styling to all non-list axes."""
    for ax in axes:
        if ax is None or isinstance(ax, list):
            continue
        ax.tick_params(labelsize=13, colors=CLR, length=4, width=1)


# =============================================================================
# Figure builders
# =============================================================================

def make_fig(panels=None):
    """Build a figure with the requested panels.

    Supports full/left/center/right spans across a 3-column grid.
    Returns (fig, panel_dict) where panel_dict maps panel names to axes.
    """
    panels = panels or ACTIVE_PANELS

    # Sidebar panels are placed via a sub-gridspec in column 2; exclude
    # them from main-row building so they don't create empty rows.
    sidebar_panels = [(name, "sidebar") for name in panels
                      if name in PANEL_CATALOG
                      and PANEL_CATALOG[name][0] == "sidebar"]

    # Group panels into rows: full-width panels get their own row,
    # left/center/right panels accumulate into shared rows.
    rows = []         # list of (height, [(name, span), ...])
    pending = []      # accumulate left/center/right panels
    pending_h = 0

    for name in panels:
        if name not in PANEL_CATALOG:
            continue
        span, height = PANEL_CATALOG[name]

        if span == "sidebar":
            continue  # handled separately below

        if span == "full":
            # Flush any pending partial row
            if pending:
                rows.append((pending_h, list(pending)))
                pending = []
                pending_h = 0
            rows.append((height, [(name, "full")]))
        else:
            pending.append((name, span))
            pending_h = max(pending_h, height)
            # Flush when we hit 3 panels or right panel
            if span == "right" or len(pending) >= 3:
                rows.append((pending_h, list(pending)))
                pending = []
                pending_h = 0

    # Flush any remaining partial row
    if pending:
        rows.append((pending_h, list(pending)))

    n_rows = len(rows)
    if n_rows == 0:
        fig = plt.figure(figsize=(22, 4))
        return fig, {}

    height_ratios = [r[0] for r in rows]

    fig_w = 22
    fig_h = fig_w * 9 / 16  # 16:9 aspect ratio
    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.subplots_adjust(left=0.05, right=0.97, top=0.97, bottom=0.04)

    has_sidebar = len(sidebar_panels) > 0

    if has_sidebar:
        n_cols = 3
        width_ratios = [1, 1, 0.8]
    else:
        n_cols = 2
        width_ratios = [1, 1]

    gs = fig.add_gridspec(n_rows, n_cols, height_ratios=height_ratios,
                          hspace=0.3, wspace=0.12, width_ratios=width_ratios)

    panel_dict = {}
    col_map = {"left": 0, "center": 1, "right": 2}
    for row_idx, (_, items) in enumerate(rows):
        for name, span in items:
            if span == "full":
                if has_sidebar:
                    ax = fig.add_subplot(gs[row_idx, 0:2])
                else:
                    ax = fig.add_subplot(gs[row_idx, :])
            elif span == "left":
                ax = fig.add_subplot(gs[row_idx, 0])
            elif span == "right":
                ax = fig.add_subplot(gs[row_idx, 1])
            elif span == "sidebar":
                ax = fig.add_subplot(gs[row_idx, 2])
            else:
                col = col_map.get(span, 0)
                ax = fig.add_subplot(gs[row_idx, col])

            if name == "weights":
                # Weights sub-axes — 8 slots to accommodate multi-layer models
                ax.remove()
                gs_w = gs[row_idx, 1].subgridspec(1, 8, wspace=0.3)
                ax = [fig.add_subplot(gs_w[0, i]) for i in range(8)]

            if name in ("header", "progress"):
                if isinstance(ax, list):
                    for a in ax:
                        a.axis("off")
                else:
                    ax.axis("off")

            panel_dict[name] = ax

    # Place sidebar panels in column 2 using a sub-gridspec.
    # Match each sidebar slot to a main content row below the header so
    # the sidebars align visually and no empty spacer is left at the bottom.
    if has_sidebar:
        n_sidebar = len(sidebar_panels)
        main_heights = height_ratios[1:]  # skip header row
        if len(main_heights) >= n_sidebar:
            sidebar_h = list(main_heights[:n_sidebar])
            trailing = sum(main_heights[n_sidebar:])
            if trailing > 0:
                sidebar_h.append(trailing)
        else:
            sidebar_h = [1] * n_sidebar
        gs_sidebar = gs[1:, 2].subgridspec(
            len(sidebar_h), 1, height_ratios=sidebar_h, hspace=0.3)
        for i, (name, _) in enumerate(sidebar_panels):
            ax = fig.add_subplot(gs_sidebar[i, 0])
            panel_dict[name] = ax

    return fig, panel_dict


def make_transient_fig(layout=None):
    """Create the transient sweep figure layout (backward compat wrapper)."""
    panels = list(LAYOUT_PRESETS[layout]) if layout else None
    fig, pd = make_fig(panels)
    # Build the old-style axes list for draw_transient_frame
    axes = [
        None,                           # 0: unused
        pd.get("drive"),                # 1: conductance heatmap
        pd.get("e_raster"),             # 2: E raster
        pd.get("i_raster"),             # 3: I raster
        pd.get("participation"),        # 4: participation
        pd.get("psd"),                  # 5: PSD
        pd.get("header"),               # 6: header
        pd.get("output"),               # 7: output raster
        pd.get("progress"),             # 8: progress bar
        pd.get("weights"),              # 9: weight histograms
        pd.get("sweep"),                # 10: threshold/sweep panel
        pd.get("digit_image"),          # 11: digit image sidebar
        pd.get("acc_curve"),            # 12: accuracy curve sidebar
        pd.get("grad_flow"),            # 13: per-layer grad flow sidebar
        pd.get("rate_curve"),           # 14: E/I rate curve sidebar
    ]
    return fig, axes


# =============================================================================
# Draw functions
# =============================================================================

_SPINNER = "|/-\\"


def draw_transient_frame(axes, ratio, spk_e, spk_i, ext_g, dt, title=None,
                         spk_o=None, sweep_var=None, sweep_range=None,
                         sweep_progress=None, weights=None, model_name="ping",
                         t_e_async=0.0006,
                         sweep_levels=None, sweep_frame_idx=None,
                         n_e=1024, n_i=256, n_h1=None,
                         step_on_ms=200.0, step_off_ms=300.0, burn_in_ms=100.0,
                         w_ei=(0.5, 0.05), w_ie=(1.0, 0.1),
                         is_coba=None, digit_image=None,
                         acc=None, loss=None, grad_ratios=None, lr=None,
                         total_epochs=None, input_rate=None,
                         spk_h1=None):
    """Draw one transient PING frame onto the given axes.

    axes is a list where entries may be None (panel not present).
    """
    if is_coba is None:
        from config import IS_COBA
        is_coba = model_name in IS_COBA

    _clear_axes(axes)
    _title_kw = dict(fontsize=13, fontweight="bold", loc="left", pad=4, color=CLR)
    vis_ms = len(spk_e) * dt  # derive from actual data, not global

    # -- Input drive panel --
    if axes[1] is not None:
        is_spikes = set(np.unique(ext_g)).issubset({0, 0.0, 1, 1.0})
        if is_spikes:
            plot_raster(axes[1], ext_g, CLR, ext_g.shape[1], dt, vis_ms)
            axes[1].set_title("Input Spikes", **_title_kw)
        else:
            axes[1].imshow(ext_g.T, aspect="auto", origin="lower", cmap="binary",
                           extent=[0, vis_ms, 0, ext_g.shape[1]],
                           interpolation="nearest")
            axes[1].set_title("Input Drive Conductance", **_title_kw)
        axes[1].set_ylabel("")
        axes[1].set_xlim(0, vis_ms)
        axes[1].set_xticklabels([])

    # -- E raster --
    if axes[2] is not None:
        plot_raster(axes[2], spk_e, CLR, n_e, dt, vis_ms)
        axes[2].set_title("E Neurons", **_title_kw)
        axes[2].set_ylabel("")
        axes[2].set_xticklabels([])

    # -- I raster (or H1 raster for multi-layer without E-I) --
    if axes[3] is not None:
        if spk_i is not None:
            plot_raster(axes[3], spk_i, CLR_ACCENT, n_i, dt, vis_ms)
            axes[3].set_title("I Neurons", **_title_kw)
        elif spk_h1 is not None:
            _n_h1 = n_h1 if n_h1 is not None else spk_h1.shape[1]
            plot_raster(axes[3], spk_h1, CLR, _n_h1, dt, vis_ms)
            axes[3].set_title("H1 Neurons", **_title_kw)
        axes[3].set_ylabel("")
        axes[3].set_xticklabels([])

    # -- Threshold panel / sweep progress --
    if len(axes) > 10 and axes[10] is not None:
        if sweep_levels is not None and sweep_frame_idx is not None:
            cur = sweep_levels[sweep_frame_idx]
            sweep_title = (f"Sweep {sweep_var}  "
                           f"{sweep_levels[0]:.3g} → {sweep_levels[-1]:.3g}  "
                           f"cur {cur:.3g}")
            _draw_sweep_progress(axes[10], sweep_levels, sweep_frame_idx,
                                 sweep_title, _title_kw)
        else:
            _draw_threshold_panel(axes[10], ext_g, spk_i, dt, _title_kw,
                                  n_e=n_e, w_ie=w_ie)

    # -- Participation --
    if axes[4] is not None:
        t_bins, rate_hz = compute_pop_rate(spk_e, n_e, dt)
        bin_ms = 2.0
        max_rate = 1000.0 / bin_ms
        participation = rate_hz / max_rate
        axes[4].plot(t_bins, participation, color=CLR, linewidth=0.8)
        axes[4].set_title("E Population Rate", **_title_kw)
        axes[4].set_ylabel("")
        axes[4].set_ylim(0, 0.5)
        axes[4].set_xlim(0, vis_ms)
        axes[4].set_xlabel("Time (ms)", color=CLR)

    # -- PSD --
    freqs, psd = compute_psd(spk_e, n_e, dt,
                             step_on_ms=step_on_ms, step_off_ms=step_off_ms,
                             burn_in_ms=burn_in_ms)
    if axes[5] is not None:
        mask = (freqs > 5) & (freqs < 200)
        has_data = mask.any() and len(psd[mask]) > 0
        if has_data:
            axes[5].plot(freqs[mask], psd[mask], color=CLR, linewidth=1)
            peak_idx = np.argmax(psd[mask])
            peak_f = freqs[mask][peak_idx]
            if psd[mask][peak_idx] > 0.3:
                axes[5].axvline(peak_f, color=CLR_ACCENT, linewidth=1.5, alpha=0.8)
                axes[5].text(peak_f + 3, 0.9, f"{peak_f:.0f} Hz",
                             color=CLR_ACCENT, fontsize=14, fontweight="bold")
        axes[5].set_title("PSD", **_title_kw)
        axes[5].set_ylabel("")
        axes[5].set_xlabel("Frequency (Hz)", color=CLR)
        axes[5].set_xlim(5, 200)
        axes[5].set_ylim(0, 1.1)

    _style_all_axes(axes)

    # -- Firing rates --
    t_sec = len(spk_e) * dt / 1000.0
    rate_e_hz = spk_e.sum() / (n_e * t_sec)
    rate_i_hz = spk_i.sum() / (n_i * t_sec) if spk_i is not None else 0

    # -- Output: horizontal bar chart of final logits (one bar per class) --
    if len(axes) > 7 and axes[7] is not None:
        ax_o = axes[7]
        if spk_o is not None and spk_o.shape[1] > 0:
            final_logits = spk_o[-1]
            if final_logits.ndim == 2:
                final_logits = final_logits[0]
            n_classes = len(final_logits)
            pred = int(np.argmax(final_logits))
            colors = [CLR_ACCENT if i == pred else CLR for i in range(n_classes)]
            ax_o.barh(range(n_classes), final_logits, color=colors,
                      edgecolor="white", linewidth=0.3)
            ax_o.set_yticks(range(n_classes))
            ax_o.set_yticklabels([str(i) for i in range(n_classes)],
                                 fontsize=8, color=CLR)
            ax_o.invert_yaxis()
            for spine_side in ("top", "right"):
                ax_o.spines[spine_side].set_visible(False)
        ax_o.set_title("Output (logits)", **_title_kw)
        ax_o.set_xlabel("", color=CLR)
        ax_o.tick_params(labelsize=8, colors=CLR, length=3)

    # -- Weight histograms --
    if len(axes) > 9 and axes[9] is not None and weights is not None:
        _draw_weight_histograms(axes[9], weights)

    # -- Fundamental frequency (PING only, overdrive window) --
    f0 = 0.0
    if model_name == "ping":
        f0 = find_fundamental_nondiff(psd, freqs)

    # -- Metrics for header line 2 --
    ie_ratio = rate_i_hz / max(rate_e_hz, 1e-9) if spk_i is not None else 0.0
    per_neuron_counts = spk_e.sum(axis=0)
    active_frac = float((per_neuron_counts > 0).sum()) / n_e
    t_sec_m = len(spk_e) * dt / 1000.0
    per_neuron_rates = per_neuron_counts / t_sec_m
    active_rates = per_neuron_rates[per_neuron_rates > 0]
    rate_cv = (float(active_rates.std() / max(active_rates.mean(), 1e-9))
               if len(active_rates) > 1 else 0.0)
    bin_steps_m = max(1, int(2.0 / dt))
    n_bins_m = len(spk_e) // bin_steps_m
    if n_bins_m > 1:
        pop_counts_m = np.array([spk_e[i * bin_steps_m:(i + 1) * bin_steps_m].sum()
                                 for i in range(n_bins_m)])
        pop_cv = float(pop_counts_m.std() / max(pop_counts_m.mean(), 1e-9))
    else:
        pop_cv = 0.0

    # -- Header --
    if len(axes) > 6 and axes[6] is not None:
        _draw_header(axes[6], title, ratio, dt, rate_e_hz, rate_i_hz,
                     spk_i is not None, model_name, t_e_async, f0=f0,
                     frame_idx=sweep_frame_idx,
                     pop_cv=pop_cv, ie_ratio=ie_ratio, active_frac=active_frac,
                     rate_cv=rate_cv, n_e=n_e, is_coba=is_coba,
                     input_rate=input_rate)

    # -- Progress bar --
    if len(axes) > 8 and axes[8] is not None:
        _draw_progress_bar(axes[8], ratio, sweep_var, sweep_range, sweep_progress)

    # -- Accuracy curve sidebar (with loss on twin axis, lr as third line) --
    if len(axes) > 12 and axes[12] is not None and acc is not None:
        _draw_acc_curve(axes[12], acc, total_epochs or 100, loss=loss, lr=lr)

    # -- Per-layer grad/weight ratio sidebar --
    if len(axes) > 13 and axes[13] is not None and grad_ratios is not None:
        _draw_grad_flow(axes[13], grad_ratios, total_epochs or 100)

    # -- E/I rate trace sidebar --
    if len(axes) > 14 and axes[14] is not None:
        _draw_rate_curve(axes[14], rate_e_hz, rate_i_hz,
                         spk_i is not None, total_epochs or 100)

    # -- Digit image sidebar --
    if len(axes) > 11 and axes[11] is not None:
        _draw_digit_image(axes[11], digit_image)




def _draw_threshold_panel(ax_th, ext_g, spk_i, dt, title_kw,
                          n_e=1024, w_ie=(1.0, 0.1)):
    """Draw the drive-vs-threshold panel."""
    t_ms_arr = np.arange(len(ext_g)) * dt

    decay_ampa = np.exp(-dt / M.tau_ampa)
    mean_drive = ext_g.mean(axis=1)
    drive_ss = mean_drive / (1 - decay_ampa)

    g_thresh = M.g_L_E * (M.V_th - M.E_L) / (M.E_e - M.V_th)

    if spk_i is not None:
        decay_gaba = np.exp(-dt / M.tau_gaba)
        gi_mean = np.zeros(len(spk_i))
        gi_acc = 0.0
        W_ie_mean = w_ie[0]
        for t in range(len(spk_i)):
            gi_acc = gi_acc * decay_gaba + spk_i[t].sum() / n_e * W_ie_mean
            gi_mean[t] = gi_acc
        inh_penalty = gi_mean * abs(M.E_i - M.V_th) / (M.E_e - M.V_th)
        g_thresh_eff = g_thresh + inh_penalty
    else:
        g_thresh_eff = np.full(len(ext_g), g_thresh)

    ax_th.plot(t_ms_arr, drive_ss, color=CLR, linewidth=1, label="Drive $g_{ss}$")
    ax_th.axhline(g_thresh, color=CLR, linewidth=1, linestyle="--", alpha=0.4,
                  label="Bare threshold")
    ax_th.plot(t_ms_arr, g_thresh_eff, color=CLR_ACCENT, linewidth=0.8, alpha=0.7,
               label="Eff. threshold")
    ax_th.fill_between(t_ms_arr, g_thresh_eff, drive_ss,
                       where=drive_ss > g_thresh_eff, color=CLR, alpha=0.08)
    ax_th.set_xlim(0, len(ext_g) * dt)
    ax_th.set_title("Drive vs Threshold", **title_kw)
    ax_th.set_ylabel("")
    ax_th.set_xticklabels([])
    ax_th.legend(fontsize=9, loc="upper right", frameon=False)


def _draw_sweep_progress(ax, values, frame_idx, title, title_kw):
    """Draw a sweep progress indicator: faint full line, solid completed, red dot."""
    ax.clear()
    frames = np.arange(len(values))
    ax.plot(frames, values, color=CLR, linewidth=1.5, alpha=0.2)
    if frame_idx > 0:
        ax.plot(frames[:frame_idx + 1], values[:frame_idx + 1],
                color=CLR, linewidth=1.5)
    ax.plot(frame_idx, values[frame_idx], "o",
            color=CLR_ACCENT, markersize=6, zorder=5)
    ax.set_xlim(0, len(values) - 1)
    ax.set_ylim(0, max(values) * 1.05)
    ax.set_title(title, **title_kw)
    ax.set_xlabel("frame", fontsize=11, color=CLR)
    ax.set_ylabel("value", fontsize=11, color=CLR)
    ax.tick_params(labelsize=10, colors=CLR)


def _draw_acc_curve(ax, acc, total_epochs, loss=None, lr=None):
    """Draw accuracy curve with optional loss on twin axis and LR (% of max)."""
    if not hasattr(ax, '_acc_history'):
        ax._acc_history = []
        ax._loss_history = []
        ax._lr_history = []
        ax._twin = ax.twinx()
    ax._acc_history.append(acc)
    if loss is not None:
        ax._loss_history.append(loss)
    if lr is not None:
        ax._lr_history.append(lr)

    ax.clear()
    ax._twin.clear()

    epochs = list(range(len(ax._acc_history)))
    ax.plot(epochs, ax._acc_history, color=CLR, linewidth=1.5, label="acc")
    ax.plot(len(epochs) - 1, acc, "o",
            color=CLR, markersize=6, zorder=5)

    # LR as dashed grey on left axis, normalized to % of max LR seen so far.
    # This makes scheduler step-downs visually obvious without a third axis.
    if ax._lr_history:
        lr_max = max(ax._lr_history)
        lr_pct = [100.0 * v / lr_max for v in ax._lr_history]
        lr_epochs = list(range(len(lr_pct)))
        ax.plot(lr_epochs, lr_pct, color=CLR_LIGHT, linewidth=1.2,
                linestyle="--", alpha=0.85, label="lr%")

    ax.set_xlim(0, max(total_epochs - 1, 1))
    ax.set_ylim(0, 105)
    ax.set_xlabel("epoch", fontsize=11, color=CLR)
    ax.set_ylabel("acc % / lr%", fontsize=11, color=CLR)
    ax.set_title("Acc + Loss + LR", fontsize=13, fontweight="bold", loc="left",
                 pad=4, color=CLR)
    ax.tick_params(labelsize=10, colors=CLR)

    if ax._loss_history:
        loss_epochs = list(range(len(ax._loss_history)))
        ax._twin.plot(loss_epochs, ax._loss_history,
                      color=CLR_ACCENT, linewidth=1.5, alpha=0.85)
        ax._twin.plot(loss_epochs[-1], ax._loss_history[-1], "o",
                      color=CLR_ACCENT, markersize=6, zorder=5)
        ax._twin.set_ylabel("loss", fontsize=11, color=CLR_ACCENT)
        ax._twin.tick_params(labelsize=10, colors=CLR_ACCENT)
        for spine in ax._twin.spines.values():
            spine.set_linewidth(2)
            spine.set_color("black")
            spine.set_visible(True)
    ax._twin.set_xlim(0, max(total_epochs - 1, 1))


_GRAD_LAYER_COLORS = {
    "W_in":  "#1f77b4",  # blue
    "W_hid": "#2ca02c",  # green (output projection)
    "W_ee":  "#9467bd",  # purple
    "W_ei":  "#d62728",  # red
    "W_ie":  "#ff7f0e",  # orange
}


def _draw_rate_curve(ax, rate_e, rate_i, has_inh, total_epochs):
    """Draw E (black) and I (red) population rates over epochs."""
    if not hasattr(ax, '_rate_history'):
        ax._rate_e_history = []
        ax._rate_i_history = []
    ax._rate_e_history.append(rate_e)
    ax._rate_i_history.append(rate_i if has_inh else 0)
    ax._rate_history = True  # marker so we don't reinit

    ax.clear()
    epochs = list(range(len(ax._rate_e_history)))
    ax.plot(epochs, ax._rate_e_history, color=CLR, linewidth=1.5, label="E")
    ax.plot(epochs[-1], ax._rate_e_history[-1], "o",
            color=CLR, markersize=6, zorder=5)
    if has_inh:
        ax.plot(epochs, ax._rate_i_history, color=CLR_ACCENT,
                linewidth=1.5, label="I")
        ax.plot(epochs[-1], ax._rate_i_history[-1], "o",
                color=CLR_ACCENT, markersize=6, zorder=5)
    ax.set_xlim(0, max(total_epochs - 1, 1))
    ax.set_ylim(bottom=0)
    ax.set_xlabel("epoch", fontsize=11, color=CLR)
    ax.set_ylabel("Hz", fontsize=11, color=CLR)
    ax.set_title("E/I Rates", fontsize=13, fontweight="bold", loc="left",
                 pad=4, color=CLR)
    ax.tick_params(labelsize=10, colors=CLR)


def _draw_grad_flow(ax, grad_ratios, total_epochs):
    """Draw per-layer ‖grad‖/‖weight‖ ratios over epochs (log y).

    Healthy band ~1e-3 to 1e-2 is shaded. Below = vanishing, above = exploding.
    grad_ratios is a dict[layer_name, float] for the current epoch.
    """
    if not hasattr(ax, '_grad_flow_history'):
        ax._grad_flow_history = {}
    for name, ratio in grad_ratios.items():
        if ratio > 0:
            ax._grad_flow_history.setdefault(name, []).append(ratio)

    ax.clear()
    ax.axhspan(1e-3, 1e-2, color=CLR_LIGHT, alpha=0.2, zorder=0)
    for name, hist in ax._grad_flow_history.items():
        color = _GRAD_LAYER_COLORS.get(name, CLR)
        epochs = list(range(len(hist)))
        ax.plot(epochs, hist, color=color, linewidth=1.3, label=name, alpha=0.9)
    ax.set_yscale("log")
    ax.set_xlim(0, max(total_epochs - 1, 1))
    ax.set_xlabel("epoch", fontsize=11, color=CLR)
    ax.set_ylabel("‖∇‖/‖W‖", fontsize=11, color=CLR)
    ax.set_title("Grad Flow", fontsize=13, fontweight="bold", loc="left",
                 pad=4, color=CLR)
    ax.tick_params(labelsize=10, colors=CLR)
    if ax._grad_flow_history:
        ax.legend(fontsize=8, loc="best", frameon=False, ncol=2,
                  handlelength=1.2, columnspacing=0.8, labelspacing=0.2)


def _draw_digit_image(ax, digit_image):
    """Draw a digit image square, left-aligned in its cell."""
    ax.clear()
    ax.axis("off")
    if digit_image is not None:
        # Create a left-aligned square inset
        pos = ax.get_position()
        h = pos.height
        w = h * (ax.figure.get_figheight() / ax.figure.get_figwidth())
        inset = ax.figure.add_axes([pos.x0, pos.y0, w, h])
        inset.imshow(digit_image, cmap="gray_r", interpolation="nearest",
                     aspect="equal")
        inset.set_xticks([])
        inset.set_yticks([])
        inset.set_title("Input", fontsize=13, fontweight="bold", loc="left",
                        pad=4, color=CLR)
        for spine in inset.spines.values():
            spine.set_linewidth(2)
            spine.set_color(CLR)




def reset_weight_xlims():
    """No-op — kept for API compat. Axes are now fully auto."""
    pass


_WEIGHT_DISPLAY_ORDER = [
    ("W_in", "In"),
    ("W_ff_", None),     # intermediate feedforward (dynamic label)
    ("W_out", "Out"),
    ("W_hid", "Hid"),    # legacy
    ("W_rec", None),     # recurrent (dynamic label)
    ("W_ee", None),      # E→E (dynamic label)
    ("W_ei", None),      # E→I (dynamic label)
    ("W_ie", None),      # I→E (dynamic label)
]

def _draw_weight_histograms(ax_ws, weights):
    """Draw weight distribution histograms dynamically for all available weights."""
    # Build ordered (label, data) list
    items = []
    used = set()

    def _add(key, label):
        if key in weights and key not in used:
            items.append((label, weights[key]))
            used.add(key)

    # Fixed keys first
    _add("W_in", "In")

    # Intermediate feedforward (W_ff_2, W_ff_3, ...)
    for k in sorted(k for k in weights if k.startswith("W_ff_")):
        idx = k.replace("W_ff_", "")
        items.append((f"H{idx}", weights[k]))
        used.add(k)

    # Output
    _add("W_out", "Out")
    _add("W_hid", "Hid")  # legacy

    # Recurrent (W_rec, W_rec_1, W_rec_2, ...)
    for k in sorted(k for k in weights if k.startswith("W_rec")):
        if k not in used:
            lbl = "Rec" if k == "W_rec" else f"Rec{k.split('_')[-1]}"
            items.append((lbl, weights[k]))
            used.add(k)

    # E-I weights (W_ee, W_ei, W_ie, or indexed versions)
    for prefix, base_lbl in [("W_ee", "E\u2192E"), ("W_ei", "E\u2192I"), ("W_ie", "I\u2192E")]:
        for k in sorted(k for k in weights if k.startswith(prefix)):
            if k not in used:
                lbl = base_lbl if k == prefix else f"{base_lbl}{k.split('_')[-1]}"
                items.append((lbl, weights[k]))
                used.add(k)

    # Draw into available axes
    for i, ax_w in enumerate(ax_ws):
        ax_w.clear()
        for spine in ax_w.spines.values():
            spine.set_linewidth(1.5)
            spine.set_color(CLR)
        if i < len(items):
            label, w_data = items[i]
            vals = w_data.ravel()
            vals = vals[np.isfinite(vals)]
            if len(vals) > 0:
                ax_w.hist(vals, bins=30, color=CLR, alpha=0.7,
                          edgecolor="white", linewidth=0.3)
            if (w_data < 0).any():
                ax_w.axvline(0, color=CLR, linewidth=0.5, alpha=0.4)
            ax_w.set_title(label, fontsize=10, fontweight="bold", pad=2, color=CLR)
        else:
            ax_w.set_visible(False)
        ax_w.set_yticks([])
        ax_w.tick_params(labelsize=8, colors=CLR, length=3)


def _draw_header(ax_header, title, ratio, dt, rate_e_hz, rate_i_hz,
                 has_inh, model_name, t_e_async, f0=0.0, frame_idx=None,
                 pop_cv=0.0, ie_ratio=0.0, active_frac=1.0, rate_cv=0.0,
                 n_e=1024, is_coba=True, input_rate=None):
    """Draw the header bar with title, stats, and metrics."""
    ax_header.clear()
    ax_header.axis("off")

    # Fixed-width metric table — label bold, value next to it
    spinner = f"{_SPINNER[frame_idx % len(_SPINNER)]} " if frame_idx is not None else ""
    f0_val = f"{f0:.0f}" if f0 > 0 else "-"
    i_val = f"{rate_i_hz:.0f}" if has_inh else "-"

    # Fixed grid columns — each col has (x_start, label_width)
    # 7 columns, same x positions for both rows.
    # Col 0 is widened to fit longer model names like SNNTORCH / SNN-EXP.
    col_x = [0.00, 0.14, 0.25, 0.34, 0.44, 0.54, 0.64]
    lw = 0.045  # label width offset

    # Short display names so col 0 doesn't overflow
    _display_names = {"snntorch": "snnTorch", "cuba": "CUBA"}
    display_model = _display_names.get(model_name, model_name.upper())

    in_str = f"{input_rate:.0f}Hz" if input_rate is not None else "-"
    row1 = [
        ("", f"{spinner}{display_model}"),
        ("dt", f"{dt:.2f}ms"),
        ("N", f"{n_e}"),
        ("in", in_str),
    ]
    row2 = [
        ("E", f"{rate_e_hz:.0f}Hz"),
        ("I", f"{i_val}Hz"),
        ("f\u2080", f"{f0_val}Hz"),
        ("CV", f"{pop_cv:.2f}"),
        ("I/E", f"{ie_ratio:.1f}"),
        ("act", f"{active_frac:.0%}"),
    ]

    fs = 19
    for row, y in [(row1, 0.82), (row2, 0.18)]:
        for i, (label, val) in enumerate(row):
            if i >= len(col_x):
                break
            x = col_x[i]
            if label:
                ax_header.text(x, y, label, fontsize=fs, fontweight="bold",
                               ha="left", va="center", transform=ax_header.transAxes,
                               color=CLR, fontfamily="monospace")
                ax_header.text(x + len(label) * 0.013 + 0.005, y, val,
                               fontsize=fs, fontweight="normal",
                               ha="left", va="center", transform=ax_header.transAxes,
                               color=CLR, fontfamily="monospace")
            else:
                ax_header.text(x, y, val, fontsize=fs, fontweight="bold",
                               ha="left", va="center", transform=ax_header.transAxes,
                               color=CLR, fontfamily="monospace")


def _draw_progress_bar(ax_prog, ratio, sweep_var, sweep_range, sweep_progress):
    """Draw the sweep progress bar."""
    ax_prog.clear()
    ax_prog.axis("off")
    if sweep_var is None:
        return

    ax_prog.set_xlim(0, 1)
    ax_prog.set_ylim(0, 1)

    lo, hi = sweep_range
    progress = sweep_progress if sweep_progress is not None else 0

    if sweep_var == "OD":
        current_val = f"{ratio:.1f}x"
        range_str = f"{lo:.1f}x \u2192 {hi:.1f}x"
    else:
        current_dt = lo + progress * (hi - lo)
        current_val = f"{current_dt:.3g} ms"
        range_str = f"{lo:.3g} \u2192 {hi:.3g} ms"

    ax_prog.text(0.0, 0.95, f"{sweep_var}  {range_str}",
                 fontsize=12, va="bottom", ha="left",
                 transform=ax_prog.transAxes, color=CLR)
    ax_prog.text(1.0, 0.95, current_val,
                 fontsize=14, va="bottom", ha="right",
                 transform=ax_prog.transAxes, fontweight="bold", color=CLR)

    ax_prog.barh(0.2, 1.0, height=0.3, color="#e8e8e8", left=0)
    ax_prog.barh(0.2, progress, height=0.3, color=CLR, left=0, alpha=0.3)
    ax_prog.plot([progress, progress], [0.0, 0.4], color=CLR_ACCENT, linewidth=2.5)


