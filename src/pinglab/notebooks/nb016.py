"""Notebook runner for entry 016 — PING E→I coupling sweep (copy of nb006).

Sweeps the E→I coupling strength from 0 → 1 with *no* stim-window
overdrive (input rate flat through the trial), walking the network
from the async baseline (E and I effectively decoupled) through the
emergence of gamma as the E→I→E feedback loop closes. Input rate and
W_in are bumped relative to the other scans so E has enough baseline
drive to recruit I at all.

Also writes numbers.json with pre/stim/post E and I population rates
from an in-Python replay at the canonical high-ei run so the MDX can
interpolate exact values and the success-criteria check can gate on
PING actually forming once the feedback loop is closed.

Notebook entry: src/docs/src/pages/notebooks/nb016.mdx
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src" / "pinglab"))

import numpy as np  # noqa: E402

from _ping_scan import (  # noqa: E402
    DATASET,
    DIGIT_CLASS,
    DT_MS,
    INPUT_RATE_HZ,
    N_HIDDEN,
    SAMPLE_IDX,
    SEED,
    SIM_MS,
    STEP_OFF_MS,
    STEP_ON_MS,
    ScanSpec,
    run_scan,
)

SLUG = "nb016"
EI_SCAN_INPUT_RATE_HZ = 200.0
EI_SCAN_W_IN_MEAN = 1.8  # 6× over the default 0.3 — bigger E baseline drive
EI_SCAN_W_IN_STD = 0.36  # 6× over the default 0.06
EI_SCAN_MIN = 0.0
EI_SCAN_MAX = 1.0
EI_SCAN_FRAMES = 100
CANON_EI = 0.8  # well inside the PING-on regime for the replay
EI_SIM_MS = 250.0  # shorter trial than nb006's 600 ms
EI_STEP_ON_MS = EI_SIM_MS / 3.0  # rate-window split: pre / stim / post each ≈ 83 ms
EI_STEP_OFF_MS = 2.0 * EI_SIM_MS / 3.0


def compute_summary_rates() -> dict:
    """Replay one canonical-ei forward pass in-process with MNIST d0s0
    spike input (flat rate, no stim-window overdrive), then extract
    pre/stim/post E/I population rates."""
    import torch  # noqa: E402
    import config as C  # noqa: E402
    import models as M  # noqa: E402
    from config import make_net, patch_dt  # noqa: E402
    from oscilloscope import (
        _extract_records,
        _load_dataset_image,
        encode_image_spikes,
        primary_hid_key,
    )  # noqa: E402

    C.cfg.n_e = N_HIDDEN
    C.cfg.n_i = N_HIDDEN // 4
    C.cfg.sim_ms = EI_SIM_MS
    C.cfg.step_on_ms = EI_STEP_ON_MS
    C.cfg.step_off_ms = EI_STEP_OFF_MS
    C.cfg.seed = SEED
    s = CANON_EI
    C.cfg.w_ei = (s, s * 0.1)
    C.cfg.w_ie = (s * C.cfg.ei_ratio, s * C.cfg.ei_ratio * 0.1)
    C._sync_globals_from_cfg(C.cfg)

    pixel_vec, _ = _load_dataset_image(DATASET, DIGIT_CLASS, SAMPLE_IDX)
    M.N_IN = len(pixel_vec)
    M.N_HID = C.N_E
    M.N_INH = C.N_I
    M.T_ms = EI_SIM_MS
    M.max_rate_hz = EI_SCAN_INPUT_RATE_HZ
    patch_dt(DT_MS)

    base_rate = M.max_rate_hz
    input_spikes = encode_image_spikes(
        pixel_vec,
        M.T_steps,
        DT_MS,
        base_rate,
        base_rate,
        C.STEP_ON_MS,
        C.STEP_OFF_MS,
        C.SEED,
    ).to(C.DEVICE)

    net = make_net(
        C.cfg,
        w_in=(EI_SCAN_W_IN_MEAN, EI_SCAN_W_IN_STD, "normal", C.W_IN_SPARSITY),
        model_name="ping",
    )
    net.recording = True
    with torch.no_grad():
        net.forward(input_spikes=input_spikes)
    rec = _extract_records(net)

    spk_e = rec[primary_hid_key(rec)]
    spk_i = rec["inh"]
    T_steps = spk_e.shape[0]
    t_ms = np.arange(T_steps) * DT_MS

    pre = (t_ms >= 0) & (t_ms < EI_STEP_ON_MS)
    stim = (t_ms >= EI_STEP_ON_MS) & (t_ms < EI_STEP_OFF_MS)
    post = (t_ms >= EI_STEP_OFF_MS) & (t_ms <= EI_SIM_MS)

    def mean_rate(spk, mask):
        return float(spk[mask].mean()) * 1000.0 / DT_MS

    return {
        "pre": {"e": mean_rate(spk_e, pre), "i": mean_rate(spk_i, pre)},
        "stim": {"e": mean_rate(spk_e, stim), "i": mean_rate(spk_i, stim)},
        "post": {"e": mean_rate(spk_e, post), "i": mean_rate(spk_i, post)},
    }


def compute_per_frame_pop_rates() -> dict:
    """For each ei_strength value in the sweep, run one in-process forward
    pass and return the smoothed E/I population rate time-series. Used for
    the stacked per-frame rate plot."""
    import torch  # noqa: E402
    import config as C  # noqa: E402
    import models as M  # noqa: E402
    from config import make_net, patch_dt  # noqa: E402
    from oscilloscope import (
        _extract_records,
        _load_dataset_image,
        encode_image_spikes,
        primary_hid_key,
    )  # noqa: E402

    pixel_vec, _ = _load_dataset_image(DATASET, DIGIT_CLASS, SAMPLE_IDX)
    M.N_IN = len(pixel_vec)
    M.T_ms = EI_SIM_MS

    ei_values = np.linspace(EI_SCAN_MIN, EI_SCAN_MAX, EI_SCAN_FRAMES).tolist()
    bin_ms = 2.0  # smoothing bin for the time-series
    bin_steps = int(round(bin_ms / DT_MS))
    out: list[dict] = []

    for ei in ei_values:
        C.cfg.n_e = N_HIDDEN
        C.cfg.n_i = N_HIDDEN // 4
        C.cfg.sim_ms = EI_SIM_MS
        C.cfg.step_on_ms = EI_STEP_ON_MS
        C.cfg.step_off_ms = EI_STEP_OFF_MS
        C.cfg.seed = SEED
        s = float(ei)
        C.cfg.w_ei = (s, s * 0.1)
        C.cfg.w_ie = (s * C.cfg.ei_ratio, s * C.cfg.ei_ratio * 0.1)
        C._sync_globals_from_cfg(C.cfg)

        M.N_HID = C.N_E
        M.N_INH = C.N_I
        M.max_rate_hz = EI_SCAN_INPUT_RATE_HZ
        patch_dt(DT_MS)

        base_rate = M.max_rate_hz
        input_spikes = encode_image_spikes(
            pixel_vec, M.T_steps, DT_MS, base_rate, base_rate,
            C.STEP_ON_MS, C.STEP_OFF_MS, C.SEED,
        ).to(C.DEVICE)
        net = make_net(
            C.cfg,
            w_in=(EI_SCAN_W_IN_MEAN, EI_SCAN_W_IN_STD, "normal", C.W_IN_SPARSITY),
            model_name="ping",
        )
        net.recording = True
        with torch.no_grad():
            net.forward(input_spikes=input_spikes)
        rec = _extract_records(net)
        spk_e = rec[primary_hid_key(rec)]  # (T, N_E)
        spk_i = rec["inh"]                  # (T, N_I)

        T_steps = spk_e.shape[0]
        n_bins = T_steps // bin_steps
        usable = n_bins * bin_steps
        e_pop = (spk_e[:usable].reshape(n_bins, bin_steps, -1)
                 .sum(axis=(1, 2))) * 1000.0 / (bin_ms * spk_e.shape[1])
        i_pop = (spk_i[:usable].reshape(n_bins, bin_steps, -1)
                 .sum(axis=(1, 2))) * 1000.0 / (bin_ms * spk_i.shape[1])
        t_ms = (np.arange(n_bins) + 0.5) * bin_ms
        out.append({
            "ei_strength": s,
            "time_ms": t_ms.tolist(),
            "e_rate_hz": e_pop.tolist(),
            "i_rate_hz": i_pop.tolist(),
        })
    return {"frames": out, "bin_ms": bin_ms}


def compute_autocorr_metric(frames_data: dict,
                            onset_skip_ms: float = 20.0,
                            lag_min_ms: float = 5.0,
                            lag_max_ms: float = 60.0) -> dict:
    """Per-frame autocorrelation of the post-onset E rate. Returns the full
    autocorrelation curve plus a single-trace rhythmicity score: the
    *prominence* of the strongest peak in the [lag_min, lag_max] window
    (peak height minus its nearest valley). Prominence is naturally zero
    when there's no real peak above the noise floor, so async traces score
    near zero without any baseline subtraction or shuffling."""
    from scipy.signal import find_peaks

    bin_ms = float(frames_data["bin_ms"])
    out: list[dict] = []
    for f in frames_data["frames"]:
        rate = np.asarray(f["e_rate_hz"], dtype=np.float64)
        skip = int(round(onset_skip_ms / bin_ms))
        x = rate[skip:]
        x = x - x.mean()
        sd = x.std()
        if sd < 1e-9:
            corr = np.zeros(x.size)
        else:
            x = x / sd
            n = x.size
            spec = np.fft.rfft(x, n=2 * n)
            corr = np.fft.irfft(spec * np.conj(spec))[:n] / n
        lag_steps = np.arange(corr.size)
        lag_ms = lag_steps * bin_ms

        win = (lag_ms >= lag_min_ms) & (lag_ms <= lag_max_ms)
        peak_lag = float("nan")
        peak_value = float("nan")
        peak_prominence = 0.0
        if win.any():
            sub = corr[win]
            sub_lag = lag_ms[win]
            # min separation: at least 5 ms apart so we ignore wiggles
            distance = max(int(round(5.0 / bin_ms)), 1)
            peaks, props = find_peaks(sub, distance=distance, prominence=0.0)
            if peaks.size:
                best = int(np.argmax(props["prominences"]))
                peak_lag = float(sub_lag[peaks[best]])
                peak_value = float(sub[peaks[best]])
                peak_prominence = float(props["prominences"][best])
        out.append({
            "ei_strength": f["ei_strength"],
            "lag_ms": lag_ms.tolist(),
            "autocorr": corr.tolist(),
            "peak_lag_ms": peak_lag,
            "peak_value": peak_value,
            "peak_prominence": peak_prominence,
        })
    return {
        "frames": out,
        "bin_ms": bin_ms,
        "onset_skip_ms": onset_skip_ms,
        "lag_min_ms": lag_min_ms,
        "lag_max_ms": lag_max_ms,
    }


def plot_autocorr_stack(autocorr_data: dict, out_path: Path) -> Path:
    """Stacked autocorrelation curves, one row per ei (sampled to ≤10)."""
    from matplotlib.patches import Rectangle
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import theme  # type: ignore[import]

    theme.apply()
    all_frames = autocorr_data["frames"]
    n_total = len(all_frames)
    if n_total <= 10:
        frames = all_frames
    else:
        idx = np.linspace(0, n_total - 1, 10).round().astype(int)
        frames = [all_frames[int(i)] for i in idx]
    n = len(frames)

    fig, axes = plt.subplots(n, 1, figsize=(10.0, 5.625),
                             sharex=True, sharey=True)
    if n == 1:
        axes = [axes]
    lag_max = autocorr_data["lag_max_ms"]
    for ax, f in zip(axes, frames):
        lag = np.asarray(f["lag_ms"])
        corr = np.asarray(f["autocorr"])
        keep = lag <= lag_max
        ax.axhline(0, color=theme.MUTED, linewidth=0.5)
        ax.plot(lag[keep], corr[keep], color=theme.INK_BLACK, linewidth=1.0)
        ax.set_ylim(-0.6, 1.05)
        ax.set_yticks([])
        ax.text(0.985, 0.85, f"ei = {f['ei_strength']:.2f}",
                ha="right", va="top", transform=ax.transAxes,
                fontsize=theme.SIZE_ANNOTATION, color=theme.INK)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
    axes[-1].set_xlabel("lag (ms)")
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.94))
    pad_x, pad_y = 0.01, 0.01
    bboxes = [ax.get_position() for ax in axes]
    x0 = min(b.x0 for b in bboxes) - pad_x
    x1 = max(b.x1 for b in bboxes) + pad_x
    y0 = min(b.y0 for b in bboxes) - pad_y
    y1 = max(b.y1 for b in bboxes) + pad_y
    fig.patches.append(Rectangle(
        (x0, y0), x1 - x0, y1 - y0, transform=fig.transFigure,
        fill=False, edgecolor=theme.INK_BLACK, linewidth=1.0,
    ))
    fig.text((x0 + x1) / 2, y1 + 0.02,
             "per-frame autocorrelation of the post-onset E rate",
             ha="center", va="bottom", fontsize=theme.SIZE_TITLE)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → {out_path}")
    return out_path


def plot_autocorr_metric(autocorr_data: dict, out_path: Path) -> Path:
    """Peak prominence of the autocorrelation in [lag_min, lag_max] vs ei —
    single-frame, single-trace rhythmicity score per frame of the sweep."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import theme  # type: ignore[import]

    theme.apply()
    rows = autocorr_data["frames"]
    eis = [r["ei_strength"] for r in rows]
    proms = [r["peak_prominence"] for r in rows]

    fig, ax = plt.subplots(figsize=(10.0, 5.625))
    ax.axhline(0, color=theme.MUTED, linewidth=0.8, linestyle="--")
    ax.plot(eis, proms, marker="o", color=theme.INK_BLACK, linewidth=1.5)
    ax.set_xlabel("ei strength")
    ax.set_ylabel("autocorrelation peak prominence (post-onset)")
    ax.set_title(
        f"rhythmicity from autocorr: peak prominence in lag "
        f"{autocorr_data['lag_min_ms']:.0f}–{autocorr_data['lag_max_ms']:.0f} ms",
        fontsize=theme.SIZE_TITLE,
    )
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → {out_path}")
    return out_path


def plot_per_frame_pop_rates(frames_data: dict, out_path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import theme  # type: ignore[import]

    from matplotlib.patches import Rectangle  # local import keeps deps explicit

    theme.apply()
    all_frames = frames_data["frames"]
    # Show at most 10 rows: evenly-spaced from start to end of the sweep.
    n_total = len(all_frames)
    if n_total <= 10:
        frames = all_frames
    else:
        idx = np.linspace(0, n_total - 1, 10).round().astype(int)
        frames = [all_frames[int(i)] for i in idx]
    n = len(frames)
    # 16:9 overall; per-row height shrinks with n.
    fig, axes = plt.subplots(n, 1, figsize=(10.0, 5.625),
                             sharex=True, sharey=True)
    if n == 1:
        axes = [axes]
    y_max = max(max(f["e_rate_hz"]) for f in frames)
    for ax, f in zip(axes, frames):
        ax.plot(f["time_ms"], f["e_rate_hz"], color=theme.INK_BLACK,
                linewidth=1.0)
        ax.set_ylim(0, y_max * 1.05)
        ax.set_yticks([])
        ax.text(0.985, 0.85, f"ei = {f['ei_strength']:.2f}",
                ha="right", va="top", transform=ax.transAxes,
                fontsize=theme.SIZE_ANNOTATION,
                color=theme.INK)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
    axes[-1].set_xlabel("time (ms)")
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.94))
    # Frame in figure coords, hugging just the plot area (axes bbox union).
    pad_x, pad_y = 0.01, 0.01
    bboxes = [ax.get_position() for ax in axes]
    x0 = min(b.x0 for b in bboxes) - pad_x
    x1 = max(b.x1 for b in bboxes) + pad_x
    y0 = min(b.y0 for b in bboxes) - pad_y
    y1 = max(b.y1 for b in bboxes) + pad_y
    fig.patches.append(Rectangle(
        (x0, y0), x1 - x0, y1 - y0, transform=fig.transFigure,
        fill=False, edgecolor=theme.INK_BLACK, linewidth=1.0,
    ))
    # Title: center on the frame's x-range, sit just above the top edge.
    fig.text((x0 + x1) / 2, y1 + 0.02,
             "E population rate per frame along the ei-strength sweep",
             ha="center", va="bottom", fontsize=theme.SIZE_TITLE)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → {out_path}")
    return out_path


def extras(tier: str, notebook_run_id: str) -> dict:
    rates = compute_summary_rates()
    r = rates
    print(
        f"  E rate  pre={r['pre']['e']:.1f} Hz  "
        f"stim={r['stim']['e']:.1f} Hz  post={r['post']['e']:.1f} Hz"
    )
    print(
        f"  I rate  pre={r['pre']['i']:.1f} Hz  "
        f"stim={r['stim']['i']:.1f} Hz  post={r['post']['i']:.1f} Hz"
    )
    print("  per-frame pop-rate replay…")
    frames_data = compute_per_frame_pop_rates()
    figures = REPO / "src" / "docs" / "public" / "figures" / "notebooks" / SLUG
    plot_per_frame_pop_rates(frames_data, figures / "pop_rates_stack.png")
    print("  per-frame autocorr metric…")
    autocorr_data = compute_autocorr_metric(frames_data)
    plot_autocorr_stack(autocorr_data, figures / "autocorr_stack.png")
    plot_autocorr_metric(autocorr_data, figures / "autocorr_metric.png")
    return {
        "rates_hz": rates,
        "canonical_ei": CANON_EI,
        "frame_pop_rates": frames_data,
        "autocorr_metric": {
            "bin_ms": autocorr_data["bin_ms"],
            "onset_skip_ms": autocorr_data["onset_skip_ms"],
            "lag_min_ms": autocorr_data["lag_min_ms"],
            "lag_max_ms": autocorr_data["lag_max_ms"],
            "frames": [
                {
                    "ei_strength": r["ei_strength"],
                    "peak_lag_ms": r["peak_lag_ms"],
                    "peak_value": r["peak_value"],
                    "peak_prominence": r["peak_prominence"],
                }
                for r in autocorr_data["frames"]
            ],
        },
    }


def evaluate_success(figures_dir, summary):
    """Criteria: scan video rendered, and PING actually forms at the
    canonical high-ei coupling (I fires across the full trial since
    input is flat). The I-rate check mirrors nb003 / nb005 — guards
    against a regression where the network never recruits I."""
    video = figures_dir / "scan_ei.mp4"
    video_ok = video.exists() and video.stat().st_size > 0
    href = "/" + str(video.relative_to(figures_dir.parents[2])) if video_ok else None

    rates = summary.get("rates_hz", {})
    i_stim = rates.get("stim", {}).get("i", 0.0)
    e_stim = rates.get("stim", {}).get("e", 0.0)
    ping_formed = i_stim > 1.0 and e_stim > 1.0
    return [
        {
            "label": "ei-strength scan video rendered",
            "passed": bool(video_ok),
            "detail": f"{video.name} ({video.stat().st_size} bytes)"
            if video_ok
            else f"missing {video.name}",
            "detail_href": href,
        },
        {
            "label": f"PING forms at canonical ei ({CANON_EI})",
            "passed": bool(ping_formed),
            "detail": f"E stim={e_stim:.1f} Hz, I stim={i_stim:.1f} Hz",
        },
    ]


if __name__ == "__main__":
    run_scan(
        ScanSpec(
            slug=SLUG,
            scan_var="ei_strength",
            scan_min=EI_SCAN_MIN,
            scan_max=EI_SCAN_MAX,
            video_name="scan_ei.mp4",
            extra_osc_args=[
                "--input-rate",
                str(EI_SCAN_INPUT_RATE_HZ),
                "--w-in",
                str(EI_SCAN_W_IN_MEAN),
                str(EI_SCAN_W_IN_STD),
                "--stim-overdrive",
                "1.0",
                "--dt",
                str(DT_MS),
            ],
            config_payload={
                "fixed_overdrive": 1.0,
                "input_rate_hz": EI_SCAN_INPUT_RATE_HZ,
                "w_in_mean": EI_SCAN_W_IN_MEAN,
                "w_in_std": EI_SCAN_W_IN_STD,
            },
            extras_fn=extras,
            criteria_fn=evaluate_success,
            fps=10,
            sim_ms=EI_SIM_MS,
            frames=EI_SCAN_FRAMES,
        )
    )
    sys.exit(0)
