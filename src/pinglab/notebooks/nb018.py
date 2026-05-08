"""Notebook runner for entry 017 — ping metrics from the population rate.

Forked from nb017 to focus on the generative rate model for the E
population: a Gaussian with a harmonically-modulated mean
    r_t ~ N(μ_0 + Σ_k [β_c^(k) cos(2π k f t) + β_s^(k) sin(2π k f t)], σ_p²)
fit by closed-form least-squares at each f on a gamma-band grid; we
pick the f that minimises the residual variance. K = 1 recovers a
smooth cosine; K = 3 supplies the asymmetric pulse shape of real
PING bursts. All-zero coefficients recover a constant-mean Gaussian
(the async limit).

The runner produces the canonical ei-strength scan video and a
stacked real-vs-simulated rate-model validation panel.

Notebook entry: src/docs/src/pages/notebooks/nb018.mdx
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

SLUG = "nb018"
EI_SCAN_INPUT_RATE_HZ = 200.0
EI_SCAN_W_IN_MEAN = 1.8  # 6× over the default 0.3 — bigger E baseline drive
EI_SCAN_W_IN_STD = 0.36  # 6× over the default 0.06
EI_SCAN_MIN = 0.0
EI_SCAN_MAX = 1.0
EI_SCAN_FRAMES = 10
CANON_EI = 0.8  # well inside the PING-on regime for the replay
EI_SIM_MS = 1000.0  # 4× the nb006/nb017/early-nb018 trial — more bins per fit
EI_STEP_ON_MS = EI_SIM_MS / 3.0  # rate-window split: pre / stim / post each ≈ 333 ms
EI_STEP_OFF_MS = 2.0 * EI_SIM_MS / 3.0


def compute_summary_rates() -> dict:
    """Replay one canonical-ei forward pass in-process with MNIST d0s0
    spike input (flat rate, no stim-window overdrive), then extract
    pre/stim/post E/I population rates."""
    import config as C  # noqa: E402
    import models as M  # noqa: E402
    import torch  # noqa: E402
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
    import config as C  # noqa: E402
    import models as M  # noqa: E402
    import torch  # noqa: E402
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
        spk_e = rec[primary_hid_key(rec)]  # (T, N_E)

        T_steps = spk_e.shape[0]
        n_bins = T_steps // bin_steps
        usable = n_bins * bin_steps
        # Participation fraction (matches the oscilloscope video panel):
        # total spikes per bin divided by N_E. Bounded in [0, 1] when each
        # neuron fires at most once per bin; equivalently it's the bin's
        # rate (Hz) divided by the maximum possible rate (1000/bin_ms).
        e_pop = (
            spk_e[:usable].reshape(n_bins, bin_steps, -1).sum(axis=(1, 2))
        ) / spk_e.shape[1]
        t_ms = (np.arange(n_bins) + 0.5) * bin_ms
        out.append(
            {
                "ei_strength": s,
                "time_ms": t_ms.tolist(),
                "e_participation": e_pop.tolist(),
            }
        )
    return {"frames": out, "bin_ms": bin_ms}


def _pulse_comb(t_ms: "np.ndarray", f_hz: float, t0_ms: float,
                sigma_b_ms: float) -> "np.ndarray":
    """Unit-amplitude Gaussian pulse train: Σ_n exp(-(t - t_0 - n/f)² / 2σ_b²),
    summed over enough integer n to cover the t_ms window plus 3σ_b on each
    side."""
    T = 1000.0 / f_hz
    t_min, t_max = float(t_ms.min()), float(t_ms.max())
    pad = 3.0 * sigma_b_ms
    n_min = int(np.floor((t_min - t0_ms - pad) / T))
    n_max = int(np.ceil((t_max - t0_ms + pad) / T))
    out = np.zeros_like(t_ms, dtype=np.float64)
    two_sigma_sq = 2.0 * sigma_b_ms ** 2
    for n in range(n_min, n_max + 1):
        centre = t0_ms + n * T
        out = out + np.exp(-((t_ms - centre) ** 2) / two_sigma_sq)
    return out


def _ping_mean(t_ms: "np.ndarray", fit: dict) -> "np.ndarray":
    """Reconstruct the ping mean curve μ_0 + A · pulse-comb(f, t_0, σ_b)
    from a fit returned by `_fit_ping`."""
    return fit["mu_0"] + fit["A"] * _pulse_comb(
        t_ms, fit["f_hz"], fit["t0_ms"], fit["sigma_b_ms"]
    )


def _sample_ping(t_ms: "np.ndarray", fit: dict, rng, n_e: int) -> "np.ndarray":
    """Ping sample: at each bin, k_t ~ Binomial(N_E, p_t) with
    p_t = clip(μ(t), 0, 1) the pulse-train mean. r_t = k_t / N_E.
    Variance at the trough (p ≈ 0) is zero — cadence reads cleanly."""
    p = np.clip(_ping_mean(t_ms, fit), 0.0, 1.0)
    return rng.binomial(n_e, p).astype(np.float64) / n_e


def _ping_grid_search(r, t_ms, f_grid_hz, sigma_b_grid_ms, phase_fracs):
    """Inner loop of `_fit_ping`: triple-grid LS over (f, t_0, σ_b);
    closed-form (μ_0, A) at each grid point. Returns the best dict
    (or None) using the same schema as `_fit_ping`."""
    best = None
    ones = np.ones_like(t_ms, dtype=np.float64)
    for f_hz in f_grid_hz:
        T = 1000.0 / float(f_hz)
        for phase_frac in phase_fracs:
            t0_ms = float(phase_frac) * T
            for sigma_b in sigma_b_grid_ms:
                comb = _pulse_comb(t_ms, float(f_hz), t0_ms, float(sigma_b))
                X = np.column_stack([ones, comb])
                try:
                    betas, *_ = np.linalg.lstsq(X, r, rcond=None)
                except np.linalg.LinAlgError:
                    continue
                if not np.all(np.isfinite(betas)) or betas[1] < 0:
                    continue  # require A ≥ 0 (pulses up, not down)
                residuals = r - X @ betas
                sigma_p_sq = float((residuals ** 2).mean())
                if not np.isfinite(sigma_p_sq):
                    continue
                if best is None or sigma_p_sq < best["sigma_p_sq"]:
                    best = {
                        "f_hz": float(f_hz),
                        "t0_ms": t0_ms,
                        "sigma_b_ms": float(sigma_b),
                        "mu_0": float(betas[0]),
                        "A": float(betas[1]),
                        "sigma_p_sq": sigma_p_sq,
                    }
    return best


def _fit_ping(r: "np.ndarray", t_ms: "np.ndarray",
              f_grid_hz: "np.ndarray",
              sigma_b_grid_ms: "np.ndarray | None" = None,
              n_phase: int = 16) -> dict | None:
    """Ping generative model — Gaussian pulse train:
        r_t = μ_0 + A · Σ_n exp(-(t - t_0 - n/f)² / 2σ_b²) + ε_t,
        ε_t ~ N(0, σ_p²),  A ≥ 0.
    Two-pass grid: coarse over the full f range, then refined around the
    coarse winner at 0.05 Hz spacing. Two passes because at f ≈ 55 Hz
    over a 1 s trace, sub-0.5 Hz grid drift accumulates several ms of
    phase error by the end. (μ_0, A) are closed-form LS at each grid
    point; we keep the (f, t_0, σ_b) that minimises σ_p²."""
    if sigma_b_grid_ms is None:
        sigma_b_grid_ms = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0])
    sigma_b_grid_ms = np.asarray(sigma_b_grid_ms, dtype=np.float64)
    phase_fracs = np.linspace(0.0, 1.0, int(n_phase), endpoint=False)

    coarse = _ping_grid_search(r, t_ms, f_grid_hz, sigma_b_grid_ms, phase_fracs)
    if coarse is None:
        return None
    f_centre = coarse["f_hz"]
    fine_f = np.arange(max(1.0, f_centre - 1.0), f_centre + 1.0 + 1e-9, 0.05)
    fine = _ping_grid_search(r, t_ms, fine_f, sigma_b_grid_ms, phase_fracs)
    best = fine if (fine is not None and fine["sigma_p_sq"] <= coarse["sigma_p_sq"]) else coarse
    best["sigma_p"] = float(np.sqrt(best["sigma_p_sq"]))
    return best



def plot_rate_model_validation(
    frames_data: dict, out_path: Path, n_rows: int = 10, f_grid_hz=None
) -> Path:
    """Stacked real-with-simulation-overlay across the ei sweep. For each
    of n_rows evenly-spaced ei values, fit the pulse-train rate model on
    the post-onset participation trace and overlay one binomial sample
    (red dashed) on the real trace (black). Async is the A → 0 limit of
    the same model, so no regime switch."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import theme  # type: ignore[import]

    if f_grid_hz is None:
        # 0.5 Hz spacing across the gamma band — coarser grids let model
        # period drift several ms over a 1 s trace, washing out burst
        # alignment and biasing A toward zero.
        f_grid_hz = np.arange(25.0, 75.01, 0.5)
    f_grid_hz = np.asarray(f_grid_hz, dtype=np.float64)

    bin_ms = float(frames_data["bin_ms"])
    onset_skip_bins = int(round(20.0 / bin_ms))
    rng = np.random.default_rng(SEED)

    all_frames = frames_data["frames"]
    n_total = len(all_frames)
    if n_total <= n_rows:
        rows = all_frames
    else:
        idx = np.linspace(0, n_total - 1, n_rows).round().astype(int)
        rows = [all_frames[int(i)] for i in idx]

    theme.apply()
    n = len(rows)
    fig, axes = plt.subplots(
        n, 1, figsize=(10.0, 11.25), sharex=True,
        gridspec_kw={"hspace": 0.0},
    )
    if n == 1:
        axes = [axes]

    panels = []
    for frame in rows:
        full_t = np.asarray(frame["time_ms"])
        full_r = np.asarray(frame["e_participation"])
        t_ms = full_t[onset_skip_bins:] - full_t[onset_skip_bins]
        r = full_r[onset_skip_bins:]
        ei = float(frame["ei_strength"])
        fit = _fit_ping(r, t_ms, f_grid_hz)
        if fit is None:
            # Degenerate trace (all zeros, etc.); fall back to constant rate.
            mu = float(np.clip(r.mean(), 0.0, 1.0))
            sim = rng.binomial(N_HIDDEN, mu, size=t_ms.size).astype(np.float64) / N_HIDDEN
        else:
            sim = _sample_ping(t_ms, fit, rng, N_HIDDEN)
        panels.append({
            "ei": ei,
            "t_ms": t_ms,
            "real": r,
            "sim": sim,
        })
    # Display only the first ~250 ms of each post-onset trace so the gamma
    # rhythm reads clearly; the fit was done on the full window. Fixed
    # y-range matches the oscilloscope video panel (plot.py: ylim(0, 0.5)).
    display_t_max_ms = 250.0
    for ax, p in zip(axes, panels):
        keep = p["t_ms"] <= display_t_max_ms
        t_disp = p["t_ms"][keep]
        real_disp = p["real"][keep]
        sim_disp = p["sim"][keep]
        ax.plot(t_disp, real_disp, color=theme.INK_BLACK, linewidth=1.0,
                label="real")
        ax.plot(t_disp, sim_disp, color="red", linewidth=1.0,
                linestyle="--", alpha=0.85, label="sim")
        # Per-row y-max: tightest bound on what's actually plotted, with a
        # 5 % margin so peaks don't kiss the spine.
        y_max = float(max(real_disp.max(), sim_disp.max())) * 1.05
        if y_max <= 0.0:
            y_max = 1.0 / N_HIDDEN  # one-spike floor for a fully empty trace
        ax.set_xlim(0, display_t_max_ms)
        ax.set_ylim(0.0, y_max)
        # Only label the per-row max — adjacent rows would otherwise collide
        # at hspace=0 (this row's 0 sits on top of the next row's y_max).
        ax.set_yticks([y_max])
        ax.set_yticklabels([f"{y_max:.2f}"])
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.text(
            0.985,
            0.85,
            f"ei = {p['ei']:.2f}",
            ha="right",
            va="top",
            transform=ax.transAxes,
            fontsize=theme.SIZE_ANNOTATION,
            color=theme.INK,
        )
    axes[0].legend(
        loc="upper left", frameon=False, fontsize=theme.SIZE_ANNOTATION,
        ncol=2, handlelength=1.5,
    )
    axes[-1].set_xlabel("post-onset time (ms)")
    fig.text(
        0.01,
        0.5,
        "E participation (fraction of $N_E$ firing per 2 ms bin)",
        rotation=90,
        ha="left",
        va="center",
        fontsize=theme.SIZE_LABEL,
        color=theme.INK,
    )

    fig.suptitle(
        "real (black) vs fitted-model sample (red dashed) across the ei sweep",
        fontsize=theme.SIZE_TITLE,
        y=0.98,
    )
    fig.tight_layout(rect=(0.04, 0.0, 1.0, 0.94))
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
    print("  rate-model validation (real vs simulated)…")
    plot_rate_model_validation(frames_data, figures / "rate_model_validation.png")
    return {
        "rates_hz": rates,
        "canonical_ei": CANON_EI,
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
