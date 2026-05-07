"""Notebook runner for entry 016 — ping metrics.

Sweeps the E→I coupling strength from 0 → 1 with input held flat
(stim-window overdrive 1×), walking the network from the async
baseline through the onset of gamma. For each frame of the resulting
video we extract three views of the trace:

1. **Population rate** — the smoothed E hidden-layer firing rate
   (2 ms bins, post-onset window). Visualised as a stacked-rows plot.
2. **Autocorrelation peak prominence** — a single-frame rhythmicity
   score derived from the FFT-autocorrelation of the post-onset rate.
3. **Bayes factor** — calibrated regime probability per frame from
   two raster-level generative models. Async = per-neuron gamma-
   renewal process Γ(k, θ); ping = shared periodic burst schedule
   (period 1/f, phase b₀) with Bernoulli(p) per-neuron participation
   and Gaussian(0, σ²) jitter. Both recipes were validated against
   the real raster at *ei* = 0 and *ei* = 0.8 in step 1a (see
   plot_raster_validation). Likelihood implementation pending —
   see Method > Bayes factor > Todo in the entry.

Also writes numbers.json with pre/stim/post E and I population rates
from an in-process replay at the canonical high-ei run, so the entry
can interpolate exact values.

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


def compute_validation_rasters() -> dict:
    """Pull real E rasters at ei=0 (async baseline) and ei=CANON_EI (ping)
    for the Bayes-factor model-validation step (Todo 1a). Each raster has
    shape (T_steps, N_E) and is stored alongside an estimated per-neuron
    mean rate (Hz) used to parameterise the matched simulation."""
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

    out: dict = {}
    for label, ei in [("async", 0.0), ("ping", CANON_EI)]:
        C.cfg.n_e = N_HIDDEN
        C.cfg.n_i = N_HIDDEN // 4
        C.cfg.sim_ms = EI_SIM_MS
        C.cfg.step_on_ms = EI_STEP_ON_MS
        C.cfg.step_off_ms = EI_STEP_OFF_MS
        C.cfg.seed = SEED
        C.cfg.w_ei = (ei, ei * 0.1)
        C.cfg.w_ie = (ei * C.cfg.ei_ratio, ei * C.cfg.ei_ratio * 0.1)
        C._sync_globals_from_cfg(C.cfg)

        M.N_HID = C.N_E
        M.N_INH = C.N_I
        M.T_ms = EI_SIM_MS
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
        T_steps, n_e = spk_e.shape
        mean_rate_hz = float(spk_e.sum() / (n_e * T_steps * DT_MS / 1000.0))
        out[label] = {"ei": float(ei), "raster": spk_e, "rate_hz": mean_rate_hz}
    return out


def _async_log_lik_marginal(spike_times_per_neuron: list, T_ms: float,
                            k_grid=(1.0, 2.0, 5.0, 10.0, 20.0, 50.0)) -> float:
    """Marginal log-likelihood of the *full* spike-train under the gamma-renewal
    model (forward-recurrence first-spike density + iid inner ISIs + censored
    post-last-spike survival), profile-θ at each grid k then mean-marginalise
    over k. Closed form for θ̂(k): ∂L/∂θ = 0 ⇒ θ̂ = ⟨τ⟩/k where ⟨τ⟩ is the mean
    of the inner ISIs across all neurons.

    For a stationary renewal process the boundary contributions are
        f_first(t₁) = S(t₁)/μ           (forward-recurrence density)
        S_cens(T - t_n) = Q(k, (T-t_n)/θ̂)  (right-censored survival)
    where μ = kθ and S(t) = Q(k, t/θ) is the gamma survival function.
    Including them is what puts the async log-likelihood on equal footing
    with the ping density (which scores all N_E·N_b pairs)."""
    from scipy.special import gammaln, gammaincc, logsumexp

    isis_ms_chunks = []
    first_spikes_ms = []
    last_gaps_ms = []
    for s in spike_times_per_neuron:
        if s.size == 0:
            continue
        first_spikes_ms.append(float(s[0]))
        last_gaps_ms.append(float(T_ms - s[-1]))
        if s.size >= 2:
            isis_ms_chunks.append(np.diff(s))
    if not isis_ms_chunks:
        return -np.inf
    isis_ms = np.concatenate(isis_ms_chunks)
    first_spikes_ms = np.asarray(first_spikes_ms, dtype=np.float64)
    last_gaps_ms = np.asarray(last_gaps_ms, dtype=np.float64)

    n_tau = isis_ms.size
    mean_isi = float(isis_ms.mean())
    mean_log_isi = float(np.log(isis_ms).mean())

    log_profiles = []
    for k in k_grid:
        theta_hat = mean_isi / k
        # Inner-ISI bulk: (k-1)⟨log τ⟩ - k - k log θ̂ - log Γ(k), times n_tau.
        ell_inner = n_tau * (
            (k - 1.0) * mean_log_isi
            - k
            - k * np.log(theta_hat)
            - float(gammaln(k))
        )
        # Boundary: first-spike forward-recurrence and post-last survival.
        log_q_first = np.log(np.clip(
            gammaincc(k, first_spikes_ms / theta_hat), 1e-300, None
        ))
        ell_first = float(log_q_first.sum()
                          - first_spikes_ms.size * np.log(k * theta_hat))
        log_q_last = np.log(np.clip(
            gammaincc(k, last_gaps_ms / theta_hat), 1e-300, None
        ))
        ell_last = float(log_q_last.sum())
        log_profiles.append(ell_inner + ell_first + ell_last)
    log_profiles = np.asarray(log_profiles, dtype=np.float64)
    return float(logsumexp(log_profiles) - np.log(len(k_grid)))


def _ping_log_lik_marginal(spike_times_ms: "np.ndarray", n_e: int, T_ms: float,
                           f_grid_hz=None, n_phase_grid: int = 16) -> float:
    """Marginal log-likelihood under the burst-MPP model, profile (p, σ²) at
    each (b₀, f) grid point then mean-marginalise over the discrete grid.
        p̂ = n_tot / (N_E·N_b)
        σ̂² = ⟨(t - b_{n*})²⟩
    Phase b₀ is gridded across [0, T_p); frequency f is gridded across the
    gamma band (default 25–75 Hz at 5 Hz steps)."""
    from scipy.special import logsumexp

    if f_grid_hz is None:
        f_grid_hz = np.arange(25.0, 76.0, 5.0)
    f_grid_hz = np.asarray(f_grid_hz, dtype=np.float64)
    n_tot = spike_times_ms.size
    if n_tot == 0:
        return -np.inf

    log_marg_per_f = []
    for f in f_grid_hz:
        T_p = 1000.0 / f
        b0_grid = np.linspace(0.0, T_p, n_phase_grid, endpoint=False)
        log_at_phase = []
        for b0 in b0_grid:
            n_b = int(np.floor((T_ms - b0) / T_p)) + 1
            if n_b <= 1:
                log_at_phase.append(-np.inf)
                continue
            burst_times = b0 + np.arange(n_b) * T_p
            burst_times = burst_times[burst_times < T_ms]
            if burst_times.size == 0:
                log_at_phase.append(-np.inf)
                continue

            # Each spike's nearest burst (binary search via searchsorted).
            idx_right = np.searchsorted(burst_times, spike_times_ms)
            idx_right = np.clip(idx_right, 1, burst_times.size - 1)
            idx_left = idx_right - 1
            d_left = np.abs(spike_times_ms - burst_times[idx_left])
            d_right = np.abs(spike_times_ms - burst_times[idx_right])
            d_min = np.minimum(d_left, d_right)

            n_pairs = n_e * burst_times.size
            p_hat = n_tot / n_pairs
            if p_hat <= 0.0 or p_hat >= 1.0:
                log_at_phase.append(-np.inf)
                continue
            sigma2 = float((d_min ** 2).mean())
            if sigma2 <= 0.0:
                log_at_phase.append(-np.inf)
                continue

            L_spike = (n_tot * (np.log(p_hat) - 0.5 * np.log(2 * np.pi * sigma2))
                       - 0.5 * float((d_min ** 2).sum()) / sigma2)
            L_skip = (n_pairs - n_tot) * np.log(1.0 - p_hat)
            log_at_phase.append(L_spike + L_skip)
        arr = np.asarray(log_at_phase, dtype=np.float64)
        if not np.isfinite(arr).any():
            log_marg_per_f.append(-np.inf)
        else:
            log_marg_per_f.append(
                float(logsumexp(arr) - np.log(n_phase_grid))
            )
    arr = np.asarray(log_marg_per_f, dtype=np.float64)
    if not np.isfinite(arr).any():
        return -np.inf
    return float(logsumexp(arr) - np.log(f_grid_hz.size))


def compute_log_bayes_factor(raster: "np.ndarray", dt_ms: float) -> dict:
    """Per-trial log Bayes factor between the ping and async generative models
    of *Method > Bayes factor* in the entry. raster has shape (T_steps, N_E).

    Returns a dict with the marginalised log-likelihoods, log-BF and the
    sigmoid-converted regime probability P(ping | D). All log quantities are
    in nats."""
    T_steps, n_e = raster.shape
    T_ms = T_steps * dt_ms

    spike_times_per_neuron = [np.flatnonzero(raster[:, j]) * dt_ms
                              for j in range(n_e)]
    spikes_flat = np.concatenate(spike_times_per_neuron) \
        if spike_times_per_neuron else np.empty(0, dtype=np.float64)
    n_isis = sum(max(s.size - 1, 0) for s in spike_times_per_neuron)

    log_p_async = _async_log_lik_marginal(spike_times_per_neuron, T_ms)
    log_p_ping = _ping_log_lik_marginal(spikes_flat, n_e, T_ms)
    log_bf = log_p_ping - log_p_async
    # Numerically stable sigmoid for huge |log_bf|.
    if not np.isfinite(log_bf):
        p_ping = float("nan")
    elif log_bf >= 0:
        p_ping = 1.0 / (1.0 + np.exp(-log_bf))
    else:
        e = np.exp(log_bf)
        p_ping = e / (1.0 + e)
    return {
        "log_p_async": log_p_async,
        "log_p_ping": log_p_ping,
        "log_bf": log_bf,
        "p_ping": float(p_ping),
        "n_spikes": int(spikes_flat.size),
        "n_isis": int(n_isis),
    }


def unit_test_bayes_factor() -> list[dict]:
    """Synthetic ground-truth raster check (Todo step 2). Generate rasters from
    each generative model at known parameters and report log-BF, p_ping for
    each. Expect strongly negative log-BF for async-renewal traces, strongly
    positive for burst-MPP traces, and near-zero for the borderline case
    where burst jitter approaches σ ≈ T_p / 4 (loses synchrony)."""
    n_e = 256
    T_ms = 250.0
    T_steps = int(T_ms / DT_MS)
    cases = [
        ("async, k=10, λ=200 Hz",
         simulate_async_renewal(n_e, T_steps, DT_MS, rate_hz=200.0,
                                shape_k=10.0, seed=1)),
        ("async, k=1 (Poisson), λ=15 Hz",
         simulate_async_renewal(n_e, T_steps, DT_MS, rate_hz=15.0,
                                shape_k=1.0, seed=2)),
        ("ping, f=50 Hz, σ=2 ms, λ=12 Hz",
         simulate_ping_burst(n_e, T_steps, DT_MS, rate_hz=12.0,
                             f_hz=50.0, jitter_ms=2.0, seed=3)),
        ("ping, f=70 Hz, σ=1.5 ms, λ=15 Hz",
         simulate_ping_burst(n_e, T_steps, DT_MS, rate_hz=15.0,
                             f_hz=70.0, jitter_ms=1.5, seed=4)),
        ("borderline ping, f=50 Hz, σ=5 ms (T_p/4)",
         simulate_ping_burst(n_e, T_steps, DT_MS, rate_hz=12.0,
                             f_hz=50.0, jitter_ms=5.0, seed=5)),
    ]
    results = []
    for label, raster in cases:
        out = compute_log_bayes_factor(raster, DT_MS)
        out["label"] = label
        results.append(out)
        print(f"  {label:42s}  log_BF={out['log_bf']:+9.1f}  "
              f"p_ping={out['p_ping']:.3f}")
    return results


def simulate_async_renewal(n_e: int, T_steps: int, dt_ms: float,
                           rate_hz: float, shape_k: float = 10.0,
                           seed: int = SEED):
    """Sample from the async generative model (Method > Bayes factor):
    per-neuron gamma-renewal process with shape k and scale θ chosen so
    mean rate = rate_hz. Each neuron j gets a uniform initial phase
    u_j ~ U(0, kθ) and ISIs τ_{j,i} ~ Γ(k, θ) iid. CV(ISI) = 1/√k;
    k = 10 (default) matches the visible ei=0 stripe pattern."""
    rng = np.random.default_rng(seed)
    T_ms = T_steps * dt_ms
    mean_isi_ms = 1000.0 / max(rate_hz, 1e-9)
    scale_ms = mean_isi_ms / shape_k
    spikes = np.zeros((T_steps, n_e), dtype=np.uint8)
    for j in range(n_e):
        t = rng.uniform(0.0, mean_isi_ms)
        while t < T_ms:
            idx = int(t / dt_ms)
            if 0 <= idx < T_steps:
                spikes[idx, j] = 1
            t += rng.gamma(shape_k, scale_ms)
    return spikes


def simulate_ping_burst(n_e: int, T_steps: int, dt_ms: float,
                        rate_hz: float, f_hz: float,
                        jitter_ms: float = 2.0, seed: int = SEED):
    """Sample from the ping generative model (Method > Bayes factor):
    shared periodic burst schedule b_n = b_0 + n/f with global phase
    b_0 ~ U(0, 1/f). For each (j, n), neuron j participates with
    probability p = rate_hz/f_hz; if it does, fires once at b_n + ε
    with ε ~ N(0, σ²), σ = jitter_ms. Per-neuron mean rate = p·f
    matches rate_hz by construction. Captures the population-
    synchronous burst alignment of real PING."""
    rng = np.random.default_rng(seed)
    T_ms = T_steps * dt_ms
    period_ms = 1000.0 / max(f_hz, 1e-9)
    p_part = min(1.0, rate_hz / max(f_hz, 1e-9))
    phase_offset = rng.uniform(0.0, period_ms)
    burst_times = phase_offset + np.arange(0, int(T_ms / period_ms) + 2) * period_ms
    burst_times = burst_times[burst_times < T_ms]
    spikes = np.zeros((T_steps, n_e), dtype=np.uint8)
    for tb in burst_times:
        participates = rng.random(n_e) < p_part
        jitters = rng.normal(0.0, jitter_ms, n_e)
        for j in np.flatnonzero(participates):
            idx = int((tb + jitters[j]) / dt_ms)
            if 0 <= idx < T_steps:
                spikes[idx, j] = 1
    return spikes


def plot_raster_validation(real_rasters: dict, autocorr_data: dict,
                           out_path: Path) -> Path:
    """2×2 raster sanity check: real vs simulated at ei=0 (async) and ei=CANON_EI (ping).
    Cosmetic check that the generative recipes can produce something that
    looks like the real data; the autocorr-shape match is the main test."""
    from matplotlib.patches import Rectangle
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import theme  # type: ignore[import]

    theme.apply()

    # Estimate ping model parameters from real ei=CANON_EI data. Frequency
    # comes from the autocorr peak lag at the matching ei value.
    lam_async = real_rasters["async"]["rate_hz"]
    lam_ping = real_rasters["ping"]["rate_hz"]
    target_ei = real_rasters["ping"]["ei"]
    peak_lag_ms = None
    if autocorr_data is not None:
        # Find the autocorr frame closest to CANON_EI.
        best = min(autocorr_data["frames"],
                   key=lambda r: abs(r["ei_strength"] - target_ei))
        peak_lag_ms = best["peak_lag_ms"]
    f_hz = 1000.0 / peak_lag_ms if (peak_lag_ms and peak_lag_ms > 0) else 50.0
    A = 0.6  # rough; tune in step 1's autocorr-shape comparison

    real_async = real_rasters["async"]["raster"]
    real_ping = real_rasters["ping"]["raster"]
    T_steps, n_e = real_async.shape
    sim_async = simulate_async_renewal(n_e, T_steps, DT_MS, lam_async)
    sim_ping = simulate_ping_burst(n_e, T_steps, DT_MS, lam_ping, f_hz)

    n_show = min(80, n_e)
    panels = [
        ("real · ei = 0",                                              real_async[:, :n_show]),
        (f"sim · async (renewal), λ = {lam_async:.1f} Hz",             sim_async[:, :n_show]),
        (f"real · ei = {target_ei:.1f}",                               real_ping[:, :n_show]),
        (f"sim · ping (burst), λ = {lam_ping:.1f}, f = {f_hz:.0f} Hz", sim_ping[:, :n_show]),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 5.625), sharex=True, sharey=True)
    axes_flat = axes.ravel()
    for ax, (title, raster) in zip(axes_flat, panels):
        t_idx, n_idx = np.where(raster > 0)
        ax.scatter(t_idx * DT_MS, n_idx, s=0.6,
                   color=theme.INK_BLACK, alpha=0.7, marker=".")
        ax.set_xlim(0, T_steps * DT_MS)
        ax.set_ylim(-1, n_show)
        ax.set_title(title, fontsize=theme.SIZE_ANNOTATION, loc="left",
                     color=theme.INK)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("time (ms)")
    for ax in axes[:, 0]:
        ax.set_ylabel("neuron")

    fig.suptitle(
        "raster sanity check: real vs simulated at ei = 0 and "
        f"ei = {target_ei:.1f}",
        fontsize=theme.SIZE_TITLE, y=0.98,
    )
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.93))
    pad_x, pad_y = 0.01, 0.01
    bboxes = [ax.get_position() for ax in axes_flat]
    x0 = min(b.x0 for b in bboxes) - pad_x
    x1 = max(b.x1 for b in bboxes) + pad_x
    y0 = min(b.y0 for b in bboxes) - pad_y
    y1 = max(b.y1 for b in bboxes) + pad_y
    fig.patches.append(Rectangle(
        (x0, y0), x1 - x0, y1 - y0, transform=fig.transFigure,
        fill=False, edgecolor=theme.INK_BLACK, linewidth=1.0,
    ))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → {out_path}")
    return out_path


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
    print("  raster validation (Bayes-factor step 1a)…")
    real_rasters = compute_validation_rasters()
    plot_raster_validation(real_rasters, autocorr_data,
                           figures / "raster_validation.png")
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
