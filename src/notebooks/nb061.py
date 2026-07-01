"""Notebook runner for entry 061 — using the gamma rhythm as a readout clock.

Granting a clean PING gamma rhythm, can the self-generated tick be *used*
as a timing reference? (Whether the rhythm survives per-cell noise is a
separate reliability question, not the subject here.)

We reuse a competent, already-trained PING from nb041 (the tg6 cell:
tau_gaba = 6 ms, ≈90% on MNIST, gamma on the task) rather than retraining —
the rhythm is what it is once the network is trained. At inference the gamma volley is treated as the clock edge that *samples*
the output readout. Two in-network coincidence-detector neurons emit the
tick (models.py): an I tap on the inhibitory pool and an E tap on the
excitatory pool. Five schemes are compared on identical spikes and weights:

  continuous    — average the output over every timestep (the trained
                  readout; the expensive, always-on baseline);
  self-clock-i  — sample at the inhibitory-volley ticks (I-tap CD neuron);
  self-clock-e  — sample at the excitatory-volley ticks (E-tap CD neuron);
  fixed-clock   — sample at a FIXED period (an external crystal), phase-swept
                  and reported at its BEST phase to steelman it;
  shuffled      — sample at random times, matched count (the floor control).

Non-inferiority logic: the self-clock should MATCH continuous (good enough
to replace the always-on readout) at a fraction of the samples, must BEAT
shuffled (the timing has to carry information), and the comparison against
the best-phase fixed clock tests whether a crystal could stand in.

Per-step output is recovered from the recorded running mem-mean:
    out[t] = mem_sum[t] / T   ->   v_out[t] = T * (out[t] - out[t-1]).

Reuses nb041 artifacts: src/artifacts/notebooks/nb041/ping__tg6__seed42.
Figures + numbers.json land in /figures/notebooks/nb061/.

Notebook entry: src/docs/content/notebooks/nb061.mdx
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from helpers.fmt import format_duration  # noqa: E402
from helpers.modal import parse_modal_gpu  # noqa: E402
from helpers.paths import artifacts_and_figures  # noqa: E402
from helpers.run_dirs import prepare as prepare_run_dirs  # noqa: E402
from helpers.run_id import next_run_id  # noqa: E402
from helpers.stamp import stamp_figure  # noqa: E402
from helpers.tier import parse_tier  # noqa: E402
from helpers import theme  # noqa: E402

SLUG = "nb061"
ARTIFACTS, FIGURES = artifacts_and_figures(SLUG)

# Reused trained cell (nb041): a competent PING in gamma on MNIST.
# The tg6 PING cell now lives in the shared training root (nb022), not nb041.
NB041_CELL = REPO / "src" / "artifacts" / "notebooks" / "training" / "ping__tg6__seed42"
TAU_GABA_MS = 6.0
SEED = 42

TIER_CONFIG = {
    "extra small": dict(n_eval=100),
    "small": dict(n_eval=300),
    "medium": dict(n_eval=600),
    "large": dict(n_eval=1200),
    "extra large": dict(n_eval=2000),
}
DEFAULT_TIER = "small"

# Non-inferiority margin (accuracy points): the self-clock is "good enough to
# replace" the always-on readout if it falls within this of continuous.
NONINFERIORITY_MARGIN = 3.0

SCHEME_COLORS = {
    "continuous": theme.LABEL,
    "self-clock-i": theme.INK_BLACK,
    "self-clock-e": theme.AMBER,
    "fixed-clock": theme.DEEP_RED,
    "shuffled": theme.GREY_LIGHT,
}
SCHEME_ORDER = ["continuous", "self-clock-i", "self-clock-e", "fixed-clock", "shuffled"]
SCHEME_LABELS = {
    "continuous": "continuous",
    "self-clock-i": "self-clock\n(I tap)",
    "self-clock-e": "self-clock\n(E tap)",
    "fixed-clock": "fixed\nclock",
    "shuffled": "shuffled",
}


# ── Trained-net loader (inlined; adds the cell's tau_gaba) ────────────


def _load_trained(train_dir: Path, n_eval: int):
    """Load the trained PING + a fixed evaluation batch, with the cell's own
    tau_gaba wired into the synaptic decay so the rhythm is correct."""
    import torch

    import models as M
    from cli.config import build_net
    from cli import _auto_device, load_dataset, seed_everything

    cfg = json.loads((train_dir / "config.json").read_text())
    seed_everything(int(cfg.get("seed", SEED)))
    M.T_ms = float(cfg["t_ms"])
    M.dt = float(cfg["dt"])
    M.T_steps = int(M.T_ms / M.dt)
    # The rhythm-setting knob: GABA decay. forward() recomputes decay_gaba from
    # M.tau_gaba and M.dt each call, so M.tau_gaba is the knob that matters;
    # M.decay_gaba is set too for any direct reads.
    M.tau_gaba = TAU_GABA_MS
    M.decay_gaba = float(np.exp(-M.dt / TAU_GABA_MS))

    hidden_sizes = cfg.get("hidden_sizes") or [int(cfg["n_hidden"])]
    M.N_HID = hidden_sizes[-1]
    M.N_INH = hidden_sizes[-1] // 4
    M.HIDDEN_SIZES = list(hidden_sizes)

    device = _auto_device()
    _, X_te, _, y_te = load_dataset(
        cfg["dataset"], max_samples=int(cfg["max_samples"]), split=True
    )
    M.N_IN = 784 if cfg["dataset"] == "mnist" else 64
    n_eval = min(n_eval, X_te.shape[0])
    X_eval, y_eval = X_te[:n_eval], y_te[:n_eval]

    w_in_cfg = cfg.get("w_in")
    w_in_arg = (
        (float(w_in_cfg[0]), float(w_in_cfg[1]))
        if isinstance(w_in_cfg, list) and len(w_in_cfg) >= 2
        else None
    )
    net = build_net(
        cfg["model"],
        w_in=w_in_arg,
        w_in_sparsity=float(cfg.get("w_in_sparsity") or 0.0),
        ei_strength=float(cfg.get("ei_strength") or 1.0),
        ei_ratio=float(cfg.get("ei_ratio") or 2.0),
        sparsity=float(cfg.get("sparsity") or 0.0),
        device=device,
        randomize_init=not bool(cfg.get("kaiming_init", False)),
        dales_law=bool(cfg.get("dales_law", True)),
        hidden_sizes=hidden_sizes,
    )
    if hasattr(net, "readout_mode"):
        net.readout_mode = cfg.get("readout_mode", "mem-mean")
    state = torch.load(train_dir / "weights.pth", map_location=device)
    net.load_state_dict(state, strict=False)
    net.eval()
    return net, cfg, X_eval, y_eval, device


# ── Tick detection + readout schemes ─────────────────────────────────


def ticks_from_cd(tick_rec: np.ndarray) -> list[np.ndarray]:
    """Per-trial tick times (step indices) read straight off the in-network
    coincidence-detector neuron's spike train (models.py emits one channel,
    one spike per inhibitory volley)."""
    if tick_rec.ndim == 3:
        tick_rec = tick_rec[:, :, 0]
    return [np.flatnonzero(tick_rec[:, b]) for b in range(tick_rec.shape[1])]


def reconstruct_vout(out_run: np.ndarray, T: int) -> np.ndarray:
    """Per-step output contribution from the recorded running mem-mean."""
    v_out = np.empty_like(out_run)
    v_out[0] = out_run[0] * T
    v_out[1:] = (out_run[1:] - out_run[:-1]) * T
    return v_out


def logits_for(v_out: np.ndarray, idx_per_trial, T: int) -> np.ndarray:
    B = v_out.shape[1]
    out = np.zeros((B, v_out.shape[2]))
    for b in range(B):
        idx = idx_per_trial[b] if len(idx_per_trial[b]) else np.arange(T)
        out[b] = v_out[idx, b, :].mean(axis=0)
    return out


def accuracy(v_out, idx_per_trial, y, T) -> float:
    return float((logits_for(v_out, idx_per_trial, T).argmax(1) == y).mean()) * 100.0


# ── Figures ──────────────────────────────────────────────────────────


def _despine(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def plot_rhythm(e_tr, i_tr, freqs, psd, f_gamma, dt_ms, out_path, run_id):
    """Figure 1: the rhythm is real gamma — single-trial raster + PSD peak."""
    theme.apply()
    plt.rcParams["savefig.bbox"] = "standard"
    fig, (ax_r, ax_p) = plt.subplots(1, 2, figsize=(12.0, 6.75), dpi=150)
    n_e, n_i = e_tr.shape[1], i_tr.shape[1]
    t_axis = np.arange(e_tr.shape[0]) * dt_ms
    et, en = np.where(e_tr)
    it, inn = np.where(i_tr)
    ax_r.scatter(t_axis[et], en, s=1.4, c=theme.INK_BLACK, marker="|", linewidths=0.4)
    ax_r.scatter(t_axis[it], inn + n_e + 8, s=1.4, c=theme.DEEP_RED, marker="|", linewidths=0.4)
    ax_r.set_yticks([n_e / 2, n_e + 8 + n_i / 2])
    ax_r.set_yticklabels(["E", "I"])
    ax_r.tick_params(axis="y", length=0)
    ax_r.set_xlabel("time (ms)")
    ax_r.set_title("PING raster — E (black) / I (red)", loc="left", fontweight="semibold")
    _despine(ax_r)

    ax_p.plot(freqs, psd, color=theme.INK_BLACK, lw=1.4)
    ax_p.axvline(f_gamma, color=theme.DEEP_RED, ls="--", lw=1.0)
    ax_p.annotate(f"f_γ ≈ {f_gamma:.0f} Hz", xy=(f_gamma, 0.9),
                  xytext=(6, 0), textcoords="offset points",
                  fontsize=theme.SIZE_ANNOTATION, color=theme.DEEP_RED)
    ax_p.set_xlim(0, 120)
    ax_p.set_xlabel("frequency (Hz)")
    ax_p.set_ylabel("normalised power")
    ax_p.set_title("Population-rate spectrum", loc="left", fontweight="semibold")
    _despine(ax_p)

    fig.suptitle("The rhythm is real gamma", fontsize=theme.SIZE_TITLE)
    fig.tight_layout()
    stamp_figure(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_clock_gating(i_rate_tr, v_out_tr, ticks_tr, dt_ms, out_path, run_id):
    """Figure 2 (money shot): the volley clock gating the readout. Top — I
    population rate with volley ticks; bottom — output trace with the
    sample-and-hold points dropped on those ticks."""
    theme.apply()
    plt.rcParams["savefig.bbox"] = "standard"
    fig, (ax_i, ax_o) = plt.subplots(
        2, 1, figsize=(12.0, 6.75), dpi=150, sharex=True,
        gridspec_kw={"height_ratios": [1, 1.3], "hspace": 0.12},
    )
    t_axis = np.arange(len(i_rate_tr)) * dt_ms
    tick_ms = ticks_tr * dt_ms

    ax_i.plot(t_axis, i_rate_tr, color=theme.DEEP_RED, lw=1.0)
    for tm in tick_ms:
        ax_i.axvline(tm, color=theme.INK_BLACK, lw=0.7, alpha=0.5)
    ax_i.set_ylabel("I rate (a.u.)")
    ax_i.set_title("Inhibitory volleys = the clock edge", loc="left", fontweight="semibold")
    _despine(ax_i)

    # readout: the winning-class output trajectory.
    win = int(v_out_tr.sum(axis=0).argmax())
    trace = v_out_tr[:, win]
    ax_o.plot(t_axis, trace, color=theme.LABEL, lw=1.0, label="output (continuous)")
    for tm in tick_ms:
        ax_o.axvline(tm, color=theme.INK_BLACK, lw=0.7, alpha=0.5)
    ax_o.scatter(tick_ms, trace[ticks_tr], s=42, color=theme.INK_BLACK, zorder=5,
                 label="sampled on tick")
    ax_o.set_xlabel("time (ms)")
    ax_o.set_ylabel(f"output, class {win}")
    ax_o.set_title("Readout sampled on the gamma tick", loc="left", fontweight="semibold")
    ax_o.legend(frameon=False, fontsize=theme.SIZE_LEGEND, loc="upper right")
    _despine(ax_o)

    fig.suptitle("The gamma rhythm gating the readout", fontsize=theme.SIZE_TITLE)
    fig.tight_layout()
    stamp_figure(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_accuracy_bars(results, n_samples, cont_acc, margin, out_path, run_id):
    """Figure 3: accuracy across schemes with samples/trial annotated."""
    theme.apply()
    plt.rcParams["savefig.bbox"] = "standard"
    fig, ax = plt.subplots(figsize=(12.0, 6.75), dpi=150)
    order = SCHEME_ORDER
    xs = np.arange(len(order))
    accs = [results[k] for k in order]
    ax.bar(xs, accs, width=0.66,
           color=[SCHEME_COLORS[k] for k in order], edgecolor="none")
    ax.axhspan(cont_acc - margin, cont_acc, color=theme.LABEL, alpha=0.12)
    ax.axhline(cont_acc, color=theme.LABEL, ls="--", lw=1.0)
    for x, a, k in zip(xs, accs, order):
        ax.text(x, a + 1.0, f"{a:.1f}%", ha="center", fontsize=theme.SIZE_ANNOTATION)
        ax.text(x, 3, f"{n_samples[k]:g}\nsamp", ha="center", va="bottom",
                fontsize=theme.SIZE_ANNOTATION, color="white")
    ax.set_xticks(xs)
    ax.set_xticklabels([SCHEME_LABELS[k] for k in order])
    ax.set_ylabel("test accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title(
        "Self-clock matches the always-on readout at a fraction of the samples",
        loc="left", fontweight="semibold", fontsize=theme.SIZE_LABEL,
    )
    _despine(ax)
    fig.tight_layout()
    stamp_figure(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_freq_spread(f_per_trial, fixed_f, out_path, run_id):
    """Figure 4 (the clincher): per-trial gamma frequency spread vs the single
    rate a fixed external clock must commit to."""
    theme.apply()
    plt.rcParams["savefig.bbox"] = "standard"
    fig, ax = plt.subplots(figsize=(12.0, 6.75), dpi=150)
    ax.hist(f_per_trial, bins=24, color=theme.INK_BLACK, alpha=0.8)
    ax.axvline(fixed_f, color=theme.DEEP_RED, ls="--", lw=1.4)
    ax.annotate("fixed clock\n(one rate for all trials)",
                xy=(fixed_f, ax.get_ylim()[1] * 0.9),
                xytext=(8, 0), textcoords="offset points",
                fontsize=theme.SIZE_ANNOTATION, color=theme.DEEP_RED, va="top")
    ax.set_xlabel("per-trial gamma frequency f_γ (Hz)")
    ax.set_ylabel("trials")
    ax.set_title(
        "The rhythm's rate varies per input — a fixed clock can't track it",
        loc="left", fontweight="semibold", fontsize=theme.SIZE_LABEL,
    )
    _despine(ax)
    fig.tight_layout()
    stamp_figure(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────


def main() -> None:
    import torch

    import models as M
    from cli import encode_batch, metrics

    tier = parse_tier(sys.argv, choices=TIER_CONFIG.keys(), default=DEFAULT_TIER)
    _ = parse_modal_gpu(sys.argv)  # accepted for interface parity; run is local
    wipe_dir = "--no-wipe-dir" not in sys.argv
    n_eval = TIER_CONFIG[tier]["n_eval"]

    t_start = time.monotonic()
    run_id = next_run_id(SLUG)
    print(f"notebook_run_id = {run_id} tier={tier} n_eval={n_eval}")
    prepare_run_dirs(SLUG, run_id, wipe=wipe_dir, skip_training=True,
                     make_artifacts=False)

    if not (NB041_CELL / "weights.pth").exists():
        raise SystemExit(
            f"missing reused cell {NB041_CELL.relative_to(REPO)}; "
            "re-render nb041 (the tg6 cell) to produce it."
        )

    net, cfg, X_eval, y_eval, device = _load_trained(NB041_CELL, n_eval)
    net.recording = True
    dt_ms = float(cfg["dt"])
    T = int(round(float(cfg["t_ms"]) / dt_ms))

    enc_gen = torch.Generator().manual_seed(SEED)
    spk = encode_batch(torch.from_numpy(X_eval).to(device), M.dt, False, generator=enc_gen)
    torch.manual_seed(0)
    with torch.no_grad():
        _ = net(input_spikes=spk, noise_std=0.0)

    out_run = net.spike_record["out"].cpu().numpy()        # (T, B, 10)
    inh = net.spike_record["inh"].cpu().numpy()            # (T, B, n_i)
    hid = net.spike_record["hid"].cpu().numpy()            # (T, B, n_e)
    tick_i = net.spike_record["tick_i"].cpu().numpy()      # CD neuron, I tap
    tick_e = net.spike_record["tick_e"].cpu().numpy()      # CD neuron, E tap
    B = out_run.shape[1]
    v_out = reconstruct_vout(out_run, T)
    i_rate = inh.sum(axis=2)                                # (T, B)

    # The clock is emitted in-network by coincidence-detector neurons (models.py):
    # one taps the inhibitory pool (I), one the excitatory pool (E). Their spike
    # trains ARE the ticks — no offline peak-finding. The I tap is the canonical
    # clock (it lands on the output peak); use it for the rhythm/frequency views.
    ticks_i = ticks_from_cd(tick_i)
    ticks_e = ticks_from_cd(tick_e)
    ticks = ticks_i
    periods = [np.median(np.diff(t)) * dt_ms for t in ticks if len(t) > 1]
    n_ticks = np.array([len(t) for t in ticks])
    n_ticks_e = np.array([len(t) for t in ticks_e])

    # The E volley leads the I volley by the E→I conduction delay; the E-tap
    # tick therefore samples a touch before the output peak.
    e_lead = [
        (e - ticks_i[b][np.argmin(np.abs(ticks_i[b] - e))]) * dt_ms
        for b in range(B) if len(ticks_i[b]) and len(ticks_e[b])
        for e in ticks_e[b]
    ]
    e_leads_i_ms = round(float(np.median(e_lead)), 2) if e_lead else None
    f_per_trial = 1000.0 / np.array(periods)
    period_med_steps = int(np.median(periods) / dt_ms)
    fixed_f = 1000.0 / (period_med_steps * dt_ms)

    rng = np.random.default_rng(0)
    all_idx = [np.arange(T)] * B
    shuf = [np.sort(rng.choice(T, max(n_ticks[b], 1), replace=False)) for b in range(B)]

    # Phase-swept fixed clock: steelman the crystal by taking its best phase.
    best_fixed, best_phase = -1.0, 0
    for ph in range(0, period_med_steps, max(1, period_med_steps // 8)):
        a = accuracy(v_out, [np.arange(ph, T, period_med_steps)] * B, y_eval, T)
        if a > best_fixed:
            best_fixed, best_phase = a, ph

    results = {
        "continuous": accuracy(v_out, all_idx, y_eval, T),
        "self-clock-i": accuracy(v_out, ticks_i, y_eval, T),
        "self-clock-e": accuracy(v_out, ticks_e, y_eval, T),
        "fixed-clock": best_fixed,
        "shuffled": accuracy(v_out, shuf, y_eval, T),
    }
    n_samples = {
        "continuous": T,
        "self-clock-i": round(float(n_ticks.mean()), 1),
        "self-clock-e": round(float(n_ticks_e.mean()), 1),
        "fixed-clock": len(range(best_phase, T, period_med_steps)),
        "shuffled": round(float(n_ticks.mean()), 1),
    }
    for k in SCHEME_ORDER:
        print(f"  {k:<12} acc={results[k]:5.1f}%  samples/trial={n_samples[k]}")

    # Representative trial for the trace figures: a correctly-classified trial
    # with a near-median tick count and a clean rhythm.
    correct = logits_for(v_out, ticks, T).argmax(1) == y_eval
    cand = [b for b in range(B) if correct[b] and n_ticks[b] >= np.median(n_ticks)]
    rep = cand[0] if cand else int(np.argmax(n_ticks))

    # PSD averaged over a subset of trials for a clean peak.
    psd_acc, freqs = None, None
    for b in range(min(B, 64)):
        fr, ps = metrics.compute_psd(hid[:, b, :], M.N_HID, dt_ms, bin_ms=1.0,
                                     step_on_ms=20.0, step_off_ms=float(cfg["t_ms"]),
                                     burn_in_ms=0.0)
        psd_acc = ps if psd_acc is None else psd_acc + ps
        freqs = fr
    psd_mean = psd_acc / max(1, min(B, 64))
    psd_mean = psd_mean / psd_mean.max() if psd_mean.max() > 0 else psd_mean
    f_gamma = metrics.find_fundamental_nondiff(psd_mean, freqs, f_lo=5.0, f_hi=120.0)

    plot_rhythm(hid[:, rep, :].astype(bool), inh[:, rep, :].astype(bool),
                freqs, psd_mean, f_gamma, dt_ms, FIGURES / "rhythm.png", run_id)
    print(f"wrote {FIGURES / 'rhythm.png'}")
    plot_clock_gating(i_rate[:, rep], v_out[:, rep, :], ticks[rep], dt_ms,
                      FIGURES / "clock_gating.png", run_id)
    print(f"wrote {FIGURES / 'clock_gating.png'}")
    plot_accuracy_bars(results, n_samples, results["continuous"],
                       NONINFERIORITY_MARGIN, FIGURES / "accuracy_bars.png", run_id)
    print(f"wrote {FIGURES / 'accuracy_bars.png'}")
    plot_freq_spread(f_per_trial, fixed_f, FIGURES / "freq_spread.png", run_id)
    print(f"wrote {FIGURES / 'freq_spread.png'}")

    duration_s = time.monotonic() - t_start
    summary = {
        "notebook_run_id": run_id,
        "git_sha": cfg.get("git_sha"),
        "duration_s": round(duration_s, 1),
        "duration": format_duration(duration_s),
        "tier": tier,
        "config": {
            "tier": tier,
            "reused_cell": str(NB041_CELL.relative_to(REPO)),
            "tau_gaba_ms": TAU_GABA_MS,
            "dataset": cfg["dataset"],
            "t_ms": float(cfg["t_ms"]),
            "dt": dt_ms,
            "n_eval": int(B),
            "seed": SEED,
        },
        "f_gamma_hz": round(float(f_gamma), 1),
        "f_per_trial": {
            "median": round(float(np.median(f_per_trial)), 1),
            "iqr": [round(float(np.percentile(f_per_trial, 25)), 1),
                    round(float(np.percentile(f_per_trial, 75)), 1)],
            "range": [round(float(f_per_trial.min()), 1),
                      round(float(f_per_trial.max()), 1)],
        },
        "mean_ticks_per_trial": round(float(n_ticks.mean()), 1),
        "accuracy": {k: round(v, 1) for k, v in results.items()},
        "samples_per_trial": n_samples,
        "fixed_clock_best_phase_steps": best_phase,
        "e_leads_i_ms": e_leads_i_ms,
        "success_criteria": {
            # Primary clock is the I tap (lands on the output peak).
            "self_clock_replaces_continuous":
                bool(results["self-clock-i"] >= results["continuous"] - NONINFERIORITY_MARGIN),
            "self_clock_above_shuffled": bool(results["self-clock-i"] > results["shuffled"]),
            "self_clock_beats_fixed": bool(results["self-clock-i"] > best_fixed),
        },
    }
    (FIGURES / "numbers.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {FIGURES / 'numbers.json'}")
    print(f"  total duration: {summary['duration']}")


if __name__ == "__main__":
    main()
