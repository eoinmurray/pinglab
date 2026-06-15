"""Notebook runner for entry 002 — the spike movie.

A time-animated movie of the *actual spikes* (not a raster): the E
population laid out on a square grid, one video frame per simulation
window, each neuron white when silent and black when it fires —
binary, no fade.

The drive follows a three-phase protocol on one continuous run of the
*same* PING network:

    0.0–0.5 s   low uniform Poisson input   → sparse, asynchronous firing
    0.5–1.0 s   high uniform Poisson input  → PING gamma volleys (the grid
                                              flashes in rhythmic waves)
    1.0–1.5 s   low input again             → back to sparse / async

so you watch the recurrent E↔I loop switch on and off in real time.

Only `--tier` (sets the video frame stride: extra-large = one frame per
dt) and `--modal-gpu` are accepted; every other knob is hardcoded below.

Outputs spike_movie.mp4 + numbers.json (per-phase E/I rates and the
high-phase gamma peak) to src/docs/public/figures/notebooks/nb002/.

Notebook entry: src/docs/src/pages/notebooks/nb002.mdx
"""

from __future__ import annotations

import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from helpers.modal import parse_modal_gpu  # noqa: E402
from helpers.run_id import next_run_id, persist as persist_run_id  # noqa: E402
from helpers.tier import parse_tier  # noqa: E402
from cli import theme  # noqa: E402

SLUG = "nb002"
ARTIFACTS = REPO / "src" / "artifacts" / "notebooks" / SLUG
FIGURES = REPO / "src" / "docs" / "public" / "figures" / "notebooks" / SLUG

# ── Network + protocol (all hardcoded — the notebook IS the recipe) ──────────
N_E = 1024              # excitatory cells → 32×32 grid
GRID = int(round(math.sqrt(N_E)))   # 32
N_I = N_E // 4          # 256 → 16×16 grid
GRID_I = int(round(math.sqrt(N_I)))  # 16
N_IN = 784              # uniform input channels (matches the nb023 drive scale)
DT_MS = 0.1
PHASE_MS = 250.0        # each of the three phases (0.25 s)
EI_STRENGTH = 1.5       # recurrent loop ON (PING)
TAU_GABA_MS = 14.0      # slower GABA decay → cleaner, lower-freq gamma
W_IN_MEAN, W_IN_STD = 1.5, 0.3
LOW_RATE_HZ = 2.0       # per-channel Poisson rate, async phase (below PING onset)
HIGH_RATE_HZ = 30.0     # per-channel Poisson rate, PING phase (on, not overpowering)
SEED = 42

# Video frame stride (simulation steps binned into one rendered frame).
# extra large = 1 → literally one frame per dt.
TIER_STRIDE = {
    "extra small": 50,
    "small": 20,
    "medium": 10,
    "large": 5,
    "extra large": 1,
}
DEFAULT_TIER = "small"
FPS = 20
F_GAMMA_BAND_HZ = (5.0, 150.0)


def build_input(T_steps: int, generator) -> "object":
    """(T, N_IN) uniform-Poisson drive: low → high → low, one phase each."""
    import torch
    steps = int(round(PHASE_MS / DT_MS))
    rates = (
        [LOW_RATE_HZ] * steps
        + [HIGH_RATE_HZ] * steps
        + [LOW_RATE_HZ] * (T_steps - 2 * steps)
    )
    p = torch.tensor(rates, dtype=torch.float32).clamp(min=0.0) * DT_MS / 1000.0
    rand = torch.rand(T_steps, N_IN, generator=generator)
    return (rand < p[:, None]).float()


def run_simulation():
    """Build the PING net, run one forward pass on the three-phase drive,
    return (spk_e, spk_i) as (T, N) numpy arrays."""
    import torch

    import cli.config as C
    import models as M
    from cli.config import make_net, patch_dt
    from cli import _extract_records, primary_hid_key

    C.cfg.n_e = N_E
    C.cfg.n_i = N_I
    C.cfg.seed = SEED
    C.cfg.ei_strength = EI_STRENGTH
    C.cfg.sim_ms = 3 * PHASE_MS   # sizes the spike-record buffer (all 3 phases)
    C._sync_globals_from_cfg(C.cfg)
    M.N_IN = N_IN
    M.N_HID = N_E
    M.N_INH = N_I
    patch_dt(DT_MS)

    M.tau_gaba = TAU_GABA_MS
    M.decay_gaba = float(np.exp(-DT_MS / TAU_GABA_MS))   # used directly in forward

    T_steps = int(round(3 * PHASE_MS / DT_MS))
    M.sim_ms = 3 * PHASE_MS
    M.T_steps = T_steps   # COBANet.forward runs/records exactly M.T_steps steps
    gen = torch.Generator().manual_seed(SEED)
    input_spikes = build_input(T_steps, gen).to(C.DEVICE)

    net = make_net(
        C.cfg, w_in=(W_IN_MEAN, W_IN_STD, "normal", C.W_IN_SPARSITY),
        model_name="ping",
    )
    net.recording = True
    with torch.no_grad():
        net.forward(input_spikes=input_spikes)
    rec = _extract_records(net)

    def to_TN(x):
        a = np.asarray(x.detach().cpu() if hasattr(x, "detach") else x)
        if a.ndim == 3:           # (T, B, N) → drop batch
            a = a[:, 0, :]
        return a

    spk_e = to_TN(rec[primary_hid_key(rec)])
    spk_i = to_TN(rec["inh"]) if "inh" in rec else np.zeros((T_steps, N_I))
    return spk_e[:T_steps], spk_i[:T_steps]


def phase_label(t_ms: float) -> str:
    if t_ms < PHASE_MS:
        return f"async — low input ({LOW_RATE_HZ:.0f} Hz)"
    if t_ms < 2 * PHASE_MS:
        return f"PING — high input ({HIGH_RATE_HZ:.0f} Hz)"
    return f"async — low input ({LOW_RATE_HZ:.0f} Hz)"


def _input_rate_hz(T: int) -> np.ndarray:
    """Per-step per-channel Poisson input rate (Hz): low → high → low."""
    steps = int(round(PHASE_MS / DT_MS))
    r = np.full(T, LOW_RATE_HZ, dtype=np.float64)
    r[steps:2 * steps] = HIGH_RATE_HZ
    return r


def _rgba_layer(mat: np.ndarray, color) -> np.ndarray:
    """Binary spike matrix → RGBA: solid `color` where it spiked, transparent
    elsewhere (so layers overlay on white without one hiding the other)."""
    from matplotlib import colors as mcolors
    r, g, b = mcolors.to_rgb(color)
    out = np.zeros((*mat.shape, 4), dtype=np.float32)
    out[..., 0], out[..., 1], out[..., 2] = r, g, b
    out[..., 3] = np.clip(mat, 0.0, 1.0)
    return out


def render_movie(spk_e: np.ndarray, spk_i: np.ndarray, stride: int,
                 out_path: Path, run_id: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from matplotlib.colors import LinearSegmentedColormap

    T = spk_e.shape[0]
    starts = list(range(0, T, stride))
    # binary frames: a cell is filled if it spiked anywhere in the frame
    # window, blank otherwise — no fade, no intermediate greys.
    e_frames, i_frames, times_ms = [], [], []
    for s in starts:
        e_frames.append(spk_e[s:s + stride].max(axis=0).reshape(GRID, GRID).astype(np.float32))
        i_frames.append(spk_i[s:s + stride].max(axis=0).reshape(GRID_I, GRID_I).astype(np.float32))
        times_ms.append(s * DT_MS)
    print(f"  {len(e_frames)} frames (stride {stride} = {stride * DT_MS:.1f} ms/frame)")

    rate = _input_rate_hz(T)
    t_full_s = np.arange(T) * DT_MS / 1000.0
    t_end = float(t_full_s[-1])
    rate_max = HIGH_RATE_HZ * 1.15

    # full-run rasters, binned to 1 ms columns: gamma volleys → vertical bands
    tb = max(1, int(round(1.0 / DT_MS)))
    nb = T // tb
    raster_e = spk_e[:nb * tb].reshape(nb, tb, -1).max(axis=1).T  # (N_E, nb)
    raster_i = spk_i[:nb * tb].reshape(nb, tb, -1).max(axis=1).T  # (N_I, nb)

    red_cmap = LinearSegmentedColormap.from_list(
        "white_red", ["#ffffff", theme.DEEP_RED])

    # 16:9 canvas matching the nb003-family scan videos (22×12.375 in @120 dpi
    # → 2640×1484). Font sizes scaled up for the larger frame.
    fs_title, fs_label, fs_tick = 22, 17, 14
    theme.apply()
    fig = plt.figure(figsize=(22, 22 * 9 / 16), dpi=120)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 2.7], wspace=0.16)
    gleft = gs[0].subgridspec(2, 1, height_ratios=[1, 1], hspace=0.22)
    gright = gs[1].subgridspec(2, 1, height_ratios=[3.0, 1.0], hspace=0.14)
    eax = fig.add_subplot(gleft[0])    # E spike grid (black)
    iax = fig.add_subplot(gleft[1])    # I spike grid (red)
    sax = fig.add_subplot(gright[0])   # the E+I raster
    rax = fig.add_subplot(gright[1])   # the input-rate clock

    e_im = eax.imshow(e_frames[0], cmap="Greys", vmin=0, vmax=1,
                      interpolation="nearest")
    i_im = iax.imshow(i_frames[0], cmap=red_cmap, vmin=0, vmax=1,
                      interpolation="nearest")
    for a, lab in ((eax, f"E  ({N_E})"), (iax, f"I  ({N_I})")):
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_color(theme.GREY_MID)
        a.set_xlabel(lab, fontsize=fs_label,
                     color=theme.DEEP_RED if a is iax else theme.INK_BLACK)
    title = fig.suptitle("", color=theme.INK_BLACK, fontsize=fs_title,
                         family="monospace", y=0.97)

    # raster — neuron index (y) vs time (x); E black, I red, stacked
    sax.imshow(_rgba_layer(raster_e, theme.INK_BLACK), aspect="auto",
               origin="lower", interpolation="nearest",
               extent=[0, t_end, 0, N_E])
    sax.imshow(_rgba_layer(raster_i, theme.DEEP_RED), aspect="auto",
               origin="lower", interpolation="nearest",
               extent=[0, t_end, N_E, N_E + N_I])
    sax.axhline(N_E, color=theme.GREY_MID, lw=0.8)
    sax.set_xlim(0, t_end)
    sax.set_ylim(0, N_E + N_I)
    sax.set_ylabel("neuron (E | I)", fontsize=fs_label)
    sax.set_xticklabels([])
    sax.tick_params(labelsize=fs_tick)
    for sp in sax.spines.values():
        sp.set_color(theme.GREY_MID)
    scursor = sax.axvline(0.0, color=theme.DEEP_RED, lw=2.0)

    # input-rate clock — low → high → low square wave
    rax.axvspan(PHASE_MS / 1000.0, 2 * PHASE_MS / 1000.0,
                color=theme.GREY_MID, alpha=0.12, lw=0)
    rax.plot(t_full_s, rate, color=theme.GREY_MID, lw=1.8,
             drawstyle="steps-post")
    rax.set_xlim(0, t_end)
    rax.set_ylim(0, rate_max)
    rax.set_xlabel("time (s)", fontsize=fs_label)
    rax.set_ylabel("input rate (Hz)", fontsize=fs_label)
    rax.tick_params(labelsize=fs_tick)
    for sp in ("top", "right"):
        rax.spines[sp].set_visible(False)
    cursor = rax.axvline(0.0, color=theme.DEEP_RED, lw=2.0)   # the moving clock

    def update(i):
        e_im.set_data(e_frames[i])
        i_im.set_data(i_frames[i])
        title.set_text(f"t = {times_ms[i] / 1000:.3f} s   |   "
                       f"{phase_label(times_ms[i])}")
        x = times_ms[i] / 1000.0
        cursor.set_xdata([x, x])
        scursor.set_xdata([x, x])
        return e_im, i_im, title, cursor, scursor

    fig.subplots_adjust(left=0.06, right=0.985, top=0.91, bottom=0.09)
    anim = animation.FuncAnimation(fig, update, frames=len(e_frames), blit=False)
    writer = animation.FFMpegWriter(fps=FPS, bitrate=6000)
    anim.save(str(out_path), writer=writer, dpi=120)   # 22in×120 → 2640×1484
    plt.close(fig)


def _gamma_peak(pop_trace: np.ndarray) -> float | None:
    from scipy import signal as sp
    x = pop_trace.astype(np.float64)
    x = x - x.mean()
    if x.size < 8 or not np.any(x):
        return None
    fs = 1000.0 / DT_MS
    freqs, psd = sp.welch(x, fs=fs, nperseg=min(len(x), 4096), scaling="density")
    band = (freqs >= F_GAMMA_BAND_HZ[0]) & (freqs <= F_GAMMA_BAND_HZ[1])
    if not band.any() or psd[band].max() == 0:
        return None
    return float(freqs[band][int(np.argmax(psd[band]))])


def phase_stats(spk_e, spk_i) -> dict:
    steps = int(round(PHASE_MS / DT_MS))
    names = ["async_pre", "ping", "async_post"]
    bounds = [(0, steps), (steps, 2 * steps), (2 * steps, spk_e.shape[0])]
    out = {}
    for nm, (a, b) in zip(names, bounds):
        e = float(spk_e[a:b].mean()) * 1000.0 / DT_MS
        i = float(spk_i[a:b].mean()) * 1000.0 / DT_MS
        out[nm] = {"e_rate_hz": e, "i_rate_hz": i}
    out["ping"]["gamma_peak_hz"] = _gamma_peak(
        spk_e[steps:2 * steps].mean(axis=1)
    )
    return out


def main() -> None:
    modal_gpu = parse_modal_gpu(sys.argv)
    tier = parse_tier(sys.argv, choices=TIER_STRIDE.keys(), default=DEFAULT_TIER)
    stride = TIER_STRIDE[tier]
    run_id = next_run_id(SLUG)
    t0 = time.time()

    print(f"notebook_run_id = {run_id}  tier={tier}  stride={stride}"
          + ("  (modal-gpu ignored: nb002 renders locally)" if modal_gpu else ""))
    if "--no-wipe-dir" not in sys.argv:
        for d in (ARTIFACTS, FIGURES):
            if d.exists():
                shutil.rmtree(d)
            print(f"[wipe] {d}")
    FIGURES.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    spk_e, spk_i = run_simulation()
    stats = phase_stats(spk_e, spk_i)
    for nm, s in stats.items():
        extra = (f"  gamma={s['gamma_peak_hz']:.1f} Hz"
                 if s.get("gamma_peak_hz") else "")
        print(f"  {nm:<10} E={s['e_rate_hz']:6.2f} Hz  I={s['i_rate_hz']:6.2f} Hz{extra}")

    render_movie(spk_e, spk_i, stride, FIGURES / "spike_movie.mp4", run_id)
    print(f"  wrote {FIGURES / 'spike_movie.mp4'}")

    summary = {
        "slug": SLUG,
        "notebook_run_id": run_id,
        "tier": tier,
        "config": {
            "n_e": N_E, "n_i": N_I, "grid": GRID, "n_in": N_IN, "dt_ms": DT_MS,
            "phase_ms": PHASE_MS, "ei_strength": EI_STRENGTH,
            "tau_gaba_ms": TAU_GABA_MS,
            "low_rate_hz": LOW_RATE_HZ, "high_rate_hz": HIGH_RATE_HZ,
            "frame_stride": stride, "fps": FPS, "seed": SEED,
        },
        "phases": stats,
    }
    (FIGURES / "numbers.json").write_text(json.dumps(summary, indent=2) + "\n")
    persist_run_id(SLUG, run_id)
    print(f"  wrote {FIGURES / 'numbers.json'}")
    print(f"  total duration: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
