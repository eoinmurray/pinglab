"""Render the recorded four-coupling balanced-network state as a video."""

from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = np.load(ROOT / "run-ai-v1" / "snapshot.npz")
OUT = ROOT / "ping-ai-state-v1.mp4"
POSTER = ROOT / "ping-ai-state-v1.png"

DT = float(DATA["dt"])
E = DATA["spk_e"]
I = DATA["spk_i"]
GE = DATA["ge_e_1"].mean(axis=1)
GI = DATA["gi_e_1"].mean(axis=1)
T, N_E = E.shape
N_I = I.shape[1]
TIME = np.arange(T) * DT


def isi_cvs(spikes):
    values = []
    for cell in range(spikes.shape[1]):
        isi = np.diff(np.flatnonzero(spikes[:, cell]) * DT)
        if isi.size >= 3 and isi.mean() > 0:
            values.append(isi.std() / isi.mean())
    return np.asarray(values)


CV_E = isi_cvs(E)
E_RATE = E.mean() * 1000.0 / DT
I_RATE = I.mean() * 1000.0 / DT

# Fixed population geometry: E outside, I inside. This is a population map,
# not an anatomical embedding.
theta_e = np.linspace(0, 2 * np.pi, N_E, endpoint=False)
theta_i = np.linspace(0, 2 * np.pi, N_I, endpoint=False)
xy_e = np.c_[np.cos(theta_e), np.sin(theta_e)]
xy_i = 0.52 * np.c_[np.cos(theta_i), np.sin(theta_i)]

BG = "#f3efe6"
BLACK = "#20201e"
RED = "#b83a32"
GREY = "#aca79d"

fig = plt.figure(figsize=(12.8, 7.2), facecolor=BG)
gs = fig.add_gridspec(2, 2, width_ratios=(0.9, 1.6), height_ratios=(1.25, 0.75),
                      left=0.055, right=0.97, top=0.82, bottom=0.09,
                      wspace=0.16, hspace=0.28)
ax_net = fig.add_subplot(gs[:, 0])
ax_raster = fig.add_subplot(gs[0, 1])
ax_balance = fig.add_subplot(gs[1, 1])

fig.text(0.055, 0.945, "PING substrate · asynchronous-irregular candidate",
         color=BLACK, fontsize=20, weight="semibold", va="top")
fig.text(0.055, 0.902,
         f"four recurrent couplings · fixed fan-in K≈10 · independent E/I drive · "
         f"E {E_RATE:.1f} Hz · I {I_RATE:.1f} Hz · median E ISI-CV {np.median(CV_E):.2f}",
         color="#68645d", fontsize=10.5, va="top")

for ax in (ax_net, ax_raster, ax_balance):
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_color("#d3cec4")

ax_net.set_aspect("equal")
ax_net.set_xlim(-1.18, 1.18)
ax_net.set_ylim(-1.18, 1.18)
ax_net.axis("off")
ax_net.text(0, 1.12, "population state", ha="center", color=BLACK, fontsize=11)
ax_net.scatter(xy_e[:, 0], xy_e[:, 1], s=7, color=GREY, alpha=0.42, linewidths=0)
ax_net.scatter(xy_i[:, 0], xy_i[:, 1], s=11, color=RED, alpha=0.34, linewidths=0)
active_e = ax_net.scatter([], [], s=45, color=BLACK, alpha=0.9, linewidths=0)
active_i = ax_net.scatter([], [], s=58, color=RED, alpha=0.95, linewidths=0)
ax_net.text(0, -0.04, "I", ha="center", va="center", color=RED, fontsize=20, weight="bold")
ax_net.text(0, -1.12, "E ring · I core\nflashes are emitted spikes",
            ha="center", va="top", color="#68645d", fontsize=9)

WINDOW_MS = 220.0
ax_raster.set_ylim(N_E + N_I + 3, -3)
ax_raster.set_xlim(0, WINDOW_MS)
ax_raster.axhline(N_E - 0.5, color="#c9c4ba", lw=0.8)
ax_raster.set_yticks([N_E / 2, N_E + N_I / 2], ["E", "I"])
ax_raster.set_xlabel("rolling time window (ms)")
ax_raster.set_title("spike raster", loc="left", fontsize=11, color=BLACK)
raster_e = ax_raster.scatter([], [], s=3.0, color=BLACK, marker="|", linewidths=0.8)
raster_i = ax_raster.scatter([], [], s=3.2, color=RED, marker="|", linewidths=0.9)
cursor = ax_raster.axvline(WINDOW_MS, color="#79746c", lw=0.8, alpha=0.7)

ax_balance.set_xlim(0, WINDOW_MS)
ymax = max(float(GE.max()), float(GI.max())) * 1.12
ax_balance.set_ylim(0, ymax)
ax_balance.set_xlabel("rolling time window (ms)")
ax_balance.set_ylabel("mean conductance (µS)")
ax_balance.set_title("competing population conductances onto E", loc="left", fontsize=11, color=BLACK)
line_ge, = ax_balance.plot([], [], color=BLACK, lw=1.5, label="excitatory")
line_gi, = ax_balance.plot([], [], color=RED, lw=1.5, label="inhibitory")
ax_balance.legend(frameon=False, loc="upper right", ncol=2, fontsize=9)
time_label = fig.text(0.97, 0.945, "0 ms", ha="right", va="top", color=BLACK, fontsize=13)

FRAMES = 300
frame_steps = np.linspace(0, T - 1, FRAMES).astype(int)


def update(frame_index):
    step = frame_steps[frame_index]
    start = max(0, step - int(WINDOW_MS / DT))
    current_window = slice(start, step + 1)
    rel_time = TIME[current_window] - TIME[start]

    active_e.set_offsets(xy_e[E[step].astype(bool)])
    active_i.set_offsets(xy_i[I[step].astype(bool)])

    te, ce = np.nonzero(E[current_window])
    ti, ci = np.nonzero(I[current_window])
    raster_e.set_offsets(np.c_[te * DT, ce] if te.size else np.empty((0, 2)))
    raster_i.set_offsets(np.c_[ti * DT, N_E + ci] if ti.size else np.empty((0, 2)))
    edge = rel_time[-1] if rel_time.size else 0.0
    cursor.set_xdata([edge, edge])

    line_ge.set_data(rel_time, GE[current_window])
    line_gi.set_data(rel_time, GI[current_window])
    time_label.set_text(f"{TIME[step]:.0f} ms")
    return active_e, active_i, raster_e, raster_i, cursor, line_ge, line_gi, time_label


update(FRAMES - 1)
fig.savefig(POSTER, dpi=160, facecolor=BG)
movie = animation.FuncAnimation(fig, update, frames=FRAMES, interval=1000 / 30, blit=False)
movie.save(OUT, writer=animation.FFMpegWriter(fps=30, bitrate=4200), dpi=120)
plt.close(fig)
print(f"wrote {OUT}")
print(f"wrote {POSTER}")
