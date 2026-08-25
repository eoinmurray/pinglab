"""EXP098: simulate and render three views of recurrent PING recruitment."""

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools" / "snn"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from execution import ExecutionSpec, simulate  # noqa: E402
from tools import snnlang as snn  # noqa: E402, TID251

from helpers.numbers import write_numbers  # noqa: E402
from helpers.run_dirs import published_run  # noqa: E402
from helpers.run_id import next_run_id  # noqa: E402

SLUG = "exp098"
CONDITIONS = ("baseline", "w-ei-zero", "input-ramp")
DT_MS = 0.25
DURATION_MS = 100.0
SEED = 7
N_E = 100
N_I = 25
INPUT_RATE_HZ = 60.0
SCALE = {
    "dt_ms": DT_MS,
    "t_ms": DURATION_MS,
    "n_input": N_E,
    "n_e": N_E,
    "n_i": N_I,
    "seed": SEED,
    "baseline_input_hz": INPUT_RATE_HZ,
    "ramp_input_hz": [20.0, 160.0],
}


def author_network(ablate_e_to_i: bool) -> snn.Bundle:
    net = snn.Network("exp098_ping_dynamics", dt=DT_MS * snn.ms)
    drive = net.input(
        "drive", shape=("time", "batch", N_E), signal_type="spikes", unit="spike"
    )
    cell = snn.components.ping(net, name="ping", n_e=N_E, n_i=N_I, source=drive)
    for projection in net.projections:
        if projection["connection"] == "recurrent":
            projection["delay"] = {"value": DT_MS, "unit": "ms"}
    if ablate_e_to_i:
        projection = next(row for row in net.projections if row["id"] == "ping_E_to_I")
        projection["enabled"] = False
    net.expose(cell.E.spikes, cell.I.spikes, name="population")
    return snn.compile(net, target="tools/snnsim")


def simulate_condition(condition: str, output_root: Path) -> None:
    run_dir = output_root / "conditions" / condition
    run_dir.mkdir(parents=True, exist_ok=True)
    n_steps = round(DURATION_MS / DT_MS)
    if condition == "input-ramp":
        input_rate_hz = np.linspace(20.0, 160.0, n_steps, dtype=np.float32)
    else:
        input_rate_hz = np.full(n_steps, INPUT_RATE_HZ, dtype=np.float32)
    rng = np.random.default_rng(SEED + 1)
    input_spikes = (
        rng.random((n_steps, 1, N_E), dtype=np.float32)
        < input_rate_hz[:, None, None] * DT_MS / 1_000.0
    ).astype(np.uint8)
    bundle = author_network(condition == "w-ei-zero")
    result = simulate(
        ExecutionSpec(
            kind="simulate",
            executor="graph",
            graph=bundle.graph,
            inputs={"drive": torch.from_numpy(input_spikes).float()},
            seed=SEED,
            recording="full",
        )
    )
    bundle.write(output_root / f"network-{condition}.bundle", visualise=True)
    e_spikes = result.recordings["population_0"].cpu().numpy()[:, 0].astype(np.uint8)
    i_spikes = result.recordings["population_1"].cpu().numpy()[:, 0].astype(np.uint8)
    v_e = result.recordings["ping_E.voltage"].cpu().numpy()[:, 0]
    v_i = result.recordings["ping_I.voltage"].cpu().numpy()[:, 0]
    g_e = result.recordings["ping_input.conductance"].cpu().numpy()[:, 0]
    g_i = result.recordings["ping_I_to_E.conductance"].cpu().numpy()[:, 0]
    w_ei = result.parameters.get("ping_E_to_I.weight")
    w_ei_array = (
        np.zeros((N_E, N_I), dtype=np.float32) if w_ei is None else w_ei.cpu().numpy()
    )
    w_ie_array = result.parameters["ping_I_to_E.weight"].cpu().numpy()
    e_t, e_cell = np.nonzero(e_spikes)
    i_t, i_cell = np.nonzero(i_spikes)
    np.savez_compressed(
        run_dir / "rasters.npz",
        e_t=e_t.astype(np.int32),
        e_cell=e_cell.astype(np.int32),
        i_t=i_t.astype(np.int32),
        i_cell=i_cell.astype(np.int32),
        dt=np.float32(DT_MS),
        T=np.int32(n_steps),
        n_e=np.int32(N_E),
        n_i=np.int32(N_I),
    )
    np.savez_compressed(
        run_dir / "recurrent-weights.npz", w_ei=w_ei_array, w_ie=w_ie_array
    )
    np.savez_compressed(
        run_dir / "voltage-traces.npz",
        v_e=v_e,
        v_i=v_i,
        g_e=g_e,
        g_i=g_i,
        input_spikes=input_spikes[:, 0],
        input_rate_hz=input_rate_hz,
        dt_ms=np.float32(DT_MS),
        resting_mv=np.float32(-65.0),
        threshold_mv=np.float32(-50.0),
    )
    print(f"{condition}: {len(e_t)} E spikes, {len(i_t)} I spikes -> {run_dir}")


def summarize(output_root: Path, condition: str) -> dict:
    root = output_root / "conditions" / condition
    raster = np.load(root / "rasters.npz")
    voltage = np.load(root / "voltage-traces.npz")
    return {
        "e_spikes": int(len(raster["e_t"])),
        "i_spikes": int(len(raster["i_t"])),
        "input_spikes": int(voltage["input_spikes"].sum()),
        "input_rate_start_hz": float(voltage["input_rate_hz"][0]),
        "input_rate_end_hz": float(voltage["input_rate_hz"][-1]),
    }


def run_subprocess(action: str, condition: str, output_root: Path) -> None:
    env = os.environ | {
        "EXP098_ACTION": action,
        "PING_CONDITION": condition,
        "PING_OUTPUT_ROOT": str(output_root),
    }
    subprocess.run([sys.executable, __file__], cwd=REPO, env=env, check=True)


def orchestrate() -> None:
    started = time.monotonic()
    run_id = next_run_id(SLUG)
    print(f"notebook_run_id = {run_id}")
    with published_run(SLUG, run_id, scale=SCALE) as (_scratch, staging):
        results = {}
        for condition in CONDITIONS:
            run_subprocess("simulate", condition, staging)
            results[condition] = summarize(staging, condition)
        for condition in ("baseline", "w-ei-zero"):
            shutil.copy2(
                staging / f"network-{condition}.bundle" / "reports" / "circuit.svg",
                staging / f"network-{condition}.svg",
            )
        for condition in CONDITIONS:
            run_subprocess("render", condition, staging)
            generated = staging / f"ping-network-dynamics-v29-{condition}.mp4"
            generated.replace(staging / f"{condition}.mp4")
        write_numbers(
            staging,
            run_id=run_id,
            duration_s=time.monotonic() - started,
            payload={
                "question": "How does recurrent E-to-I recruitment distinguish PING-like activity from an E-only network as input drive rises?",
                "results": results,
                "disposition": "revise",
            },
        )
        (staging / "SCIENTIFIC_COLLECTION_STATE.md").write_text(
            f"""# Exp098 ScientificCollectionState

- Collection: `demo`
- Lifecycle: executed `ExpScout`
- Official local run: `{run_id}`
- Scientific role: exploratory simulation demonstration
- Implementation: `experiments/exp098.py`
- Conditions: baseline, E-to-I ablation, and 20--160 Hz input ramp
- Evidence: three simulated network-dynamics videos and `numbers.json`
- Limitation: one seed, one short simulation, visual PING classification only
- Disposition: revise before any durable study
- Publication, archival, and campaign activation remain separate gates.
"""
        )
    print(f"exp098 complete: {run_id}")


ACTION = os.environ.get("EXP098_ACTION")
if ACTION is None:
    orchestrate()
    raise SystemExit(0)
if ACTION == "simulate":
    simulate_condition(
        os.environ["PING_CONDITION"], Path(os.environ["PING_OUTPUT_ROOT"])
    )
    raise SystemExit(0)
if ACTION != "render":
    raise ValueError(f"unknown EXP098_ACTION: {ACTION}")
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba

ROOT = Path(os.environ["PING_OUTPUT_ROOT"]).resolve()
CONDITION = os.environ.get("PING_CONDITION", "baseline")
if CONDITION not in {"baseline", "w-ei-zero", "input-ramp"}:
    raise ValueError(f"unknown PING_CONDITION: {CONDITION}")
RUN_DIR = ROOT / "conditions" / CONDITION
RASTER = RUN_DIR / "rasters.npz"
FPS = 20 if CONDITION == "input-ramp" else 25

data = np.load(RASTER)
weight_data = np.load(RUN_DIR / "recurrent-weights.npz")
voltage_data = np.load(RUN_DIR / "voltage-traces.npz")
w_ei = weight_data["w_ei"]
w_ie = weight_data["w_ie"]
has_ei_projection = bool(np.any(w_ei > 0.0))
has_ie_projection = bool(np.any(w_ie > 0.0))
v_e = voltage_data["v_e"]
v_i = voltage_data["v_i"]
g_e = voltage_data["g_e"]
g_i = voltage_data["g_i"]
mean_v_e = v_e.mean(axis=1)
mean_v_i = v_i.mean(axis=1)
mean_g_e = g_e.mean(axis=1)
mean_g_i = g_i.mean(axis=1)
v_rest = float(voltage_data["resting_mv"])
v_threshold = float(voltage_data["threshold_mv"])
dt = float(data["dt"])
n_steps = int(data["T"])
n_e = int(data["n_e"])
n_i = int(data["n_i"])

# Fixed spatial layout: 10x10 input and E grids, plus a centred 5x5 I grid.
input_x, input_y = np.meshgrid(
    np.linspace(0.300, 0.420, 10), np.linspace(0.435, 0.715, 10)
)
e_x, e_y = np.meshgrid(np.linspace(0.500, 0.700, 10), np.linspace(0.420, 0.730, 10))
i_x, i_y = np.meshgrid(np.linspace(0.800, 0.940, 5), np.linspace(0.465, 0.685, 5))
input_xy = np.column_stack([input_x.ravel(), input_y.ravel()])
e_xy = np.column_stack([e_x.ravel(), e_y.ravel()])
i_xy = np.column_stack([i_x.ravel(), i_y.ravel()])

input_spikes = voltage_data["input_spikes"].astype(bool)
input_rate_hz = voltage_data["input_rate_hz"]

e_by_t = [[] for _ in range(n_steps)]
i_by_t = [[] for _ in range(n_steps)]
for step, cell in zip(data["e_t"], data["e_cell"]):
    if 0 <= int(step) < n_steps:
        e_by_t[int(step)].append(int(cell))
for step, cell in zip(data["i_t"], data["i_cell"]):
    if 0 <= int(step) < n_steps:
        i_by_t[int(step)].append(int(cell))

# Reconstruct the recurrent conductance carried by every individual synapse.
# The legacy PING step consumes the previous timestep's spikes, then applies
# exponential AMPA/GABA decay before updating the target neurons.
e_spikes = np.zeros((n_steps, n_e), dtype=np.float32)
i_spikes = np.zeros((n_steps, n_i), dtype=np.float32)
e_spikes[data["e_t"], data["e_cell"]] = 1.0
i_spikes[data["i_t"], data["i_cell"]] = 1.0
g_ei = np.zeros((n_steps, n_e, n_i), dtype=np.float32)
g_ie = np.zeros((n_steps, n_i, n_e), dtype=np.float32)
decay_ampa = np.exp(-dt / 2.0)
decay_gaba = np.exp(-dt / 9.0)
for step in range(1, n_steps):
    g_ei[step] = g_ei[step - 1] * decay_ampa + e_spikes[step - 1, :, None] * w_ei
    g_ie[step] = g_ie[step - 1] * decay_gaba + i_spikes[step - 1, :, None] * w_ie
g_ei_max = max(float(g_ei.max()), 1e-12)
g_ie_max = max(float(g_ie.max()), 1e-12)
e_segments = np.stack(
    [
        np.repeat(e_xy, n_i, axis=0),
        np.tile(i_xy, (n_e, 1)),
    ],
    axis=1,
)
i_segments = np.stack(
    [
        np.repeat(i_xy, n_e, axis=0),
        np.tile(e_xy, (n_i, 1)),
    ],
    axis=1,
)

fig, ax = plt.subplots(figsize=(10, 5.625), dpi=120)
fig.patch.set_facecolor("#f3efe6")
ax.set_facecolor("#f3efe6")
plt.rcParams.update({"font.family": "DejaVu Sans"})
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

input_nodes = ax.scatter(
    input_xy[:, 0],
    input_xy[:, 1],
    s=5,
    facecolors="#f3efe6",
    edgecolors="#77726a",
    linewidths=0.45,
    zorder=2,
)
e_nodes = ax.scatter(e_xy[:, 0], e_xy[:, 1], s=8, color="#20201e", alpha=0.82, zorder=2)
i_nodes = ax.scatter(i_xy[:, 0], i_xy[:, 1], s=8, color="#a62a24", alpha=0.88, zorder=2)
ax.text(
    0.05,
    0.945,
    "PING network dynamics",
    ha="left",
    va="top",
    color="#1d1d1b",
    fontsize=16.5,
    weight="medium",
)
condition_label = {
    "baseline": "baseline",
    "w-ei-zero": r"$W_{EI}=0$",
    "input-ramp": "input ramp · 20→160 Hz",
}[CONDITION]
ax.text(
    0.05,
    0.902,
    condition_label,
    ha="left",
    va="top",
    color="#69655e",
    fontsize=8.2,
    weight="medium",
)
ax.text(
    0.135,
    0.785,
    "Population means",
    ha="center",
    color="#4f4c47",
    fontsize=8.8,
    weight="medium",
)
input_label = ax.text(
    0.360, 0.785, "", ha="center", color="#625f59", fontsize=8.8, weight="medium"
)
ax.text(
    0.600,
    0.785,
    "Excitatory  ·  100",
    ha="center",
    color="#20201e",
    fontsize=9.2,
    weight="medium",
)
ax.text(
    0.870,
    0.785,
    "Inhibitory  ·  25",
    ha="center",
    color="#a62a24",
    fontsize=9.2,
    weight="medium",
)
time_text = ax.text(
    0.95,
    0.945,
    "",
    ha="right",
    va="top",
    color="#20201e",
    fontsize=11.5,
    family="monospace",
)
count_text = ax.text(
    0.95, 0.900, "", ha="right", va="top", color="#4f4c47", fontsize=7.8
)

# Input program lives with the Input population, rather than in the global
# title/status band. The full schedule stays faint; elapsed time and the
# current-frame marker carry the animation state.
rate_time_ms = np.arange(n_steps, dtype=np.float32) * dt
rate_ax = ax.inset_axes([0.285, 0.815, 0.150, 0.060], transform=ax.transAxes)
rate_ax.set_facecolor("none")
for spine in rate_ax.spines.values():
    spine.set_visible(False)
rate_ax.set_xlim(0.0, max(float(rate_time_ms[-1]), dt))
rate_axis_max = max(160.0, float(input_rate_hz.max()))
rate_ax.set_ylim(0.0, rate_axis_max * 1.06)
rate_ax.set_xticks([])
rate_ax.set_yticks([])
rate_ax.plot(rate_time_ms, input_rate_hz, color="#c9c3b9", linewidth=1.0, zorder=1)
(rate_elapsed,) = rate_ax.plot([], [], color="#4f4c47", linewidth=1.35, zorder=2)
(rate_cursor,) = rate_ax.plot(
    [], [], color="#a62a24", linewidth=0.75, alpha=0.55, zorder=2
)
(rate_point,) = rate_ax.plot(
    [],
    [],
    marker="o",
    markersize=4.2,
    color="#a62a24",
    markeredgecolor="#f3efe6",
    markeredgewidth=0.65,
    zorder=3,
)
rate_ax.text(
    0.0,
    0.96,
    "input rate",
    transform=rate_ax.transAxes,
    ha="left",
    va="top",
    color="#77726a",
    fontsize=6.4,
)
rate_value_text = rate_ax.text(
    1.0,
    0.96,
    "",
    transform=rate_ax.transAxes,
    ha="right",
    va="top",
    color="#4f4c47",
    fontsize=6.4,
    family="monospace",
)
rate_artists = [rate_elapsed, rate_cursor, rate_point, rate_value_text]

# Population means, rendered as a compact vertical instrument bank.


def make_piston(x, label, color):
    y, width, height = 0.445, 0.016, 0.245
    ax.add_patch(
        plt.Rectangle(
            (x - width / 2, y),
            width,
            height,
            facecolor="#ebe7df",
            edgecolor="#aaa59c",
            linewidth=0.75,
            zorder=1,
        )
    )
    ax.plot(
        [x, x],
        [y - 0.008, y + height + 0.008],
        color="#aaa59c",
        linewidth=0.65,
        zorder=1,
    )
    fill = ax.add_patch(
        plt.Rectangle(
            (x - width / 2 + 0.003, y),
            width - 0.006,
            0.001,
            facecolor=color,
            edgecolor="none",
            alpha=0.20,
            zorder=2,
        )
    )
    fill._piston_travel = height - 0.010
    fill._piston_y = y
    head = ax.add_patch(
        plt.Rectangle(
            (x - width / 2 - 0.005, y),
            width + 0.010,
            0.010,
            facecolor=color,
            edgecolor="#f3efe6",
            linewidth=0.55,
            zorder=3,
        )
    )
    ax.text(
        x,
        0.722,
        label,
        ha="center",
        va="center",
        color=color,
        fontsize=7.7,
        weight="medium",
    )
    value = ax.text(
        x,
        0.414,
        "",
        ha="center",
        va="center",
        color=color,
        fontsize=6.3,
        family="monospace",
    )
    return fill, head, value


piston_ge = make_piston(0.060, r"$g_E$ µS", "#20201e")
piston_gi = make_piston(0.105, r"$g_I$ µS", "#a62a24")
piston_ve = make_piston(0.165, r"$V_E$ mV", "#20201e")
piston_vi = make_piston(0.210, r"$V_I$ mV", "#a62a24")
piston_artists = [
    artist
    for piston in (piston_ve, piston_vi, piston_ge, piston_gi)
    for artist in piston
]
comparison_voltage = [
    np.load(path / "voltage-traces.npz")
    for path in (ROOT / "conditions").iterdir()
    if (path / "voltage-traces.npz").exists()
]
conductance_scale_max = max(
    [
        float(trace[key].mean(axis=1).max())
        for trace in comparison_voltage
        for key in ("g_e", "g_i")
    ]
    + [1e-12]
)
voltage_scale_min = min(
    float(trace[key].mean(axis=1).min())
    for trace in comparison_voltage
    for key in ("v_e", "v_i")
)
voltage_scale_max = max(
    float(trace[key].mean(axis=1).max())
    for trace in comparison_voltage
    for key in ("v_e", "v_i")
)

# Border-light phase portrait using the same means as the conductance pistons.
phase_ax = ax.inset_axes([0.050, 0.045, 0.315, 0.305], transform=ax.transAxes)
phase_ax.set_facecolor("none")
phase_ax.set_box_aspect(0.75)
phase_ax.spines["top"].set_visible(False)
phase_ax.spines["right"].set_visible(False)
for spine in (phase_ax.spines["bottom"], phase_ax.spines["left"]):
    spine.set_color("#aaa59c")
    spine.set_linewidth(0.65)
phase_ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
phase_ax.set_xlabel(r"mean $g_E$", color="#625f59", fontsize=8, labelpad=2)
phase_ax.set_ylabel(r"mean $g_I$", color="#a62a24", fontsize=8, labelpad=2)
phase_ax.set_title(
    "conductance cycle", color="#4f4c47", fontsize=9, weight="medium", pad=3
)
comparison_mean_g_e = np.concatenate(
    [trace["g_e"].mean(axis=1) for trace in comparison_voltage]
)
comparison_mean_g_i = np.concatenate(
    [trace["g_i"].mean(axis=1) for trace in comparison_voltage]
)
g_e_pad = max(float(np.ptp(comparison_mean_g_e)) * 0.08, 1e-5)
g_i_pad = max(float(np.ptp(comparison_mean_g_i)) * 0.08, 1e-5)
phase_ax.set_xlim(
    float(comparison_mean_g_e.min() - g_e_pad),
    float(comparison_mean_g_e.max() + g_e_pad),
)
phase_ax.set_ylim(
    float(comparison_mean_g_i.min() - g_i_pad),
    float(comparison_mean_g_i.max() + g_i_pad),
)
(phase_point,) = phase_ax.plot(
    [],
    [],
    marker="o",
    markersize=4.5,
    color="#a62a24",
    markeredgecolor="#f3efe6",
    markeredgewidth=0.7,
    zorder=5,
)
phase_artists = []
phase_trail_steps = max(2, int(round(20.0 / dt)))

# Full-run spike raster: every input, E, and I spike remains visible from the
# first frame while a cursor alone marks the current animation time.
raster_duration_ms = n_steps * dt
raster_ax = ax.inset_axes([0.430, 0.045, 0.520, 0.305], transform=ax.transAxes)
raster_ax.set_facecolor("none")
for spine in ("top", "right", "left"):
    raster_ax.spines[spine].set_visible(False)
raster_ax.spines["bottom"].set_color("#aaa59c")
raster_ax.spines["bottom"].set_linewidth(0.65)
raster_ax.set_xlim(0.0, raster_duration_ms)
raster_ax.set_ylim(235, -1)
raster_ax.set_xticks([0, 25, 50, 75, 100])
raster_ax.set_yticks([49.5, 154.5, 222], labels=["Input", "E", "I"])
raster_ax.tick_params(axis="x", colors="#77726a", labelsize=6.5, length=2)
raster_ax.tick_params(axis="y", colors="#625f59", labelsize=6.5, length=0, pad=3)
raster_ax.set_xlabel("time (ms)", color="#77726a", fontsize=7, labelpad=2)
raster_ax.set_title(
    "spikes · complete 100 ms", color="#4f4c47", fontsize=9, weight="medium", pad=3
)
raster_input = raster_ax.scatter(
    [], [], s=2.2, color="#77726a", marker="s", linewidths=0
)
raster_e = raster_ax.scatter([], [], s=2.2, color="#20201e", marker="s", linewidths=0)
raster_i = raster_ax.scatter([], [], s=2.5, color="#a62a24", marker="s", linewidths=0)
for artist, spikes, offset in (
    (raster_input, input_spikes, 0),
    (raster_e, e_spikes, 105),
    (raster_i, i_spikes, 210),
):
    spike_times, cells = np.nonzero(spikes)
    artist.set_offsets(np.column_stack([spike_times * dt, cells + offset]))
raster_cursor = raster_ax.axvline(
    0.0, color="#a62a24", linewidth=0.9, alpha=0.72, zorder=4
)
raster_artists = [raster_input, raster_e, raster_i, raster_cursor]

active_e = ax.scatter(
    [], [], s=76, facecolors="none", edgecolors="#20201e", linewidths=1.8, zorder=5
)
active_i = ax.scatter(
    [], [], s=86, facecolors="none", edgecolors="#a62a24", linewidths=1.9, zorder=5
)
active_input = ax.scatter(
    [], [], s=42, facecolors="#77726a", edgecolors="#f3efe6", linewidths=0.8, zorder=5
)
arrow_artists = []


def voltage_sizes(voltage):
    level = np.clip((voltage - v_rest) / (v_threshold - v_rest), 0.0, 1.0)
    return 4.0 + 42.0 * np.power(level, 1.35)


def set_piston(piston, level, value_label):
    fill, head, value = piston
    y0, travel = fill._piston_y, fill._piston_travel
    level = float(np.clip(level, 0.0, 1.0))
    piston_y = y0 + travel * level
    fill.set_height(max(piston_y - y0, 0.001))
    head.set_y(piston_y)
    value.set_text(value_label)


def update_phase(step):
    while phase_artists:
        phase_artists.pop().remove()
    start = max(0, step - phase_trail_steps)
    points = np.column_stack([mean_g_e[start : step + 1], mean_g_i[start : step + 1]])
    if len(points) > 1:
        segments = np.stack([points[:-1], points[1:]], axis=1)
        rgba = np.tile(np.asarray(to_rgba("#20201e")), (len(segments), 1))
        rgba[:, 3] = np.linspace(0.04, 0.70, len(segments))
        trail = LineCollection(segments, colors=rgba, linewidths=1.15, zorder=3)
        phase_ax.add_collection(trail)
        phase_artists.append(trail)
    phase_point.set_data([mean_g_e[step]], [mean_g_i[step]])


def update_raster(step):
    current_time = step * dt
    raster_cursor.set_xdata([current_time, current_time])


def conductance_lines(values, segments, maximum, color):
    flat = values.reshape(-1)
    mask = flat > 1e-10
    if not np.any(mask):
        return None
    active = np.flatnonzero(mask)
    if len(active) > 320:
        strongest = active[np.argpartition(flat[active], -320)[-320:]]
        mask = np.zeros_like(mask)
        mask[strongest] = True
    strength = np.clip(flat[mask] / maximum, 0.0, 1.0)
    rgba = np.tile(np.asarray(to_rgba(color)), (len(strength), 1))
    rgba[:, 3] = 0.025 + 0.16 * np.sqrt(strength)
    collection = LineCollection(
        segments[mask],
        colors=rgba,
        linewidths=0.12 + 1.55 * strength,
        capstyle="round",
        zorder=3,
    )
    ax.add_collection(collection)
    return collection


def event_arrows(sources, targets, color):
    if len(sources) == 0:
        return []
    starts = np.repeat(np.asarray(sources), len(targets), axis=0)
    ends = np.tile(np.asarray(targets), (len(sources), 1))
    delta = ends - starts
    return [
        ax.quiver(
            starts[:, 0],
            starts[:, 1],
            delta[:, 0],
            delta[:, 1],
            angles="xy",
            scale_units="xy",
            scale=1,
            width=0.00034,
            headwidth=4.2,
            headlength=5.2,
            headaxislength=4.6,
            color=to_rgba(color, alpha=0.28),
            pivot="tail",
            zorder=4,
        )
    ]


def update(step):
    while arrow_artists:
        arrow_artists.pop().remove()

    e_sources = e_xy[e_by_t[step]] if e_by_t[step] else np.empty((0, 2))
    i_sources = i_xy[i_by_t[step]] if i_by_t[step] else np.empty((0, 2))
    input_ids = np.flatnonzero(input_spikes[step])
    input_sources = input_xy[input_ids]
    active_input.set_offsets(input_sources)
    active_e.set_offsets(e_sources)
    active_i.set_offsets(i_sources)
    e_nodes.set_sizes(voltage_sizes(v_e[step]))
    i_nodes.set_sizes(voltage_sizes(v_i[step]))
    voltage_span = max(voltage_scale_max - voltage_scale_min, 1e-12)
    set_piston(
        piston_ve,
        (mean_v_e[step] - voltage_scale_min) / voltage_span,
        f"{mean_v_e[step]:.1f}",
    )
    set_piston(
        piston_vi,
        (mean_v_i[step] - voltage_scale_min) / voltage_span,
        f"{mean_v_i[step]:.1f}",
    )
    set_piston(
        piston_ge, mean_g_e[step] / conductance_scale_max, f"{mean_g_e[step]:.3f}"
    )
    set_piston(
        piston_gi, mean_g_i[step] / conductance_scale_max, f"{mean_g_i[step]:.3f}"
    )
    update_phase(step)
    update_raster(step)

    e_conductance = conductance_lines(g_ei[step], e_segments, g_ei_max, "#20201e")
    i_conductance = conductance_lines(g_ie[step], i_segments, g_ie_max, "#a62a24")
    if e_conductance is not None:
        arrow_artists.append(e_conductance)
    if i_conductance is not None:
        arrow_artists.append(i_conductance)
    if len(input_sources):
        arrow_artists.extend(event_arrows(input_sources, e_xy, "#625f59"))
    if has_ei_projection:
        arrow_artists.extend(event_arrows(e_sources, i_xy, "#20201e"))
    if has_ie_projection:
        arrow_artists.extend(event_arrows(i_sources, e_xy, "#a62a24"))

    transmissions = (
        len(input_sources)
        + (len(e_sources) * n_i if has_ei_projection else 0)
        + (len(i_sources) * n_e if has_ie_projection else 0)
    )
    time_text.set_text(f"t = {step * dt:05.2f} ms   frame {step:03d}")
    count_text.set_text(
        f"{len(input_sources)} input · {len(e_sources)} E · {len(i_sources)} I spikes  ·  {transmissions} transmissions"
    )
    input_label.set_text(f"Input  ·  100  ·  {input_rate_hz[step]:.0f} Hz")
    current_time = rate_time_ms[step]
    current_rate = input_rate_hz[step]
    rate_elapsed.set_data(rate_time_ms[: step + 1], input_rate_hz[: step + 1])
    rate_cursor.set_data([current_time, current_time], [0.0, current_rate])
    rate_point.set_data([current_time], [current_rate])
    rate_value_text.set_text(f"{current_rate:.0f} Hz")
    return (
        input_nodes,
        e_nodes,
        i_nodes,
        active_input,
        active_e,
        active_i,
        time_text,
        count_text,
        input_label,
        *rate_artists,
        phase_point,
        *phase_artists,
        *raster_artists,
        *piston_artists,
        *arrow_artists,
    )


preview_frame = os.environ.get("PING_FRAME")
if preview_frame is not None:
    frame = int(preview_frame)
    update(frame)
    output = ROOT / "qa" / f"v29-{CONDITION}-frame-{frame:03d}.png"
    fig.savefig(output, dpi=160, facecolor=fig.get_facecolor())
else:
    animation = FuncAnimation(
        fig, update, frames=range(n_steps), interval=1000 / FPS, blit=False
    )
    output = ROOT / f"ping-network-dynamics-v29-{CONDITION}.mp4"
    animation.save(output, writer=FFMpegWriter(fps=FPS, bitrate=3000))
plt.close(fig)
print(output)
