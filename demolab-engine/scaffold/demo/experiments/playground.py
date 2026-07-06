"""Interactive spiking-neuron playground — pick a model, drag the sliders, watch the
voltage trace and firing rate update.

The playground doesn't simulate anything itself: every slider settle re-invokes the
neuron tool's CLI — the *same* `neuron lif` / `neuron eif` commands the experiments run —
and reads back its data files. So the playground and the pipeline share one code path and
the science can't drift. Each run lands in temp/neuron/<model>/; we read <model>.csv (the
voltage trace) and output.json (the metrics) straight back.
"""
import csv
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "neuron" / "tool.py"
TEMP = ROOT / "temp" / "neuron"


def run_model(command: str, flags: dict) -> tuple[list, list, dict]:
    """Run a `neuron <command>` CLI with these flags, then read back its trace + metrics.

    `flags` maps CLI option names (without the leading `--`) to values, e.g.
    {"current": 2.5, "v-thresh": -50.0}. Both `lif` and `eif` write the same two files —
    <command>.csv (time_ms, voltage_mV) and output.json (firing_rate_hz, n_spikes, …) — so
    one reader handles either model.
    """
    argv = [sys.executable, str(TOOL), command]
    for name, value in flags.items():
        argv += [f"--{name}", str(value)]
    subprocess.run(argv, check=True, capture_output=True, text=True)

    run_dir = TEMP / command
    t, v = [], []
    with (run_dir / f"{command}.csv").open() as f:
        for row in csv.DictReader(f):
            t.append(float(row["time_ms"]))
            v.append(float(row["voltage_mV"]))
    metrics = json.loads((run_dir / "output.json").read_text())
    return t, v, metrics


st.set_page_config(page_title="Spiking neuron", layout="wide")
st.title("Spiking-neuron playground")
st.caption(
    "Pick a model, then drag any slider in the sidebar to update. Each change re-runs the "
    "same `neuron` CLI the experiments use and reads the trace back — no separate sim here."
)

with st.sidebar:
    st.header("Model")
    model = st.radio(
        "Neuron model",
        ["LIF — leaky integrate-and-fire", "EIF — exponential integrate-and-fire"],
        label_visibility="collapsed",
    )
    command = "eif" if model.startswith("EIF") else "lif"

    st.subheader("Input")
    i_tonic = st.slider("Tonic current I (nA)", 0.0, 6.0, 2.5, 0.05)
    r_m = st.slider("Membrane resistance R_m (MΩ)", 1.0, 30.0, 10.0, 0.5)

    st.subheader("Membrane")
    tau_m = st.slider("Time constant τ_m (ms)", 1.0, 50.0, 10.0, 0.5)
    v_rest = st.slider("Resting potential V_rest (mV)", -90.0, -40.0, -65.0, 0.5)
    v_reset = st.slider("Reset potential V_reset (mV)", -90.0, -40.0, -70.0, 0.5)

    # Shared flags, then the model-specific spiking mechanism.
    flags = {
        "current": i_tonic, "r-m": r_m, "tau-m": tau_m,
        "v-rest": v_rest, "v-reset": v_reset,
    }

    if command == "lif":
        st.subheader("Threshold")
        v_thresh = st.slider("Threshold V_thresh (mV)", -60.0, -20.0, -50.0, 0.5)
        flags["v-thresh"] = v_thresh
        cut_line, cut_label = v_thresh, "V_thresh"
    else:
        st.subheader("Spike initiation")
        v_t = st.slider("Soft threshold V_T (mV)", -60.0, -20.0, -50.0, 0.5)
        delta_t = st.slider("Slope factor Δ_T (mV)", 0.5, 10.0, 2.0, 0.5)
        v_peak = st.slider("Peak / cutoff V_peak (mV)", -20.0, 20.0, 0.0, 1.0)
        flags |= {"v-t": v_t, "delta-t": delta_t, "v-peak": v_peak}
        cut_line, cut_label = v_t, "V_T (soft)"

    st.subheader("Simulation")
    duration = st.slider("Duration (ms)", 20.0, 500.0, 100.0, 10.0)
    dt = st.slider("Timestep Δt (ms)", 0.01, 1.0, 0.1, 0.01)
    flags |= {"duration": duration, "dt": dt}

try:
    t, v, metrics = run_model(command, flags)
except subprocess.CalledProcessError as e:
    st.error(f"The `neuron {command}` command failed:")
    st.code(e.stderr or str(e))
    st.stop()

firing_rate_hz = metrics["firing_rate_hz"]
n_spikes = metrics["n_spikes"]

m1, m2, m3 = st.columns(3)
m1.metric("Model", command.upper())
m2.metric("Spikes", n_spikes)
m3.metric(
    "Firing rate",
    f"{firing_rate_hz:.1f} Hz",
    delta="spiking" if n_spikes else "silent",
    delta_color="normal" if n_spikes else "off",
)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(t, v, color="#1f77b4", linewidth=0.9)
ax.axhline(cut_line, color="#cc4444", linestyle="--", linewidth=0.8, label=cut_label)
ax.axhline(v_rest, color="#888", linestyle=":", linewidth=0.7, label="V_rest")
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Membrane potential (mV)")
ax.set_xlim(0, duration)
ax.set_ylim(min(v_reset, v_rest) - 5, max(10, max(v) + 5))
ax.legend(loc="lower right", fontsize=8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
st.pyplot(fig, clear_figure=True)

with st.expander("Equations"):
    if command == "lif":
        st.markdown(
            r"""
**Leaky integrate-and-fire.** Subthreshold dynamics:

$$\tau_m \frac{dV}{dt} = -(V - V_\text{rest}) + R_m\, I$$

Spike-and-reset rule:

$$V(t) \geq V_\text{thresh} \;\Longrightarrow\; \text{spike at } t,\quad V \leftarrow V_\text{reset}$$

The neuron fires periodically when the steady-state voltage $V_\infty = V_\text{rest} + R_m I$ sits above $V_\text{thresh}$.
            """
        )
    else:
        st.markdown(
            r"""
**Exponential integrate-and-fire.** An explicit exponential term replaces the hard threshold, modelling the upswing of a spike:

$$\tau_m \frac{dV}{dt} = -(V - V_\text{rest}) + \Delta_T \exp\!\left(\frac{V - V_T}{\Delta_T}\right) + R_m\, I$$

Peak-and-reset rule:

$$V(t) \geq V_\text{peak} \;\Longrightarrow\; \text{spike at } t,\quad V \leftarrow V_\text{reset}$$

$V_T$ is now a *soft* inflection point rather than a discontinuity; small $\Delta_T$ approaches LIF, larger $\Delta_T$ smooths spike initiation.
            """
        )
