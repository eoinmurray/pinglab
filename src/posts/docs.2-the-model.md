---
title: docs.2-the-model
description: The model and its parameters
---

# The model

## TD;DR

$$
C_m \frac{dV}{dt} = g_L(E_L - V) + g_e(E_e - V) + g_i(E_i - V) + I_{ext}
$$
$$
\frac{dg_e}{dt} = -\frac{g_e}{\tau_{ampa}} + \sum_{j \in E} w_{ij}\,\delta(t - t_j - d_{ee/ei}),\quad
\frac{dg_i}{dt} = -\frac{g_i}{\tau_{gaba}} + \sum_{j \in I} w_{ij}\,\delta(t - t_j - d_{ie/ii})
$$
$$
V \ge V_{th} \Rightarrow \text{spike},\; V \leftarrow V_{reset},\; \text{refractory for } t_{ref,E/I}
$$

Where:

- $V$: membrane voltage
- $C_m$: membrane capacitance
- $g_L, E_L$: leak conductance and leak reversal
- $g_e, E_e$: excitatory conductance and reversal (AMPA)
- $g_i, E_i$: inhibitory conductance and reversal (GABA)
- $I_{ext}$: external input current
- $\tau_{ampa}, \tau_{gaba}$: excitatory/inhibitory decay constants
- $w_{ij}$: synaptic weight from neuron $j$ to neuron $i$
- $d_{ee/ei}, d_{ie/ii}$: E/I pathway delays
- $V_{th}, V_{reset}$: spike threshold and reset voltage
- $t_{ref,E/I}$: refractory duration for E/I neurons

## LIF neuron model (as implemented)

The active simulator path in the UI uses the Leaky Integrate-and-Fire (LIF) model from `src/pinglab/src/pinglab/engine/neuron_models.py` and `src/pinglab/src/pinglab/models/lif.py`.

Dynamics (per neuron):

- **Voltage**
  $$
  C_m \frac{dV}{dt} = g_L(E_L - V) + g_e(E_e - V) + g_i(E_i - V) + I_{ext}
  $$

Spike and reset:

- A spike is detected when `V_new >= V_th` and the neuron is not refractory.
- After a spike, `V` is set to `V_reset`.
- Refractory handling is enforced in `run_network` via per-neuron `refractory_countdown` using `t_ref_E`/`t_ref_I`.

## Conductance‑based synapses (as implemented)

The network uses conductance‑based synapses with separate **excitatory** and **inhibitory** conductances (`g_e`, `g_i`):

- Conductance updates are applied by the connectivity layer each step.
- **AMPA** corresponds to excitatory (E→E, E→I) conductances; **GABA** corresponds to inhibitory (I→E, I→I) conductances.
- Conductances decay exponentially each step:
  $$
  g_e \leftarrow g_e \cdot e^{-\Delta t / \tau_{ampa}},\quad
  g_i \leftarrow g_i \cdot e^{-\Delta t / \tau_{gaba}}
  $$
- Synaptic currents are injected into each neuron model via:
  $$
  g_e(E_e - V) + g_i(E_i - V)
  $$

Synaptic delays (`delay_ee`, `delay_ei`, `delay_ie`, `delay_ii`) are implemented in the connectivity backend and shift the effect of spikes between population blocks.

### Synaptic delays (mathematical view)

Delays are applied **per connection block** (E→E, E→I, I→E, I→I). If neuron *j* spikes at time $t_k$, its effect on neuron *i* is delivered to the corresponding conductance *after* the block delay:

$$
g_{e,i}(t) \; \leftarrow \; g_{e,i}(t) + w_{ij}\,\delta(t - (t_k + d_{ee/ei}))
$$

$$
g_{i,i}(t) \; \leftarrow \; g_{i,i}(t) + w_{ij}\,\delta(t - (t_k + d_{ie/ii}))
$$

In discrete time (dt = $\Delta t$) the implementation schedules the update at a **future step**:

$$
g_{e,i}[n + \Delta n] \; {+}{=}\; w_{ij}, \quad \Delta n = \max\!\left(1,\left\lfloor \frac{d}{\Delta t} \right\rceil\right)
$$

where $d$ is the relevant block delay in ms. This means the synaptic effect is a time‑shifted input that then decays with $\tau_{ampa}$ or $\tau_{gaba}$ after it lands. The delays do **not** change the decay; they only shift when the conductance bump is injected.

Why include delays? They capture axonal/dendritic transmission time and synaptic processing latency, and they materially shape network timing. Even small delays can shift phase relationships between E and I populations and affect whether oscillations stabilize or collapse.

## Equation reference (quick summary)

LIF membrane dynamics:
$$
C_m \frac{dV}{dt} = g_L(E_L - V) + g_e(E_e - V) + g_i(E_i - V) + I_{ext}
$$
- $V$: membrane voltage.
- $C_m$: membrane capacitance.
- $g_L$: leak conductance.
- $E_L$: leak reversal potential.
- $g_e, g_i$: excitatory and inhibitory synaptic conductances.
- $E_e, E_i$: excitatory and inhibitory reversal potentials.
- $I_{ext}$: externally injected current (deterministic input + noise term in simulation).

Synaptic conductance decay (continuous form):
$$
\frac{dg_e}{dt} = -\frac{g_e}{\tau_{ampa}}, \qquad
\frac{dg_i}{dt} = -\frac{g_i}{\tau_{gaba}}
$$
- $\tau_{ampa}$: excitatory (AMPA) decay time constant.
- $\tau_{gaba}$: inhibitory (GABA) decay time constant.
- Negative sign: conductances decay toward zero in absence of incoming spikes.

Synaptic conductance decay (per-step update used in simulation):
$$
g_e \leftarrow g_e e^{-\Delta t/\tau_{ampa}}, \qquad
g_i \leftarrow g_i e^{-\Delta t/\tau_{gaba}}
$$
- $\Delta t$: simulation time step (`dt`).
- $\leftarrow$: in-place update each simulation step.
- Exponential factor: exact discrete-time decay for first-order linear conductance dynamics.

Delayed spike-to-conductance update (discrete time):
$$
g_{e,i}[n+\Delta n] {+}{=} w_{ij}, \qquad
\Delta n = \max\!\left(1,\left\lfloor \frac{d_{ee/ei}}{\Delta t} \right\rceil\right)
$$
$$
g_{i,i}[n+\Delta n] {+}{=} w_{ij}, \qquad
\Delta n = \max\!\left(1,\left\lfloor \frac{d_{ie/ii}}{\Delta t} \right\rceil\right)
$$
- $w_{ij}$: synaptic weight from presynaptic neuron $j$ to postsynaptic neuron $i$.
- $n$: current discrete time index.
- $\Delta n$: delay in integer steps.
- $d_{ee/ei}, d_{ie/ii}$: configured delay (ms) for E- and I-projecting blocks.
- ${+}{=}$: delayed conductance increment is accumulated when presynaptic spikes arrive.

Spike/reset/refractory rule:
$$
\text{if } V \ge V_{th} \text{ and not refractory: spike, then } V \leftarrow V_{reset}
$$
$$
\text{refractory duration} = t_{ref,E} \text{ (E neurons)} \;\; \text{or} \;\; t_{ref,I} \text{ (I neurons)}
$$
- $V_{th}$: spike threshold.
- $V_{reset}$: reset voltage after spike emission.
- $t_{ref,E}, t_{ref,I}$: refractory duration for E and I populations.
- During refractory period, the neuron is prevented from spiking again.

## Parameters and defaults

**Note:** `dt`, `T`, `N_E`, and `N_I` are required in `NetworkConfig` (no class defaults). API `/run` defaults are resolved through `pinglab.simulator.service.DEFAULT_CONFIG` (via `src/api/src/api/web_app.py`).

| Parameter | Description | Default |
| --- | --- | --- |
| `neuron_model` | Neuron model (`lif` or `mqif`) | `lif` |
| `dt` | Simulation time step (ms) | required |
| `T` | Simulation duration (ms) | required |
| `N_E` | Number of excitatory neurons | required |
| `N_I` | Number of inhibitory neurons | required |
| `seed` | RNG seed (network/heterogeneity) | `None` |
| `delay_ee` | E→E delay (ms) | `1.5` |
| `delay_ei` | E→I delay (ms) | `1.5` |
| `delay_ie` | I→E delay (ms) | `1.5` |
| `delay_ii` | I→I delay (ms) | `1.5` |
| `V_init` | Initial membrane potential (mV) | `-65.0` |
| `E_L` | Leak reversal potential (mV) | `-65.0` |
| `E_e` | Excitatory reversal potential (mV) | `0.0` |
| `E_i` | Inhibitory reversal potential (mV) | `-80.0` |
| `C_m_E` | E membrane capacitance | `1.0` |
| `g_L_E` | E leak conductance | `0.1` |
| `C_m_I` | I membrane capacitance | `1.0` |
| `g_L_I` | I leak conductance | `0.1` |
| `V_th` | Spike threshold (mV) | `-50.0` |
| `V_reset` | Reset potential (mV) | `-65.0` |
| `t_ref_E` | E refractory period (ms) | `3.0` |
| `t_ref_I` | I refractory period (ms) | `1.5` |
| `tau_ampa` | AMPA decay time constant (ms) | `5.0` |
| `tau_gaba` | GABA decay time constant (ms) | `10.0` |
| `mqif_a` | MQIF quadratic coefficients | `[]` |
| `mqif_Vr` | MQIF quadratic centers | `[]` |
| `mqif_w_a` | MQIF slow term coefficients | `[]` |
| `mqif_w_Vr` | MQIF slow term centers | `[]` |
| `mqif_w_tau` | MQIF slow term time constants | `[]` |
| `V_th_heterogeneity_sd` | Threshold SD (relative) | `0.0` |
| `g_L_heterogeneity_sd` | Leak conductance SD (relative) | `0.0` |
| `C_m_heterogeneity_sd` | Capacitance SD (relative) | `0.0` |
| `t_ref_heterogeneity_sd` | Refractory SD (relative) | `0.0` |
| `pulse_onset_ms` | Pulse onset time (ms) | `0.0` |
| `pulse_duration_ms` | Pulse duration (ms) | `0.0` |
| `pulse_interval_ms` | Pulse interval (ms) | `0.0` |
| `pulse_amplitude_E` | Pulse amplitude (E) | `0.0` |
| `pulse_amplitude_I` | Pulse amplitude (I) | `0.0` |

### Instruments defaults in `NetworkConfig`

| Parameter | Description | Default |
| --- | --- | --- |
| `variables` | Recorded variables | `["V", "g_e", "g_i"]` |
| `neuron_ids` | Specific neuron IDs to record | `None` |
| `downsample` | Recording downsample factor | `1` |
| `population_means` | Record E/I means | `False` |
| `all_neurons` | Record all neurons | `True` |
