---
title: docs.1-units
description: Units for MQIF neuron model parameters.
---
## MQIF back-of-the-envelope units

The MQIF update in code is:

$$
C_m \frac{dV}{dt} =
g_L (E_L - V)
\; + \;
\sum_k a_k (V - V_{r,k})^2
\; + \;
I_{ext}
\; + \;
g_e (E_e - V)
\; + \;
g_i (E_i - V).
$$

To keep units consistent, we pick a unit system and stick to it. A simple choice
matching the defaults in config is:

- $V, E_L, E_e, E_i, V_r, V_{th}$: mV
- $t, dt, \tau$: ms
- $C_m$: nF (or a unitless scale factor)
- $g_L, g_e, g_i$: nS (conductance)
- $I_{ext}$: nA (current)

Then each term on the right is a current (nA). Implying:

- $g_L (E_L - V)$: nS * mV = nA
- $g_e (E_e - V), g_i (E_i - V)$: nS * mV = nA
- $I_{ext}$: nA
- $a_k (V - V_{r,k})^2$: nA -> so $a_k$ has units nA / mV^2

### Picking MQIF parameters from the scale

We use the leak term as a reference scale. With the defaults:

- $C_m = 1.0$, $g_L = 0.05$, so $\tau_m = C_m / g_L = 20$ ms
- A typical leak current is $g_L (E_L - V) \approx 0.05 * 15 = 0.75$ nA

To keep the quadratic term comparable near spike onset, we choose $a_k$ so
$$a_k (V - V_r)^2 \sim 0.5 - 5\,\text{nA}.$$
If $|V - V_r| \approx 10$ mV, then
$$a_k \sim 0.005 - 0.05\,\text{nA}/\text{mV}^2.$$

This suggests a conservative starting point:

- $V_r$ around -60 to -45 mV
- $a_k$ around 0.01 to 0.03 nA/mV^2

### Using this to set weights

Synaptic currents should be the same order of magnitude as leak and quadratic
currents in the regime we want to probe. With $E_e \approx 0$ mV and
$E_i \approx -80$ mV, $|E_e - V|$ and $|E_i - V|$ are roughly 10-30 mV near
threshold. That gives a back-of-envelope target:

$$g_{syn} \cdot 20\,\text{mV} \sim 1\,\text{nA} \Rightarrow g_{syn} \sim 0.05\,\text{nS}.$$

So for MQIF scans, we dial the mean weights down if we add strong
quadratic terms, e.g.:

- keep $g_{ee}$ and $g_{ii}$ small (0.05 to 0.3)
- tune $g_{ei}$ and $g_{ie}$ so their synaptic currents stay in the 0.5-5 nA
  band (roughly 0.02 to 0.2 nS if we use the 20 mV rule of thumb)

