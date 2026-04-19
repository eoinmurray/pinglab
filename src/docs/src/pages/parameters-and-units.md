---
layout: ../layouts/MarkdownLayout.astro
title: "Parameters & Units"
---

# Parameters & Units

All physical quantities in the codebase use the same unit system: **ms for time, mV for voltage, nF for capacitance, μS for conductance, nA for current, Hz for rates**. Time fields carry an explicit *_ms* suffix (*sim_ms*, *ref_ms_E*, *tau_gaba*); CLI flags follow the same convention (*--t-ms 600*). A Δt of 1 means 1 ms, not 1 s.

## Quantities

| Quantity | Unit | Typical value | Variable |
| --- | --- | ---: | --- |
| Integration step | ms | 0.25 | *dt*, *DT_MS* |
| Simulation length | ms | 600 | *sim_ms* |
| Membrane time constant | ms | 20 (E), 5 (I) | *tau_m_E*, *tau_m_I* |
| Refractory period | ms | 3 (E), 1.5 (I) | *ref_ms_E*, *ref_ms_I* |
| AMPA decay | ms | 2 | *tau_ampa* |
| GABA decay | ms | 9 | *tau_gaba* |
| Synaptic delay | ms | 1 | *delay_ei_ms*, *delay_ie_ms* |
| Resting / leak potential | mV | −65 | *E_L* |
| Spike threshold | mV | −50 | *V_th* |
| Reset potential | mV | −65 | *V_reset* |
| AMPA reversal | mV | 0 | *E_e* |
| GABA reversal | mV | −80 | *E_i* |
| Membrane capacitance | nF | 1.0 (E), 0.5 (I) | *C_m_E*, *C_m_I* |
| Leak conductance | μS | 0.05 (E), 0.1 (I) | *g_L_E*, *g_L_I* |
| External drive | μS | 0.0006 (async), 0.003 (PING) | *t_e_async*, *t_e_ping* |
| Input current (CUBA) | nA | 20 per spike | *input_scale* |
| CUBA weight std | nA | 32 | *W_STD_CUBA* |
| Max input rate | Hz | 25 | *max_rate_hz* |
| Population firing rate | Hz | 20–80 | *r_E*, *r_I* |
| Gamma frequency | Hz | 30–80 | *f_0* |

## COBA / PING biophysical constants

Used by [COBA](/models/#coba) and [PING](/models/#ping) on the [model ladder](/models/). Values follow neuroscience conventions (cf. Dayan & Abbott, Gerstner *Neuronal Dynamics*); the E:I asymmetry in $\tau_m, C_m, g_L, \tau_{\text{ref}}$ produces the timescale separation that makes PING dynamics possible.

| Parameter | E population | I population |
| --------- | ------------ | ------------ |
| $\tau_m$ (ms) | 20 | 5 |
| $C_m$ (nF) | 1.0 | 0.5 |
| $g_L$ (µS) | 0.05 | 0.1 |
| $\tau_{\text{ref}}$ (ms) | 3 | 1.5 |
| $E_L$ (mV) | −65 | −65 |
| $V_{\text{th}}$ (mV) | −50 | −50 |
| $V_{\text{reset}}$ (mV) | −65 | −65 |
| $E_e$ (mV, reversal) | 0 | 0 |
| $E_i$ (mV, reversal) | −80 | −80 |

Synapse time constants: $\tau_{\text{AMPA}} = 2$ ms (excitation), $\tau_{\text{GABA}} = 9$ ms (inhibition). These set the ceiling on PING's Δt-stability: once $\Delta t \gtrsim \tau_{\text{GABA}}$ the E→I→E loop cannot complete within one step.

## Internal consistency

The chosen units are self-consistent — no conversion factors appear in the integration code. Two equations carry the whole system.

The membrane time constant is $\tau = C / g$. With $C$ in nF and $g$ in μS,

$$
\tau_{[\text{ms}]} = \frac{C_{[\text{nF}]}}{g_{[\text{μS}]}}
$$

so $C_m = 1$ nF and $g_L = 0.05$ μS give $\tau_m = 20$ ms directly.

The LIF voltage update is $dv = (\Delta t / C)(-g_L(v - E_L) + I)$. With $\Delta t$ in ms, $C$ in nF, $v, E$ in mV, $g$ in μS, and $I$ in nA,

$$
dv_{[\text{mV}]} = \frac{\Delta t_{[\text{ms}]}}{C_{[\text{nF}]}} \cdot I_{[\text{nA}]}
$$

because ms·nA / nF = mV exactly.

Conductance-current products share the same ledger: $g(v - E)$ is μS × mV = nA, so synaptic currents fold into $I$ alongside any direct input current without a scale factor.

## Why not SI?

Pure SI (F, S, V, A, s) forces every value to a large negative exponent — $C_m = 10^{-9}$ F, $g_L = 5 \times 10^{-8}$ S, $\Delta t = 2.5 \times 10^{-4}$ s. The neuroscience convention (ms, mV, nF, μS, nA) keeps every typical value between $10^{-3}$ and $10^2$, which makes numerical debugging and human intuition faster. The tradeoff is that readers have to trust the unit consistency rather than verify it by plugging into SI formulas — this page is here so that trust is auditable.
