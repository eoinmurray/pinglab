#let meta = (
  title: "Parameters & Units",
  date: "2026-05-14",
  description: "The unit system used throughout the codebase and the biophysical constants for the COBA / PING model.",
  collection: "documentation",
)

#let body = [
  All physical quantities in the codebase use the same unit system: *ms for time, mV for voltage, nF for capacitance, μS for conductance, nA for current, Hz for rates*. Time fields carry an explicit _\_ms_ suffix (_sim_ms_, _ref_ms_E_, _tau_gaba_); CLI flags follow the same convention (_--t-ms 600_). A Δt of 1 means 1 ms, not 1 s.

  == Quantities

  #table(
    columns: (auto, auto, auto, auto),
    align: (left, left, right, left),
    [*Quantity*], [*Unit*], [*Typical value*], [*Variable*],
    [Integration step], [ms], [0.25], [_dt_, _DT_MS_],
    [Simulation length], [ms], [600], [_sim_ms_],
    [Membrane time constant], [ms], [20 (E), 5 (I)], [_tau_m_E_, _tau_m_I_],
    [Refractory period], [ms], [3 (E), 1.5 (I)], [_ref_ms_E_, _ref_ms_I_],
    [AMPA decay], [ms], [2], [_tau_ampa_],
    [GABA decay], [ms], [9], [_tau_gaba_],
    [Resting / leak potential], [mV], [−65], [_E_L_],
    [Spike threshold], [mV], [−50], [_V_th_],
    [Reset potential], [mV], [−65], [_V_reset_],
    [AMPA reversal], [mV], [0], [_E_e_],
    [GABA reversal], [mV], [−80], [_E_i_],
    [Membrane capacitance], [nF], [1.0 (E), 0.5 (I)], [_C_m_E_, _C_m_I_],
    [Leak conductance], [μS], [0.05 (E), 0.1 (I)], [_g_L_E_, _g_L_I_],
    [External drive], [μS], [0.0006 (async), 0.003 (PING)], [_t_e_async_, _t_e_ping_],
    [Input current (CUBA)], [nA], [20 per spike], [_input_scale_],
    [CUBA weight std], [nA], [32], [_W_STD_CUBA_],
    [Max input rate], [Hz], [25], [_max_rate_hz_],
    [Population firing rate], [Hz], [20–80], [_r_E_, _r_I_],
    [Gamma frequency], [Hz], [30–80], [_f_0_],
  )

  == COBA / PING biophysical constants

  Used by #link("/ar003/")[COBA] and #link("/ar003/")[PING]. Values follow neuroscience conventions (cf. Dayan & Abbott, Gerstner _Neuronal Dynamics_); the E:I asymmetry in $tau_m, C_m, g_L, tau_"ref"$ produces the timescale separation that makes PING dynamics possible.

  #table(
    columns: (auto, auto, auto),
    align: (left, left, left),
    [*Parameter*], [*E population*], [*I population*],
    [$tau_m$ (ms)], [20], [5],
    [$C_m$ (nF)], [1.0], [0.5],
    [$g_L$ (µS)], [0.05], [0.1],
    [$tau_"ref"$ (ms)], [3], [1.5],
    [$E_L$ (mV)], [−65], [−65],
    [$V_"th"$ (mV)], [−50], [−50],
    [$V_"reset"$ (mV)], [−65], [−65],
    [$E_e$ (mV, reversal)], [0], [0],
    [$E_i$ (mV, reversal)], [−80], [−80],
  )

  Synapse time constants: $tau_"AMPA" = 2$ ms (excitation), $tau_"GABA" = 9$ ms (inhibition). These set the ceiling on PING's Δt-stability: once $Delta t gt.tilde tau_"GABA"$ the E→I→E loop cannot complete within one step.

  == Internal consistency

  The chosen units are self-consistent — no conversion factors appear in the integration code. Two equations carry the whole system.

  The membrane time constant is $tau = C \/ g$. With $C$ in nF and $g$ in μS,

  $ tau_["ms"] = (C_["nF"]) / (g_[mu"S"]) $

  so $C_m = 1$ nF and $g_L = 0.05$ μS give $tau_m = 20$ ms directly.

  The LIF voltage update is $dif v = (Delta t \/ C)(-g_L (v - E_L) + I)$. With $Delta t$ in ms, $C$ in nF, $v, E$ in mV, $g$ in μS, and $I$ in nA,

  $ dif v_["mV"] = (Delta t_["ms"]) / (C_["nF"]) dot I_["nA"] $

  because ms·nA / nF = mV exactly.

  Conductance-current products share the same ledger: $g(v - E)$ is μS × mV = nA, so synaptic currents fold into $I$ alongside any direct input current without a scale factor.

  == Why not SI?

  Pure SI (F, S, V, A, s) forces every value to a large negative exponent — $C_m = 10^(-9)$ F, $g_L = 5 times 10^(-8)$ S, $Delta t = 2.5 times 10^(-4)$ s. The neuroscience convention (ms, mV, nF, μS, nA) keeps every typical value between $10^(-3)$ and $10^2$, which makes numerical debugging and human intuition faster. The tradeoff is that readers have to trust the unit consistency rather than verify it by plugging into SI formulas — this page is here so that trust is auditable.
]
