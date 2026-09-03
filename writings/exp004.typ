#import "contents.typ": with-contents, with-numbered-equations
#import "dataset-template.typ": with-datasets
#let meta = (
  tags: ("txt", "v35.0.0"),
  title: "Parameters & Units",
  updated_at: "2026-08-29T00:00:00Z",
  created_at: "2026-05-14T00:00:00Z",
  description: "The unit system used throughout the codebase and the biophysical constants for the COBA / PING model.",
  collection: "snnsim-docs",
  order: 2,
)

#let body = [
  == Using this reference

  Use this page when setting a CLI flag, reading a saved configuration, or checking a neuron update. The legacy SNNSIM biophysical model uses *ms for time, mV for voltage, nF for capacitance, μS for conductance, nA for current, and Hz for rates*. A `--dt 1` means 1 ms, not 1 s. Names are not uniformly suffixed: `sim_ms`, `ref_ms_E`, `tau_gaba`, and `dt` all represent milliseconds.

  The values below describe library defaults or explicitly labelled examples, not a universal experiment recipe. CLI defaults, loaded configurations, and committed experiment recipes can differ. Graph bundles declare their own units and parameters; see #link("/exp105/")[Networks, signals, and parameters].

  == Quantities

  #table(
    columns: (auto, auto, auto, auto),
    align: (left, left, right, left),
    [*Quantity*], [*Unit*], [*Default or example*], [*Variable*],
    [Integration step], [ms], [0.25], [_dt_ / `--dt`],
    [Simulation length], [ms], [200 (CLI); 600 (Config)], [_sim_ms_ / `--t-ms`],
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
    [External drive], [μS], [0.0006 (Config baseline)], [_t_e_async_],
    [Max input rate], [Hz], [25], [_max_rate_hz_],
    [Population firing rate], [Hz], [20–80], [_r_E_, _r_I_],
    [Gamma frequency], [Hz], [30–80], [_f_0_],
  )

  == COBA / PING biophysical constants

  These are the default constants in `tools/snnsim/models.py` used by #link("/exp100/")[COBANet]. They are model choices, not universal biological constants. Capacitance and leak satisfy $tau_m = C_m / g_L$. Here $tau_m$ is the passive membrane time constant, $C_m$ the capacitance, and $g_L$ the leak conductance.

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

  Synapse time constants: $tau_"AMPA" = 2$ ms (excitation), $tau_"GABA" = 9$ ms (inhibition). These are decay times, not a sufficient timestep-stability criterion. Check timestep sensitivity for the intended input, recurrence, and measurement. `--tau-gaba` overrides the inhibitory decay; `--train-leak` allows bounded, per-neuron membrane time constants instead of fixed leak defaults.

  == Internal consistency

  The chosen units are self-consistent — no conversion factors appear in the integration code. Two equations carry the whole system.

  The membrane time constant is $tau_m = C_m \/ g_L$. With $C_m$ in nF and
  $g_L$ in μS,

  $ (tau_m)_["ms"] = (C_m)_["nF"] / (g_L)_[mu"S"] $

  so $C_m = 1$ nF and $g_L = 0.05$ μS give $tau_m = 20$ ms directly.

  The LIF voltage update is $dif V_m = (Delta t_"sim" \/ C_m)(-g_L (V_m - E_L) + I_"ext")$. With $Delta t_"sim"$ in ms, $C_m$ in nF, $V_m$ and $E_L$ in mV, $g_L$ in μS, and $I_"ext"$ in nA,

  $ (dif V_m)_["mV"] = (Delta t_"sim")_["ms"] / (C_m)_["nF"] dot (I_"ext")_["nA"] $

  because ms·nA / nF = mV exactly.

  Conductance-current products share the same ledger: $g(V_m - E)$ is μS × mV = nA, so synaptic currents fold into $I_"ext"$ alongside any direct input current without a scale factor.

  == Why not SI?

  Pure SI (F, S, V, A, s) forces every value to a large negative exponent — $C_m = 10^(-9)$ F, $g_L = 5 times 10^(-8)$ S, $Delta t_"sim" = 2.5 times 10^(-4)$ s. The neuroscience convention (ms, mV, nF, μS, nA) keeps every typical value between $10^(-3)$ and $10^2$, which makes numerical debugging and human intuition faster. Conversion to SI remains available as an independent check; the equations above show why no extra scale factor is needed in these units.
  == Checking a configuration

  + *Separate defaults from overrides.* Read the saved `config.json` and the recipe that supplied it. For CLI inference, explicit flags override inherited values; omitted fields fall back to defaults.
  + *Convert rates once.* With integration timestep $Delta t_"sim"$ in ms and input rate $r_"input"$ in Hz, a Bernoulli spike encoder uses event probability $p_"event" = r_"input" Delta t_"sim" / 1000$ per step. Here $p_"event"$ is dimensionless. Check that the probability is meaningful for the chosen rate and timestep.
  + *Check duration and counters.* The legacy path uses `int(t_ms / dt)` simulation steps. Avoid assuming a non-integral duration is preserved exactly. Refractory times are also discretised to steps.
  + *Check the stored weights.* Initialization means are on a summed-coupling scale; individual stored edges are fan-in normalised. Readout weights can use direct initialization instead. See #link("/exp006/#weight-init")[Weight init].

  #link("/exp011/")[Previous: SNNSIM command-line guide] · #link("/exp100/")[Next: COBANet]
]

#let body = with-datasets("exp004", (), body)
#let body = with-numbered-equations(body)
#let body = with-contents(body)
