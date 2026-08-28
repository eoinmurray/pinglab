#import "contents.typ": with-contents
#import "run-view.typ": with-datasets
#let meta = (
  status: "[≡ TXT]",
  title: "COBANet",
  updated_at: "2026-08-28",
  date: "2026-05-14",
  description: "How COBANet updates conductances, membrane voltages, spikes, and resets, with the equations and implementation limits needed to work on the simulator.",
  collection: "snnsim-docs",
  order: 3,
)

#let body = [
  == Implementation map

  `COBANet` in `tools/snnsim/models.py` implements the built-in `--model ping` network. This page explains the minimal input→E→I→E motif; optional E→E, I→I, direct drives, adaptation, and trainable leak extend it. Read #link("/exp004/")[Parameters & Units] alongside these equations.

  #table(
    columns: (auto, 1fr),
    [*Code*], [*Responsibility*],
    [`exp_synapse`], [Decay the previous conductance, then add the current spike kick.],
    [`lif_step_expeuler`], [Integrate voltage under fixed conductances, then apply clamps, refractory gating, spike detection, and reset.],
    [`COBANet._step_body`], [Schedule population updates, recurrence, and readout.],
    [`set_sim_dt` in `config.py`], [Set timestep, duration, and derived step count for legacy execution.],
  )

  Reuse the execution interface rather than changing model globals around a live network. A small forward command is given in #link("/exp011/#quick-start")[Quick start].

  == A conductance based neuron equation

  The membrane is a capacitor ($C_m$) pierced by ion channels in parallel. Conservation of charge (Kirchhoff) balances the capacitive current $C_m dif V\/dif t$ against the total ionic current:

  $ C_m (dif V) / (dif t) = -sum_"ion" I_"ion" quad (1) $

  Each channel passes an ohmic current — its conductance $g_"ion" >= 0$ times the *driving force* $(V - E_"ion")$, the distance of $V$ from the reversal potential $E_"ion"$ (where the channel's net current vanishes, set by the Nernst equilibrium):

  $ I_"ion" = g_"ion" (V - E_"ion") quad (2) $

  Because $g_"ion" >= 0$, the current's sign lives entirely in the driving force. Summing a leak ($g_L$, $E_L$) and synaptic conductances — excitatory ($g_e$, $E_e$), inhibitory ($g_i$, $E_i$) — gives the general conductance-based (COBA) neuron:

  $ C_m (dif V) / (dif t) = -g_L (V - E_L) - g_e (V - E_e) - g_i (V - E_i) quad (3) $

  == The COBA model

  In the minimal PING motif, E receives excitation and inhibition, while I receives excitation only. These equations omit the optional I→I pathway:

  $ C_m^E (dif V^E) / (dif t) = -g_L^E (V^E - E_L) - g_e^E (V^E - E_e) - g_i^E (V^E - E_i) quad (4) $

  $ C_m^I (dif V^I) / (dif t) = -g_L^I (V^I - E_L) - g_e^I (V^I - E_e) quad (5) $

  After integration, a neuron outside its refractory period spikes at threshold $V_"th"$ and resets to $V_"reset"$. A refractory neuron cannot emit a spike:

  $ s_(t+1) = chi_(t+1) bb(1)[U_(t+1) >= V_"th"], quad V_(t+1) = cases(V_"reset" & "if spiking or refractory", U_(t+1) & "otherwise"). quad (6) $

  Here $U_(t+1)$ is the integrated candidate voltage and $chi_(t+1)$ is 1 when the refractory counter permits a spike, otherwise 0. Thresholding follows the voltage update, not the previous step's voltage.

  Each synaptic conductance is an exponential trace driven by presynaptic spikes — each spike adds its full weight as an instantaneous jump, then the conductance decays with the channel time constant; this minimal motif has no E→E connection:

  $ (dif g^E_e) / (dif t) = -(g^E_e) / (tau_"AMPA") + W_"in" sum_k delta(t - t^"inp"_k) quad (7) $

  $ (dif g^E_i) / (dif t) = -(g^E_i) / (tau_"GABA") + W_"ie" sum_k delta(t - t^i_k) quad (8) $

  $ (dif g^I_e) / (dif t) = -(g^I_e) / (tau_"AMPA") + W_"ei" sum_k delta(t - t^e_k) quad (9) $

  (7) is E's excitation from the input $W_"in"$; (8) its inhibition from I via $W_"ie"$; (9) the I population's excitation from E via $W_"ei"$.

  == Discretization

  The conductances (7)–(9) and membrane equations (4)–(5) are continuous ODEs. The implementation places spike kicks on the timestep grid. Between kicks the conductances decay by $e^(-Delta t \/ tau)$, and the supplied spike adds its full weight at the update boundary — the decay-then-add recurrence $g_(t+1) = e^(-Delta t \/ tau) g_t + s_t W$ (with the $tau$, $W$ and spike train $s$ of each of (7)–(9)). The membrane we integrate by *exponential Euler* — the same algebra for both populations (the I neuron drops $g_i$).

  Collecting on $V$ makes it linear, with total conductance $g_"tot" = g_L + g_e + g_i$:

  $ C_m (dif V) / (dif t) = -(g_L + g_e + g_i) V + (g_L E_L + g_e E_e + g_i E_i) quad (10) $

  Dividing by $g_"tot"$ gives decay-to-steady-state form, naming $tau_"eff" = C_m\/g_"tot"$ (shorter than $C_m\/g_L$ when synapses are open) and the steady-state voltage $V_oo$ (the conductance-weighted mean of the reversals):

  $ (C_m) / (g_"tot") (dif V) / (dif t) = -(V - (g_L E_L + g_e E_e + g_i E_i) / (g_"tot")) quad (11) $

  A *zero-order hold* freezes the conductances over one step $Delta t$, leaving (11) constant-coefficient with exact solution

  $ V_(t+1) = V_oo + (V_t - V_oo) e^(-Delta t \/ tau_"eff") quad (12) $

  Per population — I has no $g_i$, so its $g_"tot"$ and $V_oo$ drop those terms:

  $ g_"tot"^E = g_L^E + g_e^E + g_i^E, quad tau_"eff"^E = (C_m^E) / (g_"tot"^E), quad V_oo^E = (g_L^E E_L + g_e^E E_e + g_i^E E_i) / (g_"tot"^E) quad (13) $

  $ g_"tot"^I = g_L^I + g_e^I, quad tau_"eff"^I = (C_m^I) / (g_"tot"^I), quad V_oo^I = (g_L^I E_L + g_e^I E_e) / (g_"tot"^I) quad (14) $

  with step (12) for each population $p in {E, I}$: $V^p_(t+1) = V^p_oo + (V^p_t - V^p_oo) e^(-Delta t \/ tau^p_"eff")$.

  Equation (12) is exact only while those conductances are held fixed. In a passive interval without threshold events or clamps, subdividing that same fixed-conductance interval preserves the solution in exact arithmetic. It does not establish timestep invariance of a spiking network: conductance updates, threshold crossings, refractory counters, and recurrent scheduling still depend on the grid. Measure timestep sensitivity for the intended protocol rather than assuming firing rates or gamma frequency are invariant. The alternative `lif_step` uses forward Euler and is selected through `COBA_INTEGRATOR`.

  For each population update, conductances advance first, then the membrane integrates, then spike and reset are evaluated. This ordering defines the frozen-conductance interval. Recurrent inputs use the stored spikes supplied by `COBANet._step_body`; do not substitute a different same-step schedule while claiming equivalent dynamics.

  == Checking an implementation change

  + *Preserve the kick convention.* `exp_synapse` computes `g * decay + spikes @ W`; `(g + spikes @ W) * decay` attenuates every new kick and changes the model.
  + *Preserve spike timing.* Keep integration, refractory-counter update, thresholding, and reset in the implemented order. The returned voltage may already be reset while the emitted spike is still available.
  + *Separate forward and backward changes.* `--v-grad-dampen` modifies autograd through the increment; its local effect is described in #link("/exp015/")[Gradient Stabilisation].
  + *Test the intended extension.* A minimal PING equation does not describe enabled I→I recurrence, adaptation, noise, or state clamps. Inspect the corresponding branch in the model and extend the existing tests when changing it.

  Source reference: `tools/snnsim/models.py`, `tools/snnsim/config.py`, and `tools/snnsim/tests/test_models.py`.

  #link("/exp004/")[Previous: Parameters & Units] · #link("/exp006/")[Next: Training]
]

#let body = with-datasets("exp100", (), body)
#let body = with-contents(body)
