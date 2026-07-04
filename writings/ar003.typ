#let meta = (
  title: "COBANet",
  date: "2026-05-14",
  description: "Deriving the conductance-based (COBA) two-population neuron equations and their exact exponential-Euler discretization.",
  collection: "documentation",
)

#let body = [
  == 1. A conductance based neuron equation

  The membrane is a capacitor ($C_m$) pierced by ion channels in parallel. Conservation of charge (Kirchhoff) balances the capacitive current $C_m dif V\/dif t$ against the total ionic current:

  $ C_m (dif V) / (dif t) = -sum_"ion" I_"ion" quad (1) $

  Each channel passes an ohmic current — its conductance $g_"ion" >= 0$ times the *driving force* $(V - E_"ion")$, the distance of $V$ from the reversal potential $E_"ion"$ (where the channel's net current vanishes, set by the Nernst equilibrium):

  $ I_"ion" = g_"ion" (V - E_"ion") quad (2) $

  Because $g_"ion" >= 0$, the current's sign lives entirely in the driving force. Summing a leak ($g_L$, $E_L$) and synaptic conductances — excitatory ($g_e$, $E_e$), inhibitory ($g_i$, $E_i$) — gives the general conductance-based (COBA) neuron:

  $ C_m (dif V) / (dif t) = -g_L (V - E_L) - g_e (V - E_e) - g_i (V - E_i) quad (3) $

  == 2. The COBA model

  COBANet specialises (3) to two populations — E driven by excitation and inhibition, I by excitation only:

  $ C_m^E (dif V^E) / (dif t) = -g_L^E (V^E - E_L) - g_e^E (V^E - E_e) - g_i^E (V^E - E_i) quad (4) $

  $ C_m^I (dif V^I) / (dif t) = -g_L^I (V^I - E_L) - g_e^I (V^I - E_e) quad (5) $

  A neuron spikes at threshold $V_"th"$ and resets to $V_"reset"$ for a refractory period:

  $ s_(t+1) = bb(1)[V >= V_"th"], quad V <- V_"reset" "if" s_(t+1) = 1 "or refractory" quad (6) $

  Each synaptic conductance is an exponential trace driven by presynaptic spikes — each spike adds its full weight as an instantaneous jump, then the conductance decays with the channel time constant; there is no E→E connection:

  $ (dif g^E_e) / (dif t) = -(g^E_e) / (tau_"AMPA") + W_"in" sum_k delta(t - t^"inp"_k) quad (7) $

  $ (dif g^E_i) / (dif t) = -(g^E_i) / (tau_"GABA") + W_"ie" sum_k delta(t - t^i_k) quad (8) $

  $ (dif g^I_e) / (dif t) = -(g^I_e) / (tau_"AMPA") + W_"ei" sum_k delta(t - t^e_k) quad (9) $

  (7) is E's excitation from the input $W_"in"$; (8) its inhibition from I via $W_"ie"$; (9) the I population's excitation from E via $W_"ei"$.

  == 3. Discretization

  The conductances (7)–(9) and membrane equations (4)–(5) are continuous ODEs. The delta-driven conductances integrate *exactly* over one step: between spikes they decay by $e^(-Delta t \/ tau)$, and any spike landing in the step adds its full weight — the decay-then-add recurrence $g_(t+1) = e^(-Delta t \/ tau) g_t + W s_t$ (with the $tau$, $W$ and spike train $s$ of each of (7)–(9)). The membrane we integrate by *exponential Euler* — the same algebra for both populations (the I neuron drops $g_i$).

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

  Being the _exact_ frozen-conductance integral, (12) is *dt-invariant* — $N$ small steps equal one big step, so firing rates and the gamma frequency are physical (Hz) properties, not timestep artifacts (#link("/exp044/")[exp044]). A forward-Euler step $V_(t+1) = V_t + (Delta t \/ C_m) I_"net"(V_t)$ — not dt-invariant — is kept only as a parity toggle (_COBA_INTEGRATOR_).

  Each step runs in fixed order: conductances (7)–(9), then $g_"tot", tau_"eff", V_oo$, then the membrane step (12), then spike + reset (6). *The zero-order hold is this ordering* — the conductances advance once, then stay fixed while the membrane integrates across $Delta t$. E and I advance synchronously, phase-locking the E→I→E gamma cycle to the grid.
]
